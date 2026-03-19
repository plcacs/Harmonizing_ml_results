"""
AST + hunk based per-error fix strategy analysis for mypy fixes.

This script avoids brittle exact line-content matching by combining:
1) hunk-level changed line mapping (difflib opcodes), and
2) AST scope change detection (function/class body and annotation deltas).
"""

import ast
import difflib
import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MYPY_DIR = os.path.join(PARENT_DIR, "mypy_results", "mypy_outputs")

MODELS = {
    "GPT-5": {
        "log": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "gpt5_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_gpt5_1st_run_with_errors.json"),
    },
    "DeepSeek": {
        "log": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "deep_seek_2nd_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_deepseek_2nd_run_with_errors.json"),
    },
    "Claude": {
        "log": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "claude3_sonnet_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "claude_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_claude3_sonnet_1st_run_with_errors.json"),
    },
}

ERROR_PATTERN = re.compile(r"^.+?:(\d+): error: .+\[(.+)\]$")

STRATEGY_ORDER = [
    "type_corrected",
    "cast_added",
    "type_ignore_added",
    "annotation_removed",
    "code_modified",
    "non_local_fix",
    "changed_to_any",
    "restructured",
    "other",
]


@dataclass(frozen=True)
class ScopeInfo:
    qname: str
    start: int
    end: int


def format_percent(count: int, total: int) -> str:
    return f"{(100.0 * count / total):.1f}%" if total else "0.0%"


def print_markdown_table(headers: List[str], rows: List[List[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")


def print_strategy_taxonomy_table(strategy_counter: Counter, total_errors: int) -> None:
    rows: List[List[object]] = []
    for strategy in STRATEGY_ORDER:
        count = strategy_counter.get(strategy, 0)
        if count == 0 and strategy in {"other"}:
            continue
        rows.append([strategy, count, format_percent(count, total_errors)])
    print_markdown_table(["Strategy", "Count", "% of errors"], rows)


def print_strategy_by_error_code_table(strategy_by_error_code: Dict[str, Counter], top_k: int = 10) -> None:
    if not strategy_by_error_code:
        return
    sorted_codes = sorted(
        strategy_by_error_code.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )[:top_k]
    headers = ["Error code", "Total"] + [s for s in STRATEGY_ORDER if s != "other"]
    rows: List[List[object]] = []
    for error_code, strategies in sorted_codes:
        total = sum(strategies.values())
        row: List[object] = [error_code, total]
        for strategy in headers[2:]:
            c = strategies.get(strategy, 0)
            row.append(f"{c} ({format_percent(c, total)})")
        rows.append(row)
    print_markdown_table(headers, rows)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def parse_errors(error_list: List[str]) -> List[Tuple[int, str]]:
    parsed: List[Tuple[int, str]] = []
    for err in error_list:
        m = ERROR_PATTERN.match(err)
        if m:
            parsed.append((int(m.group(1)), m.group(2)))
    return parsed


def find_file(directory: str, filename: str) -> Optional[str]:
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def build_line_map_and_changes(
    init_lines: List[str], fixed_lines: List[str]
) -> Tuple[Dict[int, int], Set[int], Set[int], List[Tuple[str, int, int, int, int]]]:
    """
    Build:
    - init->fixed line map for equal blocks (1-based line numbers),
    - changed line sets on each side,
    - raw opcodes from SequenceMatcher for local-hunk evidence extraction.
    """
    sm = difflib.SequenceMatcher(a=init_lines, b=fixed_lines)
    line_map: Dict[int, int] = {}
    changed_init: Set[int] = set()
    changed_fixed: Set[int] = set()
    opcodes = sm.get_opcodes()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for offset in range(i2 - i1):
                line_map[i1 + offset + 1] = j1 + offset + 1
            continue
        for i in range(i1, i2):
            changed_init.add(i + 1)
        for j in range(j1, j2):
            changed_fixed.add(j + 1)
    return line_map, changed_init, changed_fixed, opcodes


def ast_scopes_and_annotations(source: str) -> Tuple[List[ScopeInfo], Dict[str, str], Dict[str, int]]:
    scopes: List[ScopeInfo] = []
    body_dump_by_qname: Dict[str, str] = {}
    ann_count_by_qname: Dict[str, int] = {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return scopes, body_dump_by_qname, ann_count_by_qname

    def walk(node: ast.AST, stack: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = ".".join(stack + [child.name])
                start = getattr(child, "lineno", 1)
                end = getattr(child, "end_lineno", start)
                scopes.append(ScopeInfo(qname=qname, start=start, end=end))
                body_dump_by_qname[qname] = ast.dump(child, include_attributes=False)
                ann_count_by_qname[qname] = count_annotations(child)
                walk(child, stack + [child.name])
            else:
                walk(child, stack)

    walk(tree, [])
    return scopes, body_dump_by_qname, ann_count_by_qname


def count_annotations(node: ast.AST) -> int:
    total = 0
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
            if arg.annotation is not None and arg.arg != "self":
                total += 1
        if node.args.vararg and node.args.vararg.annotation is not None:
            total += 1
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            total += 1
        if node.returns is not None:
            total += 1
    for child in ast.walk(node):
        if isinstance(child, ast.AnnAssign):
            total += 1
    return total


def find_innermost_scope(line_no: int, scopes: List[ScopeInfo]) -> Optional[ScopeInfo]:
    hits = [s for s in scopes if s.start <= line_no <= s.end]
    if not hits:
        return None
    return min(hits, key=lambda s: s.end - s.start)


def collect_local_hunk_blobs(
    error_line: int,
    fixed_line: Optional[int],
    init_lines: List[str],
    fixed_lines: List[str],
    opcodes: List[Tuple[str, int, int, int, int]],
    init_scope: Optional[ScopeInfo],
    near_window: int = 12,
) -> Tuple[str, str, bool]:
    """
    Return local added/removed text near an error, instead of full-file blobs.
    """
    relevant: List[Tuple[str, int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        init_hit = i1 < error_line <= i2
        fixed_hit = fixed_line is not None and (j1 < fixed_line <= j2)
        if init_hit or fixed_hit:
            relevant.append((tag, i1, i2, j1, j2))

    if not relevant and init_scope:
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            scope_overlap = not (i2 < init_scope.start or i1 + 1 > init_scope.end)
            if scope_overlap:
                relevant.append((tag, i1, i2, j1, j2))

    near_change_used = False
    if not relevant:
        nearest: Optional[Tuple[str, int, int, int, int]] = None
        nearest_dist: Optional[int] = None
        for op in opcodes:
            tag, i1, i2, _, _ = op
            if tag == "equal":
                continue
            if i1 < error_line <= i2:
                dist = 0
            else:
                dist = min(abs(error_line - (i1 + 1)), abs(error_line - i2))
            if nearest_dist is None or dist < nearest_dist:
                nearest = op
                nearest_dist = dist
        if nearest is not None and nearest_dist is not None and nearest_dist <= near_window:
            relevant = [nearest]
            near_change_used = True

    added: List[str] = []
    removed: List[str] = []
    for _, i1, i2, j1, j2 in relevant:
        removed.extend(l.strip() for l in init_lines[i1:i2])
        added.extend(l.strip() for l in fixed_lines[j1:j2])
    return "\n".join(added), "\n".join(removed), near_change_used


def get_init_line_change_tag(
    error_line: int, opcodes: List[Tuple[str, int, int, int, int]]
) -> Optional[str]:
    for tag, i1, i2, _, _ in opcodes:
        if tag == "equal":
            continue
        if i1 < error_line <= i2:
            return tag
    return None


def classify_from_context(
    error_line: int,
    init_lines: List[str],
    fixed_lines: List[str],
    line_map: Dict[int, int],
    changed_init: Set[int],
    changed_fixed: Set[int],
    opcodes: List[Tuple[str, int, int, int, int]],
    init_scopes: List[ScopeInfo],
    changed_scopes: Set[str],
    annotation_delta: Dict[str, int],
    file_has_changes: bool,
) -> str:
    fixed_line = line_map.get(error_line)
    init_scope = find_innermost_scope(error_line, init_scopes)
    scope_changed = bool(init_scope and init_scope.qname in changed_scopes)
    scope_ann_delta = annotation_delta.get(init_scope.qname, 0) if init_scope else 0
    local_changed = error_line in changed_init or (fixed_line in changed_fixed if fixed_line else False)
    init_line_change_tag = get_init_line_change_tag(error_line, opcodes)

    added_blob, removed_blob, near_change_used = collect_local_hunk_blobs(
        error_line=error_line,
        fixed_line=fixed_line,
        init_lines=init_lines,
        fixed_lines=fixed_lines,
        opcodes=opcodes,
        init_scope=init_scope,
    )

    if init_line_change_tag == "delete":
        return "restructured"

    if not local_changed and not scope_changed:
        if near_change_used:
            return "code_modified"
        if file_has_changes:
            # Error line unchanged, but file changed elsewhere; likely indirect/contextual resolution.
            return "non_local_fix"
        return "unchanged"

    if "# type: ignore" in added_blob and "# type: ignore" not in removed_blob:
        return "type_ignore_added"
    if "cast(" in added_blob and "cast(" not in removed_blob:
        return "cast_added"
    if (": Any" in added_blob or "-> Any" in added_blob) and (": Any" not in removed_blob and "-> Any" not in removed_blob):
        return "changed_to_any"
    ann_token = r":\s*[A-Za-z_][\w\[\], .]*|->"
    has_added_ann = bool(re.search(ann_token, added_blob))
    has_removed_ann = bool(re.search(ann_token, removed_blob))
    if scope_ann_delta < 0 and has_removed_ann:
        return "annotation_removed"
    if scope_ann_delta > 0 and has_added_ann:
        return "type_corrected"

    if init_scope and scope_changed:
        return "code_modified"
    if local_changed:
        return "code_modified"
    return "other"


def analyze_model(model_name: str, config: Dict[str, str]) -> None:
    for path_key in ["log", "mypy_json"]:
        if not os.path.exists(config[path_key]):
            print(f"{model_name}: {path_key} not found at {config[path_key]}")
            return

    with open(config["log"], "r", encoding="utf-8") as f:
        log = json.load(f)
    with open(config["mypy_json"], "r", encoding="utf-8") as f:
        mypy_data = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v.get("status") == "fixed"}

    strategy_counter: Counter = Counter()
    strategy_by_error_code: Dict[str, Counter] = {}
    total_errors = 0
    files_analyzed = 0

    for filename in fixed_files:
        if filename not in mypy_data:
            continue
        errors = parse_errors(mypy_data[filename].get("errors", []))
        if not errors:
            continue

        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)
        if not initial_path or not os.path.exists(fixed_path):
            continue

        init_lines = read_lines(initial_path)
        fixed_lines = read_lines(fixed_path)
        init_source = "".join(init_lines)
        fixed_source = "".join(fixed_lines)

        line_map, changed_init, changed_fixed, opcodes = build_line_map_and_changes(
            init_lines, fixed_lines
        )
        file_has_changes = bool(changed_init or changed_fixed)
        init_scopes, init_scope_dump, init_ann = ast_scopes_and_annotations(init_source)
        _, fixed_scope_dump, fixed_ann = ast_scopes_and_annotations(fixed_source)

        changed_scopes: Set[str] = set()
        annotation_delta: Dict[str, int] = {}
        all_scopes = set(init_scope_dump.keys()) | set(fixed_scope_dump.keys())
        for qname in all_scopes:
            if init_scope_dump.get(qname) != fixed_scope_dump.get(qname):
                changed_scopes.add(qname)
            annotation_delta[qname] = fixed_ann.get(qname, 0) - init_ann.get(qname, 0)

        files_analyzed += 1
        for line_num, error_code in errors:
            total_errors += 1
            strategy = classify_from_context(
                error_line=line_num,
                init_lines=init_lines,
                fixed_lines=fixed_lines,
                line_map=line_map,
                changed_init=changed_init,
                changed_fixed=changed_fixed,
                opcodes=opcodes,
                init_scopes=init_scopes,
                changed_scopes=changed_scopes,
                annotation_delta=annotation_delta,
                file_has_changes=file_has_changes,
            )
            strategy_counter[strategy] += 1
            if error_code not in strategy_by_error_code:
                strategy_by_error_code[error_code] = Counter()
            strategy_by_error_code[error_code][strategy] += 1

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - AST/Hunk Error Fix Strategy Analysis")
    print(f"  Files analyzed: {files_analyzed}, Total errors: {total_errors}")
    print(f"{'=' * 60}")

    print("\n  --- Taxonomy of fix strategies ---")
    print_strategy_taxonomy_table(strategy_counter, total_errors)

    print("\n  --- Strategy by mypy error code (top 10) ---")
    print_strategy_by_error_code_table(strategy_by_error_code, top_k=10)

    unattributed = strategy_counter.get("unchanged", 0)
    if unattributed:
        print("\n  --- Unattributed cases (not in main taxonomy) ---")
        print(
            f"  unchanged/unattributed: {unattributed}/{total_errors} "
            f"({format_percent(unattributed, total_errors)})"
        )


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
        
