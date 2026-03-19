"""
Extract one concrete BEFORE/AFTER evidence snippet per fix strategy.

This is intended for paper figures/appendix. It reuses the same AST + hunk mapping
strategy logic as `error_fix_strategy_ast_hunk_analysis.py`, but instead of only
counting, it prints one example per strategy.
"""

import json
import os
from collections import Counter
from typing import Dict, Optional, List, Tuple

from error_fix_strategy_ast_hunk_analysis import (
    MODELS,
    parse_errors,
    find_file,
    read_lines,
    build_line_map_and_changes,
    ast_scopes_and_annotations,
    find_innermost_scope,  # type: ignore[attr-defined]
    classify_from_context,  # type: ignore[attr-defined]
    collect_local_hunk_blobs,  # type: ignore[attr-defined]
)


# Strategies requested by the user / those shown in taxonomy tables.
TARGET_STRATEGIES = [
    "type_corrected",
    "cast_added",
    "type_ignore_added",
    "annotation_removed",
    "code_modified",
    "non_local_fix",
    "changed_to_any",
    "restructured",
]


def _join_lines(lines: List[str]) -> str:
    # Keep existing line content (but drop trailing newlines for display).
    return "\n".join(l.rstrip("\n") for l in lines if l is not None)


def main() -> None:
    model_name = "GPT-5"
    config = MODELS[model_name]

    for path_key in ["log", "mypy_json"]:
        if not os.path.exists(config[path_key]):
            raise FileNotFoundError(f"{model_name}: {path_key} not found at {config[path_key]}")

    with open(config["log"], "r", encoding="utf-8") as f:
        log = json.load(f)

    with open(config["mypy_json"], "r", encoding="utf-8") as f:
        mypy_data = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v.get("status") == "fixed"}
    found: Dict[str, dict] = {}

    for filename in fixed_files:
        if len(found) >= len(TARGET_STRATEGIES):
            break
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

        line_map, changed_init, changed_fixed, opcodes = build_line_map_and_changes(init_lines, fixed_lines)
        file_has_changes = bool(changed_init or changed_fixed)

        init_scopes, init_scope_dump, init_ann = ast_scopes_and_annotations(init_source)
        _, fixed_scope_dump, fixed_ann = ast_scopes_and_annotations(fixed_source)

        changed_scopes: set = set()
        annotation_delta: Dict[str, int] = {}
        all_scopes = set(init_scope_dump.keys()) | set(fixed_scope_dump.keys())
        for qname in all_scopes:
            if init_scope_dump.get(qname) != fixed_scope_dump.get(qname):
                changed_scopes.add(qname)
            annotation_delta[qname] = fixed_ann.get(qname, 0) - init_ann.get(qname, 0)

        for error_line, error_code in errors:
            if len(found) >= len(TARGET_STRATEGIES):
                break

            # classify_from_context returns a strategy label for this error line
            strategy = classify_from_context(
                error_line=error_line,
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

            if strategy not in TARGET_STRATEGIES:
                continue
            if strategy in found:
                continue

            fixed_line = line_map.get(error_line)
            init_scope = find_innermost_scope(error_line, init_scopes)
            added_blob, removed_blob, _near_change_used = collect_local_hunk_blobs(
                error_line=error_line,
                fixed_line=fixed_line,
                init_lines=init_lines,
                fixed_lines=fixed_lines,
                opcodes=opcodes,
                init_scope=init_scope,
            )

            init_text = init_lines[error_line - 1].rstrip("\n") if 1 <= error_line <= len(init_lines) else ""

            # For changed lines, line_map won't have an entry.
            # Walk opcodes to find the replaced line in the fixed file.
            if fixed_line and 1 <= fixed_line <= len(fixed_lines):
                fixed_text = fixed_lines[fixed_line - 1].rstrip("\n")
            else:
                fixed_text = ""
                idx = error_line - 1  # 0-based
                for tag, i1, i2, j1, j2 in opcodes:
                    if tag == "equal":
                        continue
                    if i1 <= idx < i2:
                        rel = idx - i1
                        mapped_j = j1 + min(rel, max(0, (j2 - j1) - 1))
                        if 0 <= mapped_j < len(fixed_lines):
                            fixed_text = fixed_lines[mapped_j].rstrip("\n")
                        break

            # For non_local_fix: error line unchanged, fix happened elsewhere.
            # Prefer a type-related hunk; fall back to the first hunk.
            if strategy == "non_local_fix" and not added_blob and not removed_blob:
                TYPE_KEYWORDS = {"Any", "Optional", "Union", "List", "Dict", "cast",
                                 "type: ignore", "Callable", "Tuple", "Set", "Type", "->", ": "}
                best_removed, best_added = "", ""
                fallback_removed, fallback_added = "", ""
                for tag, i1, i2, j1, j2 in opcodes:
                    if tag == "equal":
                        continue
                    r = _join_lines(init_lines[i1:i2]).strip()
                    a = _join_lines(fixed_lines[j1:j2]).strip()
                    if not fallback_removed and not fallback_added:
                        fallback_removed, fallback_added = r, a
                    if any(kw in r or kw in a for kw in TYPE_KEYWORDS):
                        best_removed, best_added = r, a
                        break
                removed_blob = best_removed or fallback_removed
                added_blob = best_added or fallback_added

            found[strategy] = {
                "file": filename,
                "error_line": error_line,
                "error_code": error_code,
                "before_line": init_text.strip(),
                "after_line": fixed_text.strip(),
                "removed_blob": removed_blob.strip(),
                "added_blob": added_blob.strip(),
            }

    # Print evidence.
    missing = [s for s in TARGET_STRATEGIES if s not in found]
    if missing:
        print(f"WARNING: missing evidence for: {missing}")

    for s in TARGET_STRATEGIES:
        ex = found.get(s)
        if not ex:
            continue
        print("=" * 80)
        print(f"Strategy: {s}")
        print(f"File: {ex['file']}")
        print(f"mypy error: line {ex['error_line']}  code [{ex['error_code']}]")
        print("BEFORE (error line):")
        print(ex["before_line"])
        print("AFTER (mapped line, if any):")
        print(ex["after_line"])
        if ex["removed_blob"]:
            print("\nRemoved hunk (evidence):")
            print(ex["removed_blob"])
        if ex["added_blob"]:
            print("\nAdded hunk (evidence):")
            print(ex["added_blob"])
        print()


if __name__ == "__main__":
    main()

