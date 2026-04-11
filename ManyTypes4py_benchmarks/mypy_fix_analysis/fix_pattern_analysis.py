"""
Fix Pattern Analysis (AST + difflib hybrid).

For each fixed file, compares original vs fixed using both AST structural
comparison and difflib line-level diffs. Categories:

  - param_annotation_change    : function parameter annotation added/changed/removed
  - return_type_fix            : function return annotation added/changed/removed
  - variable_reannotation      : variable annotation (AnnAssign) added/changed/removed
  - import_addition            : typing-related import added or expanded
  - import_removal             : typing-related import removed
  - cast_insertion             : cast() call added
  - none_guard_addition        : if x is not None / is None guard added
  - type_ignore_addition       : # type: ignore comment added
  - other_code_change          : any non-typing runtime change

Runs across GPT-5, DeepSeek, Claude fix directories.
"""

import ast
import difflib
import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

MODELS = {
    "GPT-5": {
        "log": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "gpt5_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fixed_files"),
    },
    "DeepSeek": {
        "log": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "deep_seek_2nd_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fixed_files"),
    },
    "Claude": {
        "log": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "claude3_sonnet_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "claude_mypy_fix", "fixed_files"),
    },
}

TYPING_MODULES = {"typing", "typing_extensions", "collections.abc"}
CAST_RE = re.compile(r"\bcast\s*\(")
NONE_GUARD_RE = re.compile(r"\bis\s+not\s+None\b|\bis\s+None\b")
TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore")

CATEGORIES = [
    "param_annotation_change",
    "return_type_fix",
    "variable_reannotation",
    "import_addition",
    "import_removal",
    "cast_insertion",
    "none_guard_addition",
    "type_ignore_addition",
    "other_code_change",
]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def safe_parse(source: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def ann_str(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def func_key(node: ast.AST, class_name: str = "") -> str:
    name = getattr(node, "name", "?")
    return f"{class_name}.{name}" if class_name else name


def extract_functions(tree: ast.AST) -> Dict[str, dict]:
    """Return {qualified_name: {params: {name: annotation_str}, return: str}}."""
    result = {}

    def _walk(body, class_name=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                _walk(node.body, node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = func_key(node, class_name)
                params = {}
                all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
                if node.args.vararg:
                    all_args.append(node.args.vararg)
                if node.args.kwarg:
                    all_args.append(node.args.kwarg)
                for arg in all_args:
                    params[arg.arg] = ann_str(arg.annotation)
                ret = ann_str(node.returns)
                result[key] = {"params": params, "return": ret}
                _walk(node.body, class_name)

    _walk(tree.body)
    return result


def extract_annotations(tree: ast.AST) -> Dict[str, str]:
    """Return {target_str: annotation_str} for module/class-level AnnAssign nodes."""
    result = {}

    def _walk(body):
        for node in body:
            if isinstance(node, ast.AnnAssign) and node.target:
                try:
                    target = ast.unparse(node.target)
                except Exception:
                    target = ast.dump(node.target)
                result[target] = ann_str(node.annotation)
            if isinstance(node, ast.ClassDef):
                _walk(node.body)

    _walk(tree.body)
    return result


def extract_typing_imports(tree: ast.AST) -> Set[str]:
    """Return set of 'from typing import X' / 'import typing' style strings."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module == m or node.module.startswith(m + ".")
                   for m in TYPING_MODULES):
                for alias in node.names:
                    imports.add(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == m or alias.name.startswith(m + ".")
                       for m in TYPING_MODULES):
                    imports.add(f"import {alias.name}")
    return imports


# ---------------------------------------------------------------------------
# Example snippet type
# ---------------------------------------------------------------------------

ExampleSnippet = Dict  # {"file": str, "category": str, "old_snippet": str, "new_snippet": str}

EXAMPLES_PER_CATEGORY = 2


# ---------------------------------------------------------------------------
# AST-level structural diff (with example collection)
# ---------------------------------------------------------------------------

def diff_functions(old_funcs: Dict, new_funcs: Dict,
                   filename: str, examples: Dict[str, List[ExampleSnippet]]) -> Counter:
    counts = Counter()
    all_keys = set(old_funcs) | set(new_funcs)
    for key in all_keys:
        old = old_funcs.get(key, {"params": {}, "return": ""})
        new = new_funcs.get(key, {"params": {}, "return": ""})

        all_params = set(old["params"]) | set(new["params"])
        for p in all_params:
            old_ann = old["params"].get(p, "")
            new_ann = new["params"].get(p, "")
            if old_ann != new_ann:
                counts["param_annotation_change"] += 1
                _collect(examples, "param_annotation_change", filename,
                         f"def {key}(..., {p}: {old_ann or '<none>'}, ...)",
                         f"def {key}(..., {p}: {new_ann or '<none>'}, ...)")

        if old["return"] != new["return"]:
            counts["return_type_fix"] += 1
            _collect(examples, "return_type_fix", filename,
                     f"def {key}(...) -> {old['return'] or '<none>'}",
                     f"def {key}(...) -> {new['return'] or '<none>'}")

    return counts


def diff_annotations(old_anns: Dict[str, str], new_anns: Dict[str, str],
                     filename: str, examples: Dict[str, List[ExampleSnippet]]) -> Counter:
    counts = Counter()
    all_keys = set(old_anns) | set(new_anns)
    for key in all_keys:
        old_val = old_anns.get(key, "")
        new_val = new_anns.get(key, "")
        if old_val != new_val:
            counts["variable_reannotation"] += 1
            old_text = f"{key}: {old_val}" if old_val else f"# {key} not annotated"
            new_text = f"{key}: {new_val}" if new_val else f"# {key} annotation removed"
            _collect(examples, "variable_reannotation", filename, old_text, new_text)
    return counts


def diff_imports(old_imports: Set[str], new_imports: Set[str],
                 filename: str, examples: Dict[str, List[ExampleSnippet]]) -> Counter:
    counts = Counter()
    added = new_imports - old_imports
    removed = old_imports - new_imports
    counts["import_addition"] = len(added)
    counts["import_removal"] = len(removed)
    for imp in sorted(added):
        _collect(examples, "import_addition", filename, "# (not present)", imp)
    for imp in sorted(removed):
        _collect(examples, "import_removal", filename, imp, "# (removed)")
    return counts


def _collect(examples: Dict[str, List[ExampleSnippet]], category: str,
             filename: str, old_snippet: str, new_snippet: str):
    if len(examples.get(category, [])) < EXAMPLES_PER_CATEGORY:
        examples.setdefault(category, []).append({
            "file": filename,
            "category": category,
            "old_snippet": old_snippet,
            "new_snippet": new_snippet,
        })


# ---------------------------------------------------------------------------
# Line-level diff for runtime / non-structural changes
# ---------------------------------------------------------------------------

def diff_lines_for_runtime(old_lines: List[str], new_lines: List[str],
                           filename: str, examples: Dict[str, List[ExampleSnippet]]) -> Counter:
    """Classify added lines that AST diffing can't catch (cast, None-guard, type:ignore)."""
    counts = Counter()
    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes()

    old_set = set(l.strip() for l in old_lines)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        old_chunk = [old_lines[i].rstrip("\n") for i in range(i1, i2)]
        new_chunk = [new_lines[j].rstrip("\n") for j in range(j1, j2)]
        old_context = "\n".join(old_chunk) if old_chunk else "# (no old lines)"
        new_context = "\n".join(new_chunk) if new_chunk else "# (no new lines)"

        added_lines = []
        if tag in ("replace", "insert"):
            added_lines = [new_lines[j].strip() for j in range(j1, j2)]

        for line in added_lines:
            if line in old_set:
                continue
            if CAST_RE.search(line):
                counts["cast_insertion"] += 1
                _collect(examples, "cast_insertion", filename, old_context, new_context)
            elif NONE_GUARD_RE.search(line):
                counts["none_guard_addition"] += 1
                _collect(examples, "none_guard_addition", filename, old_context, new_context)
            elif TYPE_IGNORE_RE.search(line):
                counts["type_ignore_addition"] += 1
                _collect(examples, "type_ignore_addition", filename, old_context, new_context)

    return counts


def count_other_changes(old_lines: List[str], new_lines: List[str],
                        ast_total: int,
                        filename: str, examples: Dict[str, List[ExampleSnippet]]) -> int:
    """Total changed lines minus AST-attributed changes = other."""
    opcodes = difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes()
    total_changed = 0
    collected_other = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        total_changed += max(i2 - i1, j2 - j1)
        if not collected_other and len(examples.get("other_code_change", [])) < EXAMPLES_PER_CATEGORY:
            old_chunk = "\n".join(old_lines[i].rstrip("\n") for i in range(i1, i2))
            new_chunk = "\n".join(new_lines[j].rstrip("\n") for j in range(j1, j2))
            if old_chunk.strip() != new_chunk.strip():
                _collect(examples, "other_code_change", filename,
                         old_chunk or "# (deleted)", new_chunk or "# (inserted)")
                collected_other = True
    return max(0, total_changed - ast_total)


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyze_file(init_path: str, fixed_path: str, filename: str,
                 examples: Dict[str, List[ExampleSnippet]]) -> Optional[Dict[str, int]]:
    with open(init_path, "r", encoding="utf-8", errors="ignore") as f:
        old_source = f.read()
    with open(fixed_path, "r", encoding="utf-8", errors="ignore") as f:
        new_source = f.read()

    if old_source == new_source:
        return None

    old_tree = safe_parse(old_source)
    new_tree = safe_parse(new_source)

    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)

    counts = Counter()

    if old_tree and new_tree:
        counts += diff_functions(extract_functions(old_tree), extract_functions(new_tree),
                                 filename, examples)
        counts += diff_annotations(extract_annotations(old_tree), extract_annotations(new_tree),
                                   filename, examples)
        counts += diff_imports(extract_typing_imports(old_tree), extract_typing_imports(new_tree),
                               filename, examples)

    counts += diff_lines_for_runtime(old_lines, new_lines, filename, examples)

    ast_attributed = sum(counts.values())
    other = count_other_changes(old_lines, new_lines, ast_attributed, filename, examples)
    if other > 0:
        counts["other_code_change"] = other

    return {k: v for k, v in counts.items() if v > 0}


# ---------------------------------------------------------------------------
# Model-level driver
# ---------------------------------------------------------------------------

def find_file(directory: str, filename: str) -> Optional[str]:
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def analyze_model(model_name: str, config: Dict[str, str]) -> Optional[Dict]:
    log_path = config["log"]
    if not os.path.exists(log_path):
        print(f"{model_name}: fix log not found at {log_path}")
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v.get("status") == "fixed"}
    print(f"\n{'='*60}")
    print(f"  {model_name}: {len(fixed_files)} fixed files")
    print(f"{'='*60}")

    totals = Counter()
    per_file = {}
    examples: Dict[str, List[ExampleSnippet]] = {}
    files_analyzed = 0

    for filename in sorted(fixed_files):
        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)
        if not initial_path or not os.path.exists(fixed_path):
            continue

        file_counts = analyze_file(initial_path, fixed_path, filename, examples)
        if not file_counts:
            continue

        files_analyzed += 1
        per_file[filename] = file_counts
        for cat, cnt in file_counts.items():
            totals[cat] += cnt

    total_changes = sum(totals.values())
    print(f"  Files analyzed: {files_analyzed}")
    print(f"  Total changes:  {total_changes}\n")

    print(f"  {'Category':<30} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*30} {'-'*6}  {'-'*6}")
    for cat in CATEGORIES:
        cnt = totals.get(cat, 0)
        pct = f"{100 * cnt / total_changes:.1f}" if total_changes else "0.0"
        print(f"  {cat:<30} {cnt:>6}  {pct:>5}%")

    return {
        "model": model_name,
        "files_analyzed": files_analyzed,
        "total_changes": total_changes,
        "totals": dict(totals),
        "per_file": per_file,
        "examples": examples,
    }


def main():
    output_dir = os.path.join(BASE_DIR, "fix_pattern_results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for model_name, config in MODELS.items():
        result = analyze_model(model_name, config)
        if result:
            all_results[model_name] = result
            out_path = os.path.join(
                output_dir,
                f"fix_patterns_{model_name.lower().replace('-', '_')}.json",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  -> Saved to {out_path}")

    summary_path = os.path.join(output_dir, "fix_pattern_summary.json")
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "files_analyzed": res["files_analyzed"],
            "total_changes": res["total_changes"],
            "totals": res["totals"],
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    examples_all = {}
    for name, res in all_results.items():
        examples_all[name] = res.get("examples", {})
    examples_path = os.path.join(output_dir, "fix_pattern_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples_all, f, indent=2, ensure_ascii=False)
    print(f"  Examples saved to {examples_path}")


if __name__ == "__main__":
    main()
