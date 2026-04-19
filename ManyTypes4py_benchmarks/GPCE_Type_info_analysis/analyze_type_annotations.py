"""
Analyze type annotation quality in function parameters and return types.

Compares two directories:
  1. deepseek_3_run     — LLM-generated .py files (recursive, in numbered subdirs)
  2. deepseek_3_stub_run — LLM-generated .pyi stubs (flat)

For every function/method, each parameter annotation and return annotation is
classified as: concrete, exact_any, partial_any, or blank.

Usage:
    python analyze_type_annotations.py
"""

import ast
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

DIR_PY = PARENT_DIR / "deepseek_3_run"
DIR_STUB = PARENT_DIR / "deepseek_3_stub_run"

OUTPUT_JSON = SCRIPT_DIR / "type_annotation_analysis.json"


def contains_any(node: ast.expr) -> bool:
    """Return True if the AST annotation node references 'Any' anywhere."""
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and contains_any(child):
            return True
    return False


def is_exact_any(node: ast.expr) -> bool:
    """Return True if the annotation is exactly 'Any' (not nested)."""
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    return False


def classify_annotation(node) -> str:
    """Classify an annotation node: 'blank', 'exact_any', 'partial_any', or 'concrete'."""
    if node is None:
        return "blank"
    if is_exact_any(node):
        return "exact_any"
    if contains_any(node):
        return "partial_any"
    return "concrete"


def analyze_file(filepath: Path) -> dict:
    """Parse a single file and return annotation statistics."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return {"_syntax_error": f"Line {e.lineno}: {e.msg}"}

    stats = {
        "total_params": 0,
        "total_returns": 0,
        "params_blank": 0,
        "params_exact_any": 0,
        "params_partial_any": 0,
        "params_concrete": 0,
        "returns_blank": 0,
        "returns_exact_any": 0,
        "returns_partial_any": 0,
        "returns_concrete": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            stats["total_params"] += 1
            cat = classify_annotation(arg.annotation)
            stats[f"params_{cat}"] += 1

        if node.args.vararg:
            stats["total_params"] += 1
            cat = classify_annotation(node.args.vararg.annotation)
            stats[f"params_{cat}"] += 1

        if node.args.kwarg:
            stats["total_params"] += 1
            cat = classify_annotation(node.args.kwarg.annotation)
            stats[f"params_{cat}"] += 1

        stats["total_returns"] += 1
        cat = classify_annotation(node.returns)
        stats[f"returns_{cat}"] += 1

    return stats


def collect_files(directory: Path, extension: str, recursive: bool) -> dict[str, Path]:
    """Collect files keyed by base name (without extension)."""
    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    files = {}
    for p in directory.glob(pattern):
        stem = p.stem
        files[stem] = p
    return files


def aggregate(per_file: dict[str, dict]) -> dict:
    """Sum per-file stats into totals."""
    totals = {}
    for stats in per_file.values():
        if stats is None or "_syntax_error" in stats:
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    return totals


def print_summary(label: str, totals: dict, file_count: int, parse_errors: int) -> None:
    tp = totals.get("total_params", 0)
    tr = totals.get("total_returns", 0)
    total_slots = tp + tr

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Files analyzed:  {file_count}")
    print(f"  Parse errors:    {parse_errors}")
    print(f"  Total params:    {tp}")
    print(f"  Total returns:   {tr}")
    print(f"  Total slots:     {total_slots}")
    print()

    for kind, prefix in [("Parameters", "params"), ("Returns", "returns")]:
        total = totals.get(f"total_{'params' if prefix == 'params' else 'returns'}", 0)
        blank = totals.get(f"{prefix}_blank", 0)
        exact = totals.get(f"{prefix}_exact_any", 0)
        partial = totals.get(f"{prefix}_partial_any", 0)
        concrete = totals.get(f"{prefix}_concrete", 0)

        print(f"  {kind} (total={total}):")
        print(f"    Concrete:     {concrete:>6}  ({concrete / total * 100:.1f}%)" if total else f"    Concrete:     {concrete:>6}")
        print(f"    Exact Any:    {exact:>6}  ({exact / total * 100:.1f}%)" if total else f"    Exact Any:    {exact:>6}")
        print(f"    Partial Any:  {partial:>6}  ({partial / total * 100:.1f}%)" if total else f"    Partial Any:  {partial:>6}")
        print(f"    Blank:        {blank:>6}  ({blank / total * 100:.1f}%)" if total else f"    Blank:        {blank:>6}")
        print()


def run_analysis(directory: Path, extension: str, recursive: bool, label: str) -> dict:
    files = collect_files(directory, extension, recursive)
    per_file = {}
    parse_errors = {}
    for name, path in sorted(files.items()):
        result = analyze_file(path)
        if result is not None and "_syntax_error" in result:
            parse_errors[name] = result["_syntax_error"]
        per_file[name] = result

    totals = aggregate(per_file)
    file_count = sum(1 for v in per_file.values() if v is not None and "_syntax_error" not in v)
    print_summary(label, totals, file_count, len(parse_errors))

    if parse_errors:
        from collections import Counter
        error_types = Counter()
        for err_msg in parse_errors.values():
            msg_part = err_msg.split(": ", 1)[1] if ": " in err_msg else err_msg
            error_types[msg_part] += 1

        print(f"  Syntax error breakdown ({len(parse_errors)} files):")
        for msg, count in error_types.most_common():
            print(f"    {count:>4}x  {msg}")
        print()

    return {
        "label": label,
        "directory": str(directory),
        "files_analyzed": file_count,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "totals": totals,
        "per_file": per_file,
    }


def main() -> None:
    results = {}

    results["deepseek_3_run"] = run_analysis(
        DIR_PY, ".py", recursive=True,
        label="deepseek_3_run (.py files)"
    )

    results["deepseek_3_stub_run"] = run_analysis(
        DIR_STUB, ".pyi", recursive=False,
        label="deepseek_3_stub_run (.pyi stubs)"
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
