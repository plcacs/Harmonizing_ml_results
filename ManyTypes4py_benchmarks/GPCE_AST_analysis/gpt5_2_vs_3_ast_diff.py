"""
Compare two run directories by AST: function and class names.
Usage: python gpt5_2_vs_3_ast_diff.py RUN_A RUN_B [--output OUT.json]
Example: python gpt5_2_vs_3_ast_diff.py ../untyped_benchmarks ../gpt5_2_run -o untyped_vs_gpt5_2.json
Example: python gpt5_2_vs_3_ast_diff.py ../untyped_benchmarks ../claude3_sonnet_3_run -o untyped_vs_claude3_sonnet_3.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def iter_py_files(root: str):
    """Yield (rel_path, abs_path) for each .py under root."""
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root)
            yield rel.replace("\\", "/"), abs_path


def collect_names(source: str) -> tuple[list[str], list[str], bool]:
    """
    Parse source and return (function_names, class_names, parse_ok).
    Includes nested functions/classes (names only, no scoping).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], [], False
    funcs: list[str] = []
    classes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return funcs, classes, True


def run_name(path: str) -> str:
    """Short name for a run (e.g. 'gpt5_2_run', 'untyped_benchmarks')."""
    return os.path.basename(os.path.normpath(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two run dirs by AST (function/class names). Writes JSON with diffs."
    )
    parser.add_argument(
        "run_a",
        help="First run directory (e.g. untyped_benchmarks or path to it)",
    )
    parser.add_argument(
        "run_b",
        help="Second run directory (e.g. gpt5_2_run or claude3_sonnet_3_run)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: <run_a_name>_vs_<run_b_name>_ast_diff.json in script dir)",
    )
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    if not os.path.isdir(run_a):
        raise SystemExit(f"Not a directory: {run_a}")
    if not os.path.isdir(run_b):
        raise SystemExit(f"Not a directory: {run_b}")

    name_a = run_name(run_a)
    name_b = run_name(run_b)
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        out_path = os.path.join(SCRIPT_DIR, f"{name_a}_vs_{name_b}_ast_diff.json")

    # Run A: index by rel_path and by basename (so flat run_a works when run_b has subdirs)
    run_a_by_path = {}
    run_a_by_basename: dict[str, str] = {}
    for rel, path in iter_py_files(run_a):
        run_a_by_path[rel] = path
        basename = os.path.basename(rel)
        if basename not in run_a_by_basename:
            run_a_by_basename[basename] = path

    run_b_files = {rel: path for rel, path in iter_py_files(run_b)}
    results: list[dict] = []
    total = len(run_b_files)
    for idx, (rel_path, path_b) in enumerate(run_b_files.items()):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"Processed {idx + 1}/{total} files...")
        path_a = run_a_by_path.get(rel_path) or run_a_by_basename.get(os.path.basename(rel_path))
        if not path_a or not os.path.isfile(path_a):
            continue
        with open(path_a, "r", encoding="utf-8", errors="replace") as f:
            code_a = f.read()
        with open(path_b, "r", encoding="utf-8", errors="replace") as f:
            code_b = f.read()
        funcs_a, classes_a, ok_a = collect_names(code_a)
        funcs_b, classes_b, ok_b = collect_names(code_b)
        if not ok_a:
            results.append({
                "file": rel_path,
                "skipped": True,
                "reason": f"SyntaxError in {name_a}",
            })
            continue
        if not ok_b:
            results.append({
                "file": rel_path,
                "skipped": True,
                "reason": f"SyntaxError in {name_b}",
            })
            continue
        set_fa, set_fb = set(funcs_a), set(funcs_b)
        set_ca, set_cb = set(classes_a), set(classes_b)
        only_fa = sorted(set_fa - set_fb)
        only_fb = sorted(set_fb - set_fa)
        only_ca = sorted(set_ca - set_cb)
        only_cb = sorted(set_cb - set_ca)
        if only_fa or only_fb or only_ca or only_cb:
            results.append({
                "file": rel_path,
                f"functions_only_in_{name_a}": only_fa,
                f"functions_only_in_{name_b}": only_fb,
                f"classes_only_in_{name_a}": only_ca,
                f"classes_only_in_{name_b}": only_cb,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
