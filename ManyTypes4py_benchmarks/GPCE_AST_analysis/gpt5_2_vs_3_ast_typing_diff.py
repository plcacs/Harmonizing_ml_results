"""
Compare run directories by AST: function/class names and typing-related calls.

Usage:
  # Two runs (as before)
  python gpt5_2_vs_3_ast_typing_diff.py RUN_A RUN_B [--output OUT.json]

  # Three runs (e.g. untyped baseline + two typed runs)
  python gpt5_2_vs_3_ast_typing_diff.py UNTYPED RUN_A RUN_B [--output OUT.json]

Examples:
  python gpt5_2_vs_3_ast_typing_diff.py ../untyped_benchmarks ../gpt5_2_run -o untyped_vs_gpt5_2_typing.json
  python gpt5_2_vs_3_ast_typing_diff.py ../untyped_benchmarks ../gpt5_2_run ../gpt5_3_run -o untyped_vs_gpt5_2_vs_gpt5_3_typing.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def iter_py_files(root: str) -> Iterable[tuple[str, str]]:
    """Yield (rel_path, abs_path) for each .py under root."""
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root)
            yield rel.replace("\\", "/"), abs_path


def collect_names_and_typing_calls(source: str) -> tuple[List[str], List[str], Dict[str, int], bool]:
    """
    Parse source and return:
      (function_names, class_names, typing_call_counts, parse_ok).

    - Includes nested functions/classes (names only, no scoping).
    - typing_call_counts is a simple counter keyed by short call name, e.g. "cast".
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], [], {}, False

    funcs: List[str] = []
    classes: List[str] = []
    typing_calls: Counter[str] = Counter()

    # Typing-related call names we care about (short names).
    target_call_names = {
        "cast",
        "reveal_type",
        "TypeGuard",
        "assert_type",
        "assert_never",
        "overload",
        "Annotated",
        # Builtins that are often used in type-checking patterns.
        "isinstance",
        "issubclass",
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Call):
            # We only track the "short" name of the call, not full qualification.
            func_node = node.func
            short_name: str | None = None
            if isinstance(func_node, ast.Name):
                short_name = func_node.id
            elif isinstance(func_node, ast.Attribute):
                short_name = func_node.attr
            if short_name and short_name in target_call_names:
                typing_calls[short_name] += 1

    return funcs, classes, dict(typing_calls), True


def run_name(path: str) -> str:
    """Short name for a run (e.g. 'gpt5_2_run', 'untyped_benchmarks')."""
    return os.path.basename(os.path.normpath(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two run dirs by AST (function/class names and typing-related calls). "
            "Writes JSON with per-file diffs."
        )
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
        "run_c",
        nargs="?",
        default=None,
        help="Optional third run directory (e.g. gpt5_3_run). If given, output includes its counts too.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <run_a_name>_vs_<run_b_name>_ast_typing_diff.json in script dir)",
    )
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    run_c = os.path.abspath(args.run_c) if args.run_c is not None else None

    if not os.path.isdir(run_a):
        raise SystemExit(f"Not a directory: {run_a}")
    if not os.path.isdir(run_b):
        raise SystemExit(f"Not a directory: {run_b}")
    if run_c is not None and not os.path.isdir(run_c):
        raise SystemExit(f"Not a directory: {run_c}")

    name_a = run_name(run_a)
    name_b = run_name(run_b)
    name_c = run_name(run_c) if run_c is not None else None

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        if name_c is None:
            out_path = os.path.join(SCRIPT_DIR, f"{name_a}_vs_{name_b}_ast_typing_diff.json")
        else:
            out_path = os.path.join(
                SCRIPT_DIR,
                f"{name_a}_vs_{name_b}_vs_{name_c}_ast_typing_diff.json",
            )

    # Run A: index by rel_path and by basename (so flat run_a works when run_b has subdirs).
    run_a_by_path: Dict[str, str] = {}
    run_a_by_basename: Dict[str, str] = {}
    for rel, path in iter_py_files(run_a):
        run_a_by_path[rel] = path
        basename = os.path.basename(rel)
        if basename not in run_a_by_basename:
            run_a_by_basename[basename] = path

    run_b_files = {rel: path for rel, path in iter_py_files(run_b)}

    # Optional run C: just index by rel_path and basename similarly.
    run_c_by_path: Dict[str, str] = {}
    run_c_by_basename: Dict[str, str] = {}
    if run_c is not None:
        for rel, path in iter_py_files(run_c):
            run_c_by_path[rel] = path
            basename = os.path.basename(rel)
            if basename not in run_c_by_basename:
                run_c_by_basename[basename] = path
    results: List[dict] = []
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

        funcs_a, classes_a, typing_calls_a, ok_a = collect_names_and_typing_calls(code_a)
        funcs_b, classes_b, typing_calls_b, ok_b = collect_names_and_typing_calls(code_b)

        funcs_c: List[str] = []
        classes_c: List[str] = []
        typing_calls_c: Dict[str, int] = {}
        ok_c = True

        path_c = None
        if run_c is not None:
            path_c = run_c_by_path.get(rel_path) or run_c_by_basename.get(os.path.basename(rel_path))
            if path_c and os.path.isfile(path_c):
                with open(path_c, "r", encoding="utf-8", errors="replace") as f:
                    code_c = f.read()
                funcs_c, classes_c, typing_calls_c, ok_c = collect_names_and_typing_calls(code_c)

        if not ok_a:
            results.append(
                {
                    "file": rel_path,
                    "skipped": True,
                    "reason": f"SyntaxError in {name_a}",
                }
            )
            continue
        if not ok_b:
            results.append(
                {
                    "file": rel_path,
                    "skipped": True,
                    "reason": f"SyntaxError in {name_b}",
                }
            )
            continue
        if run_c is not None and path_c and not ok_c:
            results.append(
                {
                    "file": rel_path,
                    "skipped": True,
                    "reason": f"SyntaxError in {name_c}",
                }
            )
            continue

        set_fa, set_fb = set(funcs_a), set(funcs_b)
        set_ca, set_cb = set(classes_a), set(classes_b)
        only_fa = sorted(set_fa - set_fb)
        only_fb = sorted(set_fb - set_fa)
        only_ca = sorted(set_ca - set_cb)
        only_cb = sorted(set_cb - set_ca)

        # Compare typing call counts.
        all_call_keys = set(typing_calls_a) | set(typing_calls_b) | set(typing_calls_c)
        typing_diffs: Dict[str, dict] = {}
        for key in sorted(all_call_keys):
            count_a = typing_calls_a.get(key, 0)
            count_b = typing_calls_b.get(key, 0)
            entry_counts: Dict[str, int] = {
                f"count_in_{name_a}": count_a,
                f"count_in_{name_b}": count_b,
            }
            if run_c is not None and path_c:
                entry_counts[f"count_in_{name_c}"] = typing_calls_c.get(key, 0)

            # Only record if there is any difference across the runs.
            counts_set = set(entry_counts.values())
            if len(counts_set) > 1:
                typing_diffs[key] = entry_counts

        if only_fa or only_fb or only_ca or only_cb or typing_diffs:
            entry: dict = {"file": rel_path}
            if only_fa or only_fb:
                entry.update(
                    {
                        f"functions_only_in_{name_a}": only_fa,
                        f"functions_only_in_{name_b}": only_fb,
                    }
                )
            if only_ca or only_cb:
                entry.update(
                    {
                        f"classes_only_in_{name_a}": only_ca,
                        f"classes_only_in_{name_b}": only_cb,
                    }
                )
            if typing_diffs:
                entry["typing_calls_diff"] = typing_diffs
            results.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()

