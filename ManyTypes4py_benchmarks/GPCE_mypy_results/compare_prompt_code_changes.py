"""
Compare code changes between GPT-5 run 2 (old prompt) and run 3 (new prompt)
relative to the original untyped files.

Usage:
    python compare_prompt_code_changes.py
"""

import ast
import difflib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

UNTYPED_DIR = os.path.join(PARENT_DIR, "untyped_benchmarks")
RUN2_DIR = os.path.join(PARENT_DIR, "gpt5_2_run")
RUN3_DIR = os.path.join(PARENT_DIR, "gpt5_3_run")

OUTPUT_JSON = os.path.join(SCRIPT_DIR, "prompt_code_change_comparison.json")


def collect_files(run_dir):
    """Collect all .py files from a run directory (nested subdirs) keyed by basename."""
    files = {}
    for root, _dirs, filenames in os.walk(run_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                files[fname] = os.path.join(root, fname)
    return files


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (IOError, OSError):
        return None


def compute_diff_stats(original, modified):
    """Compute line-level diff statistics."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    diff = list(difflib.unified_diff(orig_lines, mod_lines, n=0))

    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

    matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)
    similarity = round(matcher.ratio(), 4)

    return {
        "lines_added": added,
        "lines_removed": removed,
        "total_changes": added + removed,
        "similarity": similarity,
        "original_lines": len(orig_lines),
        "modified_lines": len(mod_lines),
    }


def extract_function_signatures(source):
    """Extract function name -> (args_annotation_count, has_return_annotation) from AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    sigs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotated_args = sum(
                1 for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs
                if a.annotation is not None
            )
            total_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
            has_return = node.returns is not None
            sigs[node.name] = {
                "total_args": total_args,
                "annotated_args": annotated_args,
                "has_return_annotation": has_return,
            }
    return sigs


def check_body_changes(original, modified):
    """Check if changes go beyond just annotations (i.e., logic/body was modified)."""
    try:
        orig_tree = ast.parse(original)
        mod_tree = ast.parse(modified)
    except SyntaxError:
        return {"parse_error": True}

    orig_funcs = {}
    mod_funcs = {}

    for node in ast.walk(orig_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            orig_funcs[node.name] = "".join(ast.dump(n) for n in node.body)

    for node in ast.walk(mod_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mod_funcs[node.name] = "".join(ast.dump(n) for n in node.body)

    common = set(orig_funcs.keys()) & set(mod_funcs.keys())
    body_changed = sum(1 for f in common if orig_funcs[f] != mod_funcs[f])
    funcs_added = len(set(mod_funcs.keys()) - set(orig_funcs.keys()))
    funcs_removed = len(set(orig_funcs.keys()) - set(mod_funcs.keys()))

    return {
        "total_functions": len(common),
        "body_changed_count": body_changed,
        "funcs_added": funcs_added,
        "funcs_removed": funcs_removed,
        "annotation_only": body_changed == 0 and funcs_added == 0 and funcs_removed == 0,
    }


def analyze_run(untyped_files, run_files, run_name):
    """Analyze a single run against untyped originals."""
    results = {}
    for fname, run_path in sorted(run_files.items()):
        untyped_path = untyped_files.get(fname)
        if not untyped_path:
            continue

        original = read_file(untyped_path)
        modified = read_file(run_path)
        if original is None or modified is None:
            continue

        diff_stats = compute_diff_stats(original, modified)
        body_info = check_body_changes(original, modified)

        results[fname] = {**diff_stats, **body_info}

    return results


def print_comparison(run2_results, run3_results):
    """Print side-by-side summary of both runs."""
    common_files = set(run2_results.keys()) & set(run3_results.keys())
    print(f"\nFiles in run 2: {len(run2_results)}")
    print(f"Files in run 3: {len(run3_results)}")
    print(f"Common files  : {len(common_files)}")

    r2 = [run2_results[f] for f in common_files]
    r3 = [run3_results[f] for f in common_files]

    def avg(lst, key):
        vals = [r[key] for r in lst if key in r]
        return round(sum(vals) / len(vals), 2) if vals else 0

    def pct(lst, key):
        vals = [r[key] for r in lst if key in r]
        return round(sum(vals) / len(vals) * 100, 1) if vals else 0

    print(f"\n{'Metric':<35} {'Run 2 (old)':>12} {'Run 3 (new)':>12}")
    print("-" * 61)
    print(f"{'Avg lines added':<35} {avg(r2, 'lines_added'):>12} {avg(r3, 'lines_added'):>12}")
    print(f"{'Avg lines removed':<35} {avg(r2, 'lines_removed'):>12} {avg(r3, 'lines_removed'):>12}")
    print(f"{'Avg total changes':<35} {avg(r2, 'total_changes'):>12} {avg(r3, 'total_changes'):>12}")
    print(f"{'Avg similarity ratio':<35} {avg(r2, 'similarity'):>12} {avg(r3, 'similarity'):>12}")
    print(f"{'Avg functions with body changes':<35} {avg(r2, 'body_changed_count'):>12} {avg(r3, 'body_changed_count'):>12}")
    print(f"{'Avg functions added':<35} {avg(r2, 'funcs_added'):>12} {avg(r3, 'funcs_added'):>12}")
    print(f"{'Avg functions removed':<35} {avg(r2, 'funcs_removed'):>12} {avg(r3, 'funcs_removed'):>12}")
    print(f"{'% annotation-only files':<35} {pct(r2, 'annotation_only'):>11}% {pct(r3, 'annotation_only'):>11}%")

    # Bucket by total_changes
    print("\n\n=== Distribution of Total Line Changes (common files) ===\n")
    buckets = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 99999)]
    print(f"  {'Range':<20} {'Run 2':>8} {'Run 3':>8}")
    for lo, hi in buckets:
        label = f"[{lo}, {hi})" if hi != 99999 else f"[{lo}+)"
        c2 = sum(1 for r in r2 if lo <= r.get("total_changes", 0) < hi)
        c3 = sum(1 for r in r3 if lo <= r.get("total_changes", 0) < hi)
        print(f"  {label:<20} {c2:>8} {c3:>8}")


def main():
    untyped_files = {f: os.path.join(UNTYPED_DIR, f)
                     for f in os.listdir(UNTYPED_DIR) if f.endswith(".py")}

    run2_files = collect_files(RUN2_DIR)
    run3_files = collect_files(RUN3_DIR)

    print("Analyzing run 2 (old prompt)...")
    run2_results = analyze_run(untyped_files, run2_files, "gpt5_2_run")

    print("Analyzing run 3 (new prompt)...")
    run3_results = analyze_run(untyped_files, run3_files, "gpt5_3_run")

    print_comparison(run2_results, run3_results)

    output = {"gpt5_2_run": run2_results, "gpt5_3_run": run3_results}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
