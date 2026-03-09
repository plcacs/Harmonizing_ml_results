"""
Q1: Does the initial LLM response change file structure, or do structure
changes only happen during the iterative mypy-fix process?

Three-way comparison: untyped original → initial LLM response → fixed version
"""

import json
import os
import ast
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
UNTYPED_DIR = os.path.join(PARENT_DIR, "untyped_benchmarks")

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


def find_file(directory, filename):
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_func_info(code):
    """Return dict of {func_name: (param_count, body_stmt_count)}."""
    funcs = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return funcs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            param_count = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
            if node.args.vararg:
                param_count += 1
            if node.args.kwarg:
                param_count += 1
            body_count = len(node.body)
            funcs[node.name] = (param_count, body_count)
    return funcs


def get_class_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def structure_changed(code_a, code_b):
    """AST-based comparison: checks functions, classes, params, body statements."""
    funcs_a = get_func_info(code_a)
    funcs_b = get_func_info(code_b)

    if set(funcs_a.keys()) != set(funcs_b.keys()):
        return True
    for name in funcs_a:
        if funcs_a[name] != funcs_b[name]:
            return True

    if get_class_names(code_a) != get_class_names(code_b):
        return True

    return False


def line_diff(code_a, code_b):
    return abs(len(code_a.splitlines()) - len(code_b.splitlines()))


def analyze_model(model_name, config):
    if not os.path.exists(config["log"]):
        print(f"{model_name}: log not found")
        return

    with open(config["log"], "r") as f:
        log = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v["status"] == "fixed"}

    both_changed = 0
    only_initial_changed = 0
    only_fix_changed = 0
    neither_changed = 0
    skipped = 0

    initial_line_diffs = []
    fix_line_diffs = []

    for filename in fixed_files:
        untyped_path = os.path.join(UNTYPED_DIR, filename)
        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)

        if not os.path.exists(untyped_path) or not initial_path or not os.path.exists(fixed_path):
            skipped += 1
            continue

        untyped = read_file(untyped_path)
        initial = read_file(initial_path)
        fixed = read_file(fixed_path)

        initial_changed = structure_changed(untyped, initial)
        fix_changed = structure_changed(initial, fixed)

        if initial_changed and fix_changed:
            both_changed += 1
        elif initial_changed:
            only_initial_changed += 1
        elif fix_changed:
            only_fix_changed += 1
        else:
            neither_changed += 1

        initial_line_diffs.append(line_diff(untyped, initial))
        fix_line_diffs.append(line_diff(initial, fixed))

    total = both_changed + only_initial_changed + only_fix_changed + neither_changed

    print(f"\n{'='*60}")
    print(f"  {model_name} — Three-Way Comparison ({total} files)")
    print(f"{'='*60}")
    print(f"  Structure changed in initial LLM response only: {only_initial_changed}/{total} ({100*only_initial_changed/total:.1f}%)")
    print(f"  Structure changed in fix iterations only:       {only_fix_changed}/{total} ({100*only_fix_changed/total:.1f}%)")
    print(f"  Structure changed in both phases:               {both_changed}/{total} ({100*both_changed/total:.1f}%)")
    print(f"  No structure change at all:                     {neither_changed}/{total} ({100*neither_changed/total:.1f}%)")
    print(f"  Skipped (file not found):                       {skipped}")

    print(f"\n  --- Line count differences (avg) ---")
    print(f"  Untyped → Initial LLM: {sum(initial_line_diffs)/len(initial_line_diffs):.1f} lines")
    print(f"  Initial LLM → Fixed:   {sum(fix_line_diffs)/len(fix_line_diffs):.1f} lines")


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
