"""
What specific annotation changes does the fix loop make?

Compares initial LLM response vs fixed version at the AST level to track:
- Parameter annotations changed (from X → Y)
- Return annotations changed
- Variable annotations changed
- Annotations added/removed
- Most common type transitions
"""

import json
import os
import ast
import glob
from collections import Counter

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


def find_file(directory, filename):
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def annotation_to_str(node):
    """Convert an AST annotation node to a readable string."""
    if node is None:
        return None
    return ast.unparse(node)


def get_param_annotations(code):
    """Return list of (func_name, param_name, annotation_str) for all parameters."""
    annotations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return annotations
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
            for arg in all_args:
                annotations.append((node.name, arg.arg, annotation_to_str(arg.annotation)))
            if node.args.vararg:
                annotations.append((node.name, "*" + node.args.vararg.arg, annotation_to_str(node.args.vararg.annotation)))
            if node.args.kwarg:
                annotations.append((node.name, "**" + node.args.kwarg.arg, annotation_to_str(node.args.kwarg.annotation)))
    return annotations


def get_return_annotations(code):
    """Return list of (func_name, annotation_str)."""
    annotations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return annotations
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append((node.name, annotation_to_str(node.returns)))
    return annotations


def get_var_annotations(code):
    """Return list of (var_name, annotation_str)."""
    annotations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return annotations
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotations.append((node.target.id, annotation_to_str(node.annotation)))
    return annotations


def compare_annotations(initial_code, fixed_code):
    init_params = {(f, p): a for f, p, a in get_param_annotations(initial_code)}
    fixed_params = {(f, p): a for f, p, a in get_param_annotations(fixed_code)}

    init_returns = {f: a for f, a in get_return_annotations(initial_code)}
    fixed_returns = {f: a for f, a in get_return_annotations(fixed_code)}

    init_vars = {v: a for v, a in get_var_annotations(initial_code)}
    fixed_vars = {v: a for v, a in get_var_annotations(fixed_code)}

    changes = {
        "param_changed": [],
        "param_added": [],
        "param_removed": [],
        "return_changed": [],
        "return_added": [],
        "return_removed": [],
        "var_changed": [],
        "var_added": [],
        "var_removed": [],
    }

    # Parameter annotations
    all_param_keys = set(init_params.keys()) | set(fixed_params.keys())
    for key in all_param_keys:
        init_ann = init_params.get(key)
        fixed_ann = fixed_params.get(key)
        if init_ann == fixed_ann:
            continue
        if init_ann is not None and fixed_ann is not None:
            changes["param_changed"].append((key, init_ann, fixed_ann))
        elif init_ann is None and fixed_ann is not None:
            changes["param_added"].append((key, fixed_ann))
        elif init_ann is not None and fixed_ann is None:
            changes["param_removed"].append((key, init_ann))

    # Return annotations
    all_return_keys = set(init_returns.keys()) | set(fixed_returns.keys())
    for key in all_return_keys:
        init_ann = init_returns.get(key)
        fixed_ann = fixed_returns.get(key)
        if init_ann == fixed_ann:
            continue
        if init_ann is not None and fixed_ann is not None:
            changes["return_changed"].append((key, init_ann, fixed_ann))
        elif init_ann is None and fixed_ann is not None:
            changes["return_added"].append((key, fixed_ann))
        elif init_ann is not None and fixed_ann is None:
            changes["return_removed"].append((key, init_ann))

    # Variable annotations
    all_var_keys = set(init_vars.keys()) | set(fixed_vars.keys())
    for key in all_var_keys:
        init_ann = init_vars.get(key)
        fixed_ann = fixed_vars.get(key)
        if init_ann == fixed_ann:
            continue
        if init_ann is not None and fixed_ann is not None:
            changes["var_changed"].append((key, init_ann, fixed_ann))
        elif init_ann is None and fixed_ann is not None:
            changes["var_added"].append((key, fixed_ann))
        elif init_ann is not None and fixed_ann is None:
            changes["var_removed"].append((key, init_ann))

    return changes


def analyze_model(model_name, config):
    if not os.path.exists(config["log"]):
        print(f"{model_name}: log not found")
        return

    with open(config["log"], "r") as f:
        log = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v["status"] == "fixed"}

    total_param_changed = 0
    total_param_added = 0
    total_param_removed = 0
    total_return_changed = 0
    total_return_added = 0
    total_return_removed = 0
    total_var_changed = 0
    total_var_added = 0
    total_var_removed = 0
    files_with_no_ann_changes = 0

    transition_counter = Counter()
    changed_to_any = 0
    changed_from_any = 0
    total_changed = 0

    for filename in fixed_files:
        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)

        if not initial_path or not os.path.exists(fixed_path):
            continue

        changes = compare_annotations(read_file(initial_path), read_file(fixed_path))

        total_param_changed += len(changes["param_changed"])
        total_param_added += len(changes["param_added"])
        total_param_removed += len(changes["param_removed"])
        total_return_changed += len(changes["return_changed"])
        total_return_added += len(changes["return_added"])
        total_return_removed += len(changes["return_removed"])
        total_var_changed += len(changes["var_changed"])
        total_var_added += len(changes["var_added"])
        total_var_removed += len(changes["var_removed"])

        has_changes = any(len(v) > 0 for v in changes.values())
        if not has_changes:
            files_with_no_ann_changes += 1

        for _, old, new in changes["param_changed"]:
            transition_counter[(old, new)] += 1
            total_changed += 1
            if new == "Any":
                changed_to_any += 1
            if old == "Any":
                changed_from_any += 1

        for _, old, new in changes["return_changed"]:
            transition_counter[(old, new)] += 1
            total_changed += 1
            if new == "Any":
                changed_to_any += 1
            if old == "Any":
                changed_from_any += 1

        for _, old, new in changes["var_changed"]:
            transition_counter[(old, new)] += 1
            total_changed += 1
            if new == "Any":
                changed_to_any += 1
            if old == "Any":
                changed_from_any += 1

    total_files = len(fixed_files)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Annotation Diff Analysis ({total_files} fixed files)")
    print(f"{'='*60}")

    print(f"\n  --- Annotation counts ---")
    print(f"  Params changed:   {total_param_changed}")
    print(f"  Params added:     {total_param_added}")
    print(f"  Params removed:   {total_param_removed}")
    print(f"  Returns changed:  {total_return_changed}")
    print(f"  Returns added:    {total_return_added}")
    print(f"  Returns removed:  {total_return_removed}")
    print(f"  Vars changed:     {total_var_changed}")
    print(f"  Vars added:       {total_var_added}")
    print(f"  Vars removed:     {total_var_removed}")

    print(f"\n  --- Summary ---")
    total_all_changes = (total_param_changed + total_param_added + total_param_removed +
                         total_return_changed + total_return_added + total_return_removed +
                         total_var_changed + total_var_added + total_var_removed)
    print(f"  Total annotation modifications: {total_all_changes}")
    print(f"  Files with zero annotation changes: {files_with_no_ann_changes}/{total_files}")

    print(f"\n  --- Any transitions ---")
    print(f"  Changed TO Any:   {changed_to_any}/{total_changed} ({100*changed_to_any/total_changed:.1f}%)" if total_changed else "  No changes")
    print(f"  Changed FROM Any: {changed_from_any}/{total_changed} ({100*changed_from_any/total_changed:.1f}%)" if total_changed else "")

    print(f"\n  --- Top 15 type transitions (old → new) ---")
    for (old, new), count in transition_counter.most_common(15):
        print(f"    {old} → {new}: {count}")


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
