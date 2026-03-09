"""
Q2: What are the specific structure changes beyond cast insertion?

Compares initial LLM response vs fixed version using AST to identify:
- Functions/classes added or removed
- Function parameter counts changed
- Function body statement counts changed
- New non-typing imports added
- type: ignore added
- cast() added
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


def get_non_typing_imports(code):
    imports = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "typing" not in node.module:
                imports.add(node.module)
    return imports


def count_type_ignore(code):
    return sum(1 for line in code.splitlines() if "# type: ignore" in line)


def count_cast(code):
    return code.count("cast(")


def analyze_file(initial_code, fixed_code):
    changes = {
        "funcs_added": [],
        "funcs_removed": [],
        "funcs_params_changed": [],
        "funcs_body_changed": [],
        "classes_added": [],
        "classes_removed": [],
        "imports_added": [],
        "type_ignore_added": False,
        "cast_added": False,
        "annotation_only": True,
    }

    init_funcs = get_func_info(initial_code)
    fixed_funcs = get_func_info(fixed_code)

    for name in fixed_funcs:
        if name not in init_funcs:
            changes["funcs_added"].append(name)
            changes["annotation_only"] = False
    for name in init_funcs:
        if name not in fixed_funcs:
            changes["funcs_removed"].append(name)
            changes["annotation_only"] = False

    for name in set(init_funcs) & set(fixed_funcs):
        init_params, init_body = init_funcs[name]
        fixed_params, fixed_body = fixed_funcs[name]
        if init_params != fixed_params:
            changes["funcs_params_changed"].append(name)
            changes["annotation_only"] = False
        if init_body != fixed_body:
            changes["funcs_body_changed"].append(name)
            changes["annotation_only"] = False

    init_classes = get_class_names(initial_code)
    fixed_classes = get_class_names(fixed_code)
    changes["classes_added"] = sorted(fixed_classes - init_classes)
    changes["classes_removed"] = sorted(init_classes - fixed_classes)
    if changes["classes_added"] or changes["classes_removed"]:
        changes["annotation_only"] = False

    init_imports = get_non_typing_imports(initial_code)
    fixed_imports = get_non_typing_imports(fixed_code)
    changes["imports_added"] = sorted(fixed_imports - init_imports)
    if changes["imports_added"]:
        changes["annotation_only"] = False

    if count_type_ignore(fixed_code) > count_type_ignore(initial_code):
        changes["type_ignore_added"] = True
    if count_cast(fixed_code) > count_cast(initial_code):
        changes["cast_added"] = True

    return changes


def analyze_model(model_name, config):
    if not os.path.exists(config["log"]):
        print(f"{model_name}: log not found")
        return

    with open(config["log"], "r") as f:
        log = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v["status"] == "fixed"}

    change_types = Counter()
    annotation_only_count = 0
    total = 0
    all_details = {}

    for filename in fixed_files:
        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)

        if not initial_path or not os.path.exists(fixed_path):
            continue

        initial_code = read_file(initial_path)
        fixed_code = read_file(fixed_path)
        changes = analyze_file(initial_code, fixed_code)
        all_details[filename] = changes
        total += 1

        if changes["annotation_only"]:
            annotation_only_count += 1
        if changes["funcs_added"]:
            change_types["functions_added"] += 1
        if changes["funcs_removed"]:
            change_types["functions_removed"] += 1
        if changes["funcs_params_changed"]:
            change_types["function_params_changed"] += 1
        if changes["funcs_body_changed"]:
            change_types["function_body_changed"] += 1
        if changes["classes_added"]:
            change_types["classes_added"] += 1
        if changes["classes_removed"]:
            change_types["classes_removed"] += 1
        if changes["imports_added"]:
            change_types["non_typing_imports_added"] += 1
        if changes["type_ignore_added"]:
            change_types["type_ignore_added"] += 1
        if changes["cast_added"]:
            change_types["cast_added"] += 1

    print(f"\n{'='*60}")
    print(f"  {model_name} — Detailed Structure Changes ({total} fixed files)")
    print(f"{'='*60}")
    print(f"  Annotation-only fixes (no structure change): {annotation_only_count}/{total} ({100*annotation_only_count/total:.1f}%)")
    print(f"\n  --- Types of structure changes ---")
    for change, count in change_types.most_common():
        print(f"    {change}: {count}/{total} ({100*count/total:.1f}%)")

    out_path = os.path.join(BASE_DIR, f"structure_changes_{model_name.lower().replace('-', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(all_details, f, indent=2)
    print(f"\n  Per-file details saved to: {out_path}")


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
