import json
import os
import ast
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

MODELS = {
    "GPT-5": {
        "log": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fix_log.json"),
        "fixed_dir": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fixed_files"),
        "original_dir": os.path.join(PARENT_DIR, "gpt5_1st_run"),
    },
    "DeepSeek": {
        "log": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fix_log.json"),
        "fixed_dir": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fixed_files"),
        "original_dir": os.path.join(PARENT_DIR, "deep_seek_2nd_run"),
    },  "Claude": {
        "log": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
        "fixed_dir": os.path.join(PARENT_DIR, "claude_mypy_fix", "fixed_files"),
        "original_dir": os.path.join(PARENT_DIR, "claude3_sonnet_1st_run"),
    },
    # "Claude": {
    #     "log": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
    #     "fixed_dir": os.path.join(PARENT_DIR, "claude_mypy_fix", "fixed_files"),
    #     "original_dir": os.path.join(PARENT_DIR, "claude3_sonnet_1st_run"),
    # },
}


def find_file(directory, filename):
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def count_any(code):
    return code.count(": Any") + code.count("-> Any")


def count_type_ignore(code):
    return sum(1 for line in code.splitlines() if "# type: ignore" in line)


def count_cast(code):
    return code.count("cast(")


def get_typing_imports(code):
    imports = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "typing" in node.module:
            for alias in node.names:
                imports.add(alias.name)
    return imports


def get_annotations(code):
    """Count parameter annotations, return annotations, and variable annotations."""
    param_anns = 0
    return_anns = 0
    var_anns = 0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return param_anns, return_anns, var_anns
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation:
                    param_anns += 1
            if node.args.vararg and node.args.vararg.annotation:
                param_anns += 1
            if node.args.kwarg and node.args.kwarg.annotation:
                param_anns += 1
            if node.returns:
                return_anns += 1
        if isinstance(node, ast.AnnAssign):
            var_anns += 1
    return param_anns, return_anns, var_anns


def code_body_changed(orig_code, fixed_code):
    """Check if non-annotation code changed by comparing function body line counts."""
    def strip_annotations_and_imports(code):
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("from typing") or stripped.startswith("import typing"):
                continue
            if "# type: ignore" in stripped:
                line = line.split("# type: ignore")[0].rstrip()
            lines.append(line)
        return lines

    orig_lines = strip_annotations_and_imports(orig_code)
    fixed_lines = strip_annotations_and_imports(fixed_code)
    return len(orig_lines) != len(fixed_lines)


def analyze_file(original_path, fixed_path):
    orig = read_file(original_path)
    fixed = read_file(fixed_path)

    orig_any = count_any(orig)
    fixed_any = count_any(fixed)

    orig_ignore = count_type_ignore(orig)
    fixed_ignore = count_type_ignore(fixed)

    orig_cast = count_cast(orig)
    fixed_cast = count_cast(fixed)

    orig_imports = get_typing_imports(orig)
    fixed_imports = get_typing_imports(fixed)
    new_imports = fixed_imports - orig_imports

    orig_param, orig_ret, orig_var = get_annotations(orig)
    fixed_param, fixed_ret, fixed_var = get_annotations(fixed)

    return {
        "line_count_orig": len(orig.splitlines()),
        "line_count_fixed": len(fixed.splitlines()),
        "any_orig": orig_any,
        "any_fixed": fixed_any,
        "any_increased": fixed_any > orig_any,
        "type_ignore_orig": orig_ignore,
        "type_ignore_fixed": fixed_ignore,
        "type_ignore_added": fixed_ignore > orig_ignore,
        "cast_orig": orig_cast,
        "cast_fixed": fixed_cast,
        "new_typing_imports": sorted(new_imports),
        "param_annotations_orig": orig_param,
        "param_annotations_fixed": fixed_param,
        "return_annotations_orig": orig_ret,
        "return_annotations_fixed": fixed_ret,
        "var_annotations_orig": orig_var,
        "var_annotations_fixed": fixed_var,
        "code_structure_changed": code_body_changed(orig, fixed),
    }


def analyze_model(model_name, config):
    log_path = config["log"]
    if not os.path.exists(log_path):
        print(f"{model_name}: log not found")
        return

    with open(log_path, "r") as f:
        log = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v["status"] == "fixed"}
    print(f"\n{'='*60}")
    print(f"  {model_name} — Detailed Fix Analysis ({len(fixed_files)} fixed files)")
    print(f"{'='*60}")

    all_results = {}
    any_increased_count = 0
    type_ignore_added_count = 0
    cast_added_count = 0
    structure_changed_count = 0
    all_new_imports = {}

    for filename in fixed_files:
        orig_path = find_file(config["original_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)

        if not orig_path or not os.path.exists(fixed_path):
            continue

        result = analyze_file(orig_path, fixed_path)
        all_results[filename] = result

        if result["any_increased"]:
            any_increased_count += 1
        if result["type_ignore_added"]:
            type_ignore_added_count += 1
        if result["cast_fixed"] > result["cast_orig"]:
            cast_added_count += 1
        if result["code_structure_changed"]:
            structure_changed_count += 1
        for imp in result["new_typing_imports"]:
            all_new_imports[imp] = all_new_imports.get(imp, 0) + 1

    total = len(all_results)
    if total == 0:
        print("  No files to analyze.")
        return

    total_any_orig = sum(r["any_orig"] for r in all_results.values())
    total_any_fixed = sum(r["any_fixed"] for r in all_results.values())
    total_ignore_orig = sum(r["type_ignore_orig"] for r in all_results.values())
    total_ignore_fixed = sum(r["type_ignore_fixed"] for r in all_results.values())

    print(f"\n  --- Any usage ---")
    print(f"  Total 'Any' (original):     {total_any_orig}")
    print(f"  Total 'Any' (fixed):        {total_any_fixed}")
    print(f"  Files where Any increased:  {any_increased_count}/{total} ({100*any_increased_count/total:.1f}%)")

    print(f"\n  --- # type: ignore ---")
    print(f"  Total (original):           {total_ignore_orig}")
    print(f"  Total (fixed):              {total_ignore_fixed}")
    print(f"  Files where added:          {type_ignore_added_count}/{total} ({100*type_ignore_added_count/total:.1f}%)")

    print(f"\n  --- cast() usage ---")
    print(f"  Files where cast added:     {cast_added_count}/{total} ({100*cast_added_count/total:.1f}%)")

    print(f"\n  --- Code structure ---")
    print(f"  Files with structure change: {structure_changed_count}/{total} ({100*structure_changed_count/total:.1f}%)")

    print(f"\n  --- New typing imports (top 10) ---")
    for imp, count in sorted(all_new_imports.items(), key=lambda x: -x[1])[:10]:
        print(f"    {imp}: {count} files")

    # Save per-file details
    out_path = os.path.join(BASE_DIR, f"detailed_analysis_{model_name.lower().replace('-', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Per-file details saved to: {out_path}")


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
