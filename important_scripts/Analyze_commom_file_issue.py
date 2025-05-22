import json
import ast
import os
import subprocess

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None  # skip invalid files

    total_func_params = 0
    typed_func_params = 0
    total_vars = 0
    typed_vars = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.kwonlyargs:
                total_func_params += 1
                if arg.annotation:
                    typed_func_params += 1
            if node.args.vararg and node.args.vararg.annotation:
                total_func_params += 1
                typed_func_params += 1
            if node.args.kwarg and node.args.kwarg.annotation:
                total_func_params += 1
                typed_func_params += 1
        elif isinstance(node, ast.AnnAssign):
            total_vars += 1
            typed_vars += 1
        elif isinstance(node, ast.Assign):
            total_vars += 1

    return {
        "total_func_params": total_func_params,
        "typed_func_params": typed_func_params,
        "total_vars": total_vars,
        "typed_vars": typed_vars
    }

def get_mypy_error_count(filename):
    command = [
        "mypy",
        "--ignore-missing-imports",
        "--allow-untyped-defs",
        "--no-incremental",
        "--disable-error-code=no-redef",
        "--cache-dir=/dev/null",
        filename,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        error_lines = [line for line in result.stdout.splitlines() if ":" in line]
        return len(error_lines)
    except Exception as e:
        return f"error: {str(e)}"

def analyze_json(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        feature_matches = json.load(f)

    result = {}

    for feature_key, paths in feature_matches.items():
        result[feature_key] = {}

        for label in ['old', 'new']:
            if not paths[label]: continue
            file_path = paths[label][0]
            stats = analyze_file(file_path)
            if stats is None:
                stats = { "error": "Could not parse file" }
            else:
                stats["mypy_error_count"] = get_mypy_error_count(file_path)

            result[feature_key][label] = {
                "file": file_path,
                **stats
            }

    with open(output_json_path, 'w') as out:
        json.dump(result, out, indent=2)

if __name__ == "__main__":
    input_json = "feature_group_matches_o1_mini.json"
    output_json = "file_typing_analysis_o1_mini.json"
    analyze_json(input_json, output_json)
