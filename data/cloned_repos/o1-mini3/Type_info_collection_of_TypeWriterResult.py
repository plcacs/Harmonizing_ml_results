import os
import json
import ast

# Load the provided JSON file
json_path = "o1mini_stats_3.json"
with open(json_path, "r") as f:
    deepseek_data = json.load(f)

def extract_param_types(py_file: str, functions: list):
    if not os.path.exists(py_file):
        return {}

    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except Exception:
        return {}

    type_info = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            if func_name not in [f[0] for f in functions]:
                continue
            func_info = {}
            for arg in node.args.args:
                if arg.annotation:
                    try:
                        ann = ast.unparse(arg.annotation)
                    except Exception:
                        ann = "unknown"
                    func_info[arg.arg] = ann
            if node.returns:
                try:
                    func_info["return"] = ast.unparse(node.returns)
                except Exception:
                    func_info["return"] = "unknown"
            type_info[func_name] = func_info
    return type_info

# Process each file in the input JSON
output = {}
for filename, data in deepseek_data.items():
    if data.get("score", -1) != 0:
        continue  # Skip non-zero s
    py_filename = filename.split("_deepseek_")[0] + ".py"
    functions_to_check = data["updated_config"]
    param_types = extract_param_types(filename, functions_to_check)

    extracted = {}
    for func, param in functions_to_check:
        if func in param_types and param in param_types[func]:
            extracted.setdefault(func, {})[param] = param_types[func][param]
        elif func in param_types and param == "return":
            extracted.setdefault(func, {})[param] = param_types[func].get("return", "None")

    output[filename] = extracted
with open("extracted_type_hints_type_writer_o1mini_1_score0.json", "w") as f:
    json.dump(output, f, indent=4)