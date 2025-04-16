import os
import ast
from collections import defaultdict
import json
def count_functions_and_parameters(file_path):
    """ Count functions and their parameters in a given Python file """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        return 0, []

    function_count = 0
    parameter_counts = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
            param_count = len(node.args.args) + len(node.args.kwonlyargs)
            if node.args.vararg:
                param_count += 1  # *args
            if node.args.kwarg:
                param_count += 1  # **kwargs
            parameter_counts.append(param_count)

    return function_count, parameter_counts

def analyze_python_files(root_dir):
    """ Analyze Python files in a directory and group them based on parameter count ranges """
    param_ranges = defaultdict(list)
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py') and not file.endswith('_gemma.py') and not file.endswith('OpenAIAPI.py') and not file.endswith('_llama.py') and not file.endswith('_gpt4o.py'):
                file_path = os.path.join(subdir, file)
                func_count, param_counts = count_functions_and_parameters(file_path)
                
                if func_count >= 5:  # Consider only files with at least 5 functions
                    for count in param_counts:
                        if count >=5:
                            if count>=5 and count<=10:
                                param_ranges["05-10"].append(file_path)
                            elif count < 20:
                                param_ranges["10-20"].append(file_path)
                            elif count < 30:
                                param_ranges["20-30"].append(file_path)
                            elif count < 50:
                                param_ranges["30-50"].append(file_path)
                            else:
                                param_ranges["50+"].append(file_path)

    return param_ranges
def save_to_json(data, json_file):
    """ Save the results to a JSON file """
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
if __name__ == "__main__":
    directory = os.getcwd()  # Change this to your root directory
    result = analyze_python_files(directory)
    json_file = "python_files_for_ml_inference_group1.json"  
    save_to_json(result, json_file)
    print(f"Data saved in json file:{json_file}")

    print("Python files grouped by parameter count range:")
    for key, paths in sorted(result.items()):
        print(f"{key}: {len(paths)} files")
        """for path in paths:
            print(f"  - {path}")  # Print file paths"""
