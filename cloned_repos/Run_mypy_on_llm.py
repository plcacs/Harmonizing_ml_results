import subprocess
import os
import glob
import json
import ast

def count_parameters(filename):
    total_params = 0
    annotated_params = 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_params += len(node.args.args)
                for arg in node.args.args:
                    if arg.annotation:
                        annotated_params += 1
    except (FileNotFoundError, SyntaxError):
        pass
    return total_params, annotated_params

def run_mypy_and_save_results(directory, output_file):
    results = {}
    # Recursively find all .py files in directory and subdirectories
    for filename in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
        print("Current file: ", filename)
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
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            output = process.stdout.strip()
            error_output = process.stderr.strip()
            file_result = {}
            total_params, annotated_params = count_parameters(filename)
            file_result["stats"] = {
                "total_parameters": total_params,
                "parameters_with_annotations": annotated_params,
            }
            if process.returncode == 0:
                file_result["error_count"] = 0
                file_result["isCompiled"] = True
                file_result["errors"] = []
            else:
                error_count = output.count("\n")
                errors = output.splitlines() if output else error_output.splitlines()
                if error_count == 0 and error_output:
                    error_count = error_output.count('\n')
                file_result["error_count"] = error_count
                file_result["isCompiled"] = False
                file_result["errors"]=errors
            # Use only the file name as the key
            file_key = os.path.basename(filename)
            results[file_key] = file_result
        except FileNotFoundError:
            print("Error: mypy not found. Make sure it's installed and in your PATH.")
            return
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_mypy_and_save_results("deep_seek", "mypy_results_deepseek_with_errors.json")