import subprocess
import os
import glob
import json
import ast

def count_parameters(filename):
    """Counts total and annotated parameters in a Python file."""
    total_params = 0
    annotated_params = 0
    try:
        with open(filename, "r", encoding="utf-8") as f: # added encoding="utf-8"
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_params += len(node.args.args)
                for arg in node.args.args:
                    if arg.annotation:
                        annotated_params += 1
    except (FileNotFoundError, SyntaxError):
        pass  # Handle errors gracefully
    return total_params, annotated_params

def run_mypy_and_save_results(directory=".", output_file="mypy_results_no_type.json"):
    """
    Runs mypy on each Python file, counts parameters, and saves results in JSON.
    """
    results = {}
    with open("grouped_file_paths.json", "r") as f:
        grouped_paths = json.load(f)
    all_files = []
    for file_list in grouped_paths.values():
        all_files.extend(file_list)
    print(len(all_files))
    for filename in all_files:
        
        command = [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--disable-error-code=no-redef",
            "--cache-dir=/dev/null",  # Avoid cache
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
            else:
                error_count = output.count("\n")
                if error_count == 0 and error_output:
                    error_count = error_output.count('\n')
                file_result["error_count"] = error_count
                file_result["isCompiled"] = False

            results[os.path.basename(filename)] = file_result
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            

        except FileNotFoundError:
            print(f"Error: mypy not found. Make sure it's installed and in your PATH.")
            return

    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_mypy_and_save_results()