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

def run_mypy_and_save_results(directory=".", output_file="mypy_results_deepseek.json"):
    """
    Runs mypy on each Python file, counts parameters, and saves results in JSON.
    """
    results = {}

    for filename in glob.glob(os.path.join(directory, "*.py")):
        
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
                file_result["errors"] = []
            else:
                errors = output.splitlines() if output else error_output.splitlines()
                file_result["error_count"] = len(errors)
                file_result["isCompiled"] = False
                file_result["errors"] = errors


            results[os.path.basename(filename)] = file_result

            """with open(f"{os.path.basename(filename)}_mypy_results.txt", "w") as f:
                f.write(output)
                if error_output:
                    f.write(error_output)"""

        except FileNotFoundError:
            print(f"Error: mypy not found. Make sure it's installed and in your PATH.")
            return

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_mypy_and_save_results()