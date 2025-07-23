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
    all_files = list(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True))
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    for i, filename in enumerate(all_files, 1):
        print(f"Processing file {i}/{total_files}: {filename}")
        abs_path = os.path.abspath(filename)
        command = [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--disable-error-code=no-redef",
            "--cache-dir=/dev/null",
            abs_path,
        ]
        try:
            process = subprocess.run(
                command,
                cwd=os.path.dirname(abs_path),
                capture_output=True,
                text=True,
                check=False
            )
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
                if error_count == 0 and error_output:
                    error_count = error_output.count('\n')
                errors = output.splitlines() if output else error_output.splitlines()
                file_result["error_count"] = error_count
                file_result["isCompiled"] = False
                file_result["errors"] = errors
            file_key = os.path.basename(filename)
            results[file_key] = file_result
        except FileNotFoundError:
            print("Error: mypy not found. Make sure it's installed and in your PATH.")
            return
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
    print(f"\\nResults saved to {output_file}")
import time
if __name__ == "__main__":
    start_time = time.time()
    run_mypy_and_save_results("gpt4o_2nd_run", "mypy_results/mypy_results_gpt4o_2nd_run_with_errors.json")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")