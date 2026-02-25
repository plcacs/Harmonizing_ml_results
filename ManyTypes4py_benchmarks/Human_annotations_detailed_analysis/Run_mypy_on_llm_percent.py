import subprocess
import os
import glob
import json
import ast
import time


def has_syntax_error(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        return False
    except SyntaxError:
        return True


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


def run_mypy_and_save_results(directory, output_file, skip_syntax_errors=True):
    results = {}
    all_files = list(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True))
    total_files = len(all_files)
    skipped = 0
    print(f"Total files to process: {total_files}")

    for i, filename in enumerate(all_files, 1):
        print(f"Processing file {i}/{total_files}: {filename}")
        abs_path = os.path.abspath(filename)
        file_key = os.path.basename(filename)

        if skip_syntax_errors and has_syntax_error(abs_path):
            skipped += 1
            print(f"  -> Skipped (syntax error)")
            results[file_key] = {
                "stats": {"total_parameters": 0, "parameters_with_annotations": 0},
                "error_count": 1,
                "isCompiled": False,
                "errors": ["SyntaxError: file could not be parsed"],
                "skipped_syntax_error": True,
            }
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            continue

        command = [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--disable-error-code=no-redef",
            "--python-version=3.10",
            "--cache-dir=nul",
            abs_path,
        ]
        try:
            process = subprocess.run(
                command,
                cwd=os.path.dirname(abs_path),
                capture_output=True,
                text=True,
                check=False,
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
                errors = output.splitlines() if output else error_output.splitlines()
                error_count = sum(1 for line in errors if ": error:" in line)
                file_result["error_count"] = error_count
                file_result["isCompiled"] = False
                file_result["errors"] = errors
            results[file_key] = file_result
        except FileNotFoundError:
            print("Error: mypy not found. Make sure it's installed and in your PATH.")
            return
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"\nSkipped {skipped}/{total_files} files due to syntax errors")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    start_time = time.time()

    base_dir = "o3_mini_outputs"
    """subdirs = [
        "o3_mini_ten_percent_typed_output",
        "o3_mini_twenty_percent_typed_output",
        "o3_mini_thirty_percent_typed_output",
        "o3_mini_forty_percent_typed_output",
        "o3_mini_fifty_percent_typed_output",
        "o3_mini_sixty_percent_typed_output",
        "o3_mini_seventy_percent_typed_output",
        "o3_mini_eighty_percent_typed_output",
        "o3_mini_ninety_percent_typed_output",
    ]"""
    subdirs = [
        
        "o3_mini_fully_percent_typed_output",
    ]

    for subdir in subdirs:
        input_dir = os.path.join(base_dir, subdir)
        output_file = os.path.join(base_dir, f"mypy_results_{subdir}.json")
        print(f"\n{'=' * 60}")
        print(f"Processing: {subdir}")
        print(f"{'=' * 60}")
        run_mypy_and_save_results(input_dir, output_file, skip_syntax_errors=False)

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
