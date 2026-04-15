import subprocess
import os
import glob
import json
import ast
import time


def _strip_markdown_fences(text):
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _iter_function_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _count_fn_params(fn_node):
    args = fn_node.args
    return (
        len(getattr(args, "posonlyargs", []))
        + len(args.args)
        + len(args.kwonlyargs)
        + (1 if args.vararg else 0)
        + (1 if args.kwarg else 0)
    )


def _count_annotated_params(fn_node):
    args = fn_node.args
    count = 0
    for arg in getattr(args, "posonlyargs", []):
        if arg.annotation:
            count += 1
    for arg in args.args:
        if arg.annotation:
            count += 1
    for arg in args.kwonlyargs:
        if arg.annotation:
            count += 1
    if args.vararg and args.vararg.annotation:
        count += 1
    if args.kwarg and args.kwarg.annotation:
        count += 1
    return count


def count_parameters(filename):
    total_params = 0
    annotated_params = 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()
        source = _strip_markdown_fences(source)
        tree = ast.parse(source)
        for node in _iter_function_nodes(tree):
            total_params += _count_fn_params(node)
            annotated_params += _count_annotated_params(node)
    except (FileNotFoundError, SyntaxError):
        pass
    return total_params, annotated_params


def run_mypy_and_save_results(directory, output_file):
    results = {}
    all_py_files = list(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True))
    total_py = len(all_py_files)

    has_stub = []
    skipped = []
    for f in all_py_files:
        stub_path = os.path.splitext(f)[0] + ".pyi"
        if os.path.exists(stub_path):
            has_stub.append(f)
        else:
            skipped.append(os.path.basename(f))

    total_files = len(has_stub)
    print(f"Found {total_py} .py files, {total_files} have matching .pyi stubs, {len(skipped)} skipped")

    for i, filename in enumerate(has_stub, 1):
        print(f"Processing file {i}/{total_files}: {filename}")
        abs_path = os.path.abspath(filename)
        command = [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--python-version=3.10",
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
                check=False,
            )
            output = process.stdout.strip()
            error_output = process.stderr.strip()
            file_result = {}
            stub_path = os.path.splitext(filename)[0] + ".pyi"
            total_params, annotated_params = count_parameters(stub_path)
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
                    error_count = error_output.count("\n")
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

    results["_metadata"] = {
        "total_py_files": total_py,
        "files_with_stubs": total_files,
        "skipped_no_stub": skipped,
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    start_time = time.time()

   
    run_mypy_and_save_results(
        "../deepseek_3_stub_run",
        "mypy_results_deepseek_3_stub_run_with_errors.json",
    )
    """
    run_mypy_and_save_results(
        "../gpt5_1_stub_run",
        "mypy_results_gpt5_1_stub_run_with_errors.json",
    )
    
    run_mypy_and_save_results(
        "../claude_stub_1_run",
        "mypy_results_claude_stub_1_run_with_errors.json",
    )"""

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
