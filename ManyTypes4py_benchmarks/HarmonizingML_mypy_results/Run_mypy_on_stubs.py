import subprocess
import os
import glob
import json
import ast
import time
import shutil


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


def run_mypy_and_save_results(py_directory, stub_directory, output_file):
    results = {}
    all_py_files = list(glob.glob(os.path.join(py_directory, "**", "*.py"), recursive=True))
    total_py = len(all_py_files)

    # Match .py files to their .pyi stubs by stem name
    stub_map = {}
    for stub_path in glob.glob(os.path.join(stub_directory, "*.pyi")):
        stem = os.path.splitext(os.path.basename(stub_path))[0]
        stub_map[stem] = stub_path

    has_stub = []
    skipped = []
    for f in all_py_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in stub_map:
            has_stub.append((f, stub_map[stem]))
        else:
            skipped.append(os.path.basename(f))

    total_files = len(has_stub)
    print(f"Found {total_py} .py files, {total_files} have matching stubs, {len(skipped)} skipped")

    for i, (py_file, stub_file) in enumerate(has_stub, 1):
        print(f"Processing file {i}/{total_files}: {py_file}")

        abs_py = os.path.abspath(py_file)
        py_dir = os.path.dirname(abs_py)

        # Copy the .pyi stub right next to the .py file
        stub_basename = os.path.basename(stub_file)
        temp_stub = os.path.join(py_dir, stub_basename)
        copied = False
        try:
            if not os.path.exists(temp_stub):
                shutil.copy2(stub_file, temp_stub)
                copied = True

            command = [
                "mypy",
                "--ignore-missing-imports",
                "--no-incremental",
                "--python-version=3.10",
                "--disable-error-code=no-redef",
                "--cache-dir=/dev/null",
                abs_py,
            ]

            process = subprocess.run(
                command,
                cwd=py_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            output = process.stdout.strip()
            error_output = process.stderr.strip()

            # Count only actual error lines, not summary lines
            all_lines = output.splitlines() if output else error_output.splitlines()
            error_lines = [l for l in all_lines if ": error:" in l]

            # Parameter stats from the stub
            total_params, annotated_params = count_parameters(stub_file)

            file_result = {
                "stats": {
                    "total_parameters": total_params,
                    "parameters_with_annotations": annotated_params,
                },
                "error_count": len(error_lines),
                "isCompiled": process.returncode == 0,
                "errors": error_lines,
            }

            # Use relative path as key to avoid basename collisions
            rel_key = os.path.relpath(py_file, py_directory)
            results[rel_key] = file_result

        finally:
            # Always clean up the copied stub
            if copied and os.path.exists(temp_stub):
                os.remove(temp_stub)

    results["_metadata"] = {
        "total_py_files": total_py,
        "files_with_stubs": total_files,
        "skipped_no_stub": skipped,
    }

    # Write once, after the loop
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    start_time = time.time()

   
    run_mypy_and_save_results(
        "../deepseek_3_stub_run",
        "../deepseek_3_stub_run",
        "mypy_results_deepseek_3_stub_run_with_errors.json",
    )
    """
    run_mypy_and_save_results(
        "../gpt5_1_stub_run",
        "../gpt5_1_stub_run",
        "mypy_results_gpt5_1_stub_run_with_errors.json",
    )
    
    run_mypy_and_save_results(
        "../claude_stub_1_run",
        "../claude_stub_1_run",
        "mypy_results_claude_stub_1_run_with_errors.json",
    )"""

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
