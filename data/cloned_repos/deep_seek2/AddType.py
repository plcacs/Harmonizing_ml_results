import ast
import astor
import subprocess
from typing import Dict, List, Tuple
import json
import os
import time

timeout = 1800
ERROR_LOG_FILE = "type_errors.json"

from typing import List, Tuple, Set, FrozenSet

def apply_config(initial_code: str, config: FrozenSet[Tuple[str, str]]) -> str:
    """
    Applies the given configuration (set of removed annotations) to the code and returns the modified version.
    """
    tree = ast.parse(initial_code)
    
    class TypeRestorer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if (node.name, "return") in config:
                node.returns = None
            for arg in node.args.args:
                if (node.name, arg.arg) in config:
                    arg.annotation = None
            return self.generic_visit(node)
    
    return astor.to_source(TypeRestorer().visit(tree))


def log_initial_errors(file_path: str, errors: str):
    if not errors.strip():
        return
    
    error_log = {}
    if os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, "r") as f:
            try:
                error_log = json.load(f)
            except json.JSONDecodeError:
                pass
    
    error_log[file_path] = errors
    
    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(error_log, f, indent=4)

# Step 1: Collect the existing type annotations
def collect_type_hints(code: str):
    annotations = []
    stats = {"total_type_annotations": 0, "total_parameters": 0, "parameters_with_annotations": 0}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return annotations, stats, False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats["total_parameters"] += len(node.args.args)
            for arg in node.args.args:
                if arg.annotation:
                    annotations.append((node.name, arg.arg))
                    stats["parameters_with_annotations"] += 1
            if node.returns:
                annotations.append((node.name, "return"))
    
    stats["total_type_annotations"] = len(annotations)
    return annotations, stats, True

# Step 2: Type checking function
def typecheck(code: str) -> int:
    with open("temp_file.py", "w",encoding='utf-8') as f:
        f.write(code)
    command = [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--disable-error-code=no-redef",
            "--cache-dir=/dev/null",  # Avoid cache
            "temp_file.py",
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    error_count = sum(1 for line in result.stdout.splitlines() if "error:" in line)
    return error_count, result.stdout

# Step 3: Modified Algorithm 1
def assign_types(input_file:str,initial_code: str, annotations: List[Tuple[str, str]],start_time: float) :
    current_code = initial_code
    current_score,initial_errors = typecheck(current_code)
    log_initial_errors(input_file, initial_errors)
    print("current_score",current_score)
    if current_score==0:
        return current_code
    work_set = [current_code]
    done = {current_code}
    best_code = current_code
    min_errors = current_score
    while work_set:
        if time.time() - start_time > timeout:
            print("Skipping file due to timeout.")
            return None  # Timeout reached
        current_code = work_set.pop(0)
        current_score,_ = typecheck(current_code)
        print("current_score",current_score)
        if current_score < min_errors:
            best_code = current_code
            min_errors = current_score
        if current_score == 0:
            return best_code
        for func_name, param in annotations:
            tree = ast.parse(current_code)
            class TypeRestorer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == func_name:
                        if param == "return":
                            node.returns = None
                        for arg in node.args.args:
                            if arg.arg == param:
                                arg.annotation = None
                    return self.generic_visit(node)
            new_tree = TypeRestorer().visit(tree)
            new_code = astor.to_source(new_tree)
            new_score,_ = typecheck(new_code)
            print("new_score",current_score)
            if new_score == 0:
                return new_code
            if new_code not in done and new_score < min_errors:
                work_set.append(new_code)
                done.add(new_code)
    return best_code

# Main function
def main(input_file: str,start_time: float):
    with open("failed_original_compiled_no_types.json", "r") as f:
        failed_files = set(json.load(f))
    
    if input_file not in failed_files:
        return None, False
    
    with open(input_file, 'r',encoding='utf-8') as f:
        original_code = f.read()
    
    annotations, original_stats, isCompiled = collect_type_hints(original_code)
    if not isCompiled:
        return None, False
    
    best_code = assign_types(input_file,original_code, annotations,start_time)
    if best_code is None:
        return original_stats, None, False
    _, updated_stats, _ = collect_type_hints(best_code)
    return original_stats, updated_stats, True

# Process only files from failed_original_compiled_no_types.json
def process_type_analysis_results():
    output_file = "deepseek_stats_2.json"
    
    # Load existing results if the file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as outfile:
            try:
                updated_results = json.load(outfile)
            except json.JSONDecodeError:
                updated_results = {}
    else:
        updated_results = {}
    
    with open("failed_original_compiled_no_types.json", "r") as f:
        failed_files = set(json.load(f))
    
    for file_path in failed_files:
        if os.path.exists(file_path) and file_path not in updated_results:
            print(f"Processing: {file_path}")
            start_time = time.time()
            original_stats, updated_stats, isCompiled = main(file_path,start_time)
            if not isCompiled:
                print("Error processing file due to timeout.")
                continue
            updated_results[file_path] = {
                "original_total_parameters": original_stats["total_parameters"],
                "original_parameters_with_annotations": original_stats["parameters_with_annotations"],
                "updated_parameters_with_annotations": updated_stats["parameters_with_annotations"],
                "time_taken": time.time() - start_time
            }
            # Write each result to file immediately
            with open(output_file, "w") as outfile:
                json.dump(updated_results, outfile, indent=4)
    
    print(f"Updated results saved to {output_file}")

if __name__ == "__main__":
    process_type_analysis_results()
