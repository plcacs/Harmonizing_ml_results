import ast
import astor
import subprocess
from typing import Dict, List, Tuple
import json
import os
import time

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
    return sum(1 for line in result.stdout.splitlines() if "error:" in line)

# Step 3: Modified Algorithm 1
def assign_types(initial_code: str, annotations: List[Tuple[str, str]]):
    current_code = initial_code
    current_score = typecheck(current_code)
    work_set = [current_code]
    done = {current_code}
    best_code = current_code
    min_errors = current_score
    while work_set:
        current_code = work_set.pop(0)
        current_score = typecheck(current_code)
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
            new_score = typecheck(new_code)
            if new_score == 0:
                return new_code
            if new_code not in done and new_score < min_errors:
                work_set.append(new_code)
                done.add(new_code)
    return best_code

# Main function
def main(input_file: str):
    with open("failed_original_compiled_no_types.json", "r") as f:
        failed_files = set(json.load(f))
    
    if input_file not in failed_files:
        return None, False
    
    with open(input_file, 'r',encoding='utf-8') as f:
        original_code = f.read()
    
    annotations, original_stats, isCompiled = collect_type_hints(original_code)
    if not isCompiled:
        return None, False
    
    best_code = assign_types(original_code, annotations)
    _, updated_stats, _ = collect_type_hints(best_code)
    return original_stats, updated_stats, True

# Process only files from failed_original_compiled_no_types.json
def process_type_analysis_results():
    output_file = "gpt40_stats_1.json"
    
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
            original_stats, updated_stats, isCompiled = main(file_path)
            if not isCompiled:
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
