import ast
import astor
import subprocess
from typing import Dict, List, Tuple
import json
import os
import time
timeout = 1800
ERROR_LOG_FILE = 'type_errors.json'
from typing import List, Tuple, Set, FrozenSet

def apply_config(initial_code, config):
    """
    Applies the given configuration (set of removed annotations) to the code and returns the modified version.
    """
    tree = ast.parse(initial_code)

    class TypeRestorer(ast.NodeTransformer):

        def visit_FunctionDef(self, node):
            if (node.name, 'return') in config:
                node.returns = None
            for arg in node.args.args:
                if (node.name, arg.arg) in config:
                    arg.annotation = None
            return self.generic_visit(node)
    return astor.to_source(TypeRestorer().visit(tree))

def log_initial_errors(file_path, errors):
    if not errors.strip():
        return
    error_log = {}
    if os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, 'r') as f:
            try:
                error_log = json.load(f)
            except json.JSONDecodeError:
                pass
    error_log[file_path] = errors
    with open(ERROR_LOG_FILE, 'w') as f:
        json.dump(error_log, f, indent=4)

def collect_type_hints(code):
    annotations = []
    stats = {'total_type_annotations': 0, 'total_parameters': 0, 'parameters_with_annotations': 0}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (annotations, stats, False)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats['total_parameters'] += len(node.args.args)
            for arg in node.args.args:
                if arg.annotation:
                    annotations.append((node.name, arg.arg))
                    stats['parameters_with_annotations'] += 1
            if node.returns:
                annotations.append((node.name, 'return'))
    stats['total_type_annotations'] = len(annotations)
    return (annotations, stats, True)

def typecheck(code):
    with open('temp_file.py', 'w', encoding='utf-8') as f:
        f.write(code)
    command = ['mypy', '--ignore-missing-imports', '--allow-untyped-defs', '--no-incremental', '--disable-error-code=no-redef', '--cache-dir=/dev/null', 'temp_file.py']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    error_count = sum((1 for line in result.stdout.splitlines() if 'error:' in line))
    return (error_count, result.stdout)

def assign_types(input_file, initial_code, annotations, start_time):
    """
    Implements Algorithm 1 (Greedy Type Assignment) using a queue (FIFO) 
    and annotation configurations instead of full code storage.
    """
    current_score, initial_errors = typecheck(initial_code)
    log_initial_errors(input_file, initial_errors)
    if current_score == 0:
        return frozenset()
    work_set = [frozenset()]
    done = {frozenset()}
    best_config = frozenset()
    best_score = current_score
    while work_set:
        if time.time() - start_time > timeout:
            print('Skipping file due to timeout.')
            return (best_config, best_score)
        current_config = work_set.pop(0)
        for func_name, param in annotations:
            new_config = current_config | {(func_name, param)}
            if new_config not in done:
                new_code = apply_config(initial_code, new_config)
                new_score, _ = typecheck(new_code)
                print('Current new_score for file: ', input_file, new_score)
                if new_score < best_score:
                    best_config = new_config
                    best_score = new_score
                    work_set.append(new_config)
                if new_score == 0:
                    return (new_config, 0)
                done.add(new_config)
    return (best_config, best_score)

def main(input_file, start_time):
    with open('failed_original_compiled_no_types.json', 'r') as f:
        failed_files = set(json.load(f))
    if input_file not in failed_files:
        return (None, False)
    with open(input_file, 'r', encoding='utf-8') as f:
        original_code = f.read()
    annotations, original_stats, isCompiled = collect_type_hints(original_code)
    best_config, score = assign_types(input_file, original_code, annotations, start_time)
    print('Best config for file:', input_file, 'is:', best_config)
    print('score', score)
    return (len(annotations), len(best_config), best_config, True, score)

def process_type_analysis_results():
    output_file = 'deepseek_stats_2.json'
    if os.path.exists(output_file):
        with open(output_file, 'r') as outfile:
            try:
                updated_results = json.load(outfile)
            except json.JSONDecodeError:
                updated_results = {}
    else:
        updated_results = {}
    with open('failed_original_compiled_no_types.json', 'r') as f:
        failed_files = set(json.load(f))
    for file_path in failed_files:
        if os.path.exists(file_path) and file_path not in updated_results:
            print(f'Processing: {file_path}')
            start_time = time.time()
            original_param_count, updated_param_count, updated_param, isCompiled, score = main(file_path, start_time)
            if not isCompiled:
                print('Error processing file due to timeout.')
                continue
            updated_results[file_path] = {'original_parameters_with_annotations': original_param_count, 'updated_parameters_with_annotations': original_param_count - updated_param_count, 'updated_config': list(updated_param), 'time_taken': time.time() - start_time, 'score': score}
            with open(output_file, 'w') as outfile:
                json.dump(updated_results, outfile, indent=4)
    print(f'Updated results saved to {output_file}')
if __name__ == '__main__':
    process_type_analysis_results()