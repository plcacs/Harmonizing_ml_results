import ast
import astor
import sys
import subprocess
import os
import json
def remove_type_annotations(file_path):
    
    
    stats = {
        'total_parameters': 0,
        'parameters_with_annotations': 0
    }
    with open(file_path, 'r',encoding='utf-8') as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code,stats,False
    

    class TypeAnnotationRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Remove type annotations from function arguments
            for arg in node.args.args:
                stats['total_parameters'] += 1
                if arg.annotation:
                    stats['parameters_with_annotations'] += 1
                    
                    arg.annotation = None

            # Remove return type annotation
            if node.returns:
                stats['total_parameters'] += 1
                stats['parameters_with_annotations'] += 1
                node.returns = None

            self.generic_visit(node)
            return node

    # Remove type annotations
    remover = TypeAnnotationRemover()
    tree = remover.visit(tree)

    # Convert the modified AST back to source code
    modified_code = astor.to_source(tree)
    return modified_code, stats,True



def process_mypyoutput(output):
    
    errors = 0
    missing_types = 0
    for line in output.splitlines():
        if "error:" in line and "type" in line.lower():  # Focus on type-related errors
            errors += 1
        if "missing type annotation" in line:
            missing_types += 1

    # Calculate score
   
    return errors

def run_mypy_original(file_path):
    """Run mypy on the original file before removing type annotations."""
    try:
        result = subprocess.run(
            ['mypy', '--ignore-missing-imports', '--allow-untyped-defs', file_path],
            capture_output=True, text=True
        )
        original_mypy_errors = process_mypyoutput(result.stdout)
        return original_mypy_errors
    except FileNotFoundError:
        print("Error: Mypy is not installed. Please install it using 'pip install mypy'.")
        return None


def main(input_file):
    """Run mypy on the original file, then remove annotations and run mypy again."""
    
    # Run mypy on the original file
   

    output_file = 'output_no_types.py'

    # Remove type annotations and collect stats
    modified_code, stats, isCompiled = remove_type_annotations(input_file)
    if not isCompiled:
        return 0,0, stats, isCompiled
    original_mypy_errors = run_mypy_original(input_file)
    # Save the modified code to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_code)

    print(f"Type annotations removed. Modified code saved to {output_file}")
    print("Statistics:")
    print(f"  Total parameters: {stats['total_parameters']}")
    print(f"  Parameters with annotations: {stats['parameters_with_annotations']}")

    # Run mypy on the modified file
    try:
        result = subprocess.run(
            ['mypy', '--ignore-missing-imports', '--allow-untyped-defs', output_file],
            capture_output=True, text=True
        )
        modified_mypy_errors = process_mypyoutput(result.stdout)
        return original_mypy_errors,modified_mypy_errors,stats,modified_mypy_errors == 0
        """return {
            "original_mypy_errors": original_mypy_errors,
            "modified_mypy_errors": modified_mypy_errors,
            "stats": stats,
            "isCompiled": modified_mypy_errors == 0
        }"""
    except FileNotFoundError:
        print("Error: Mypy is not installed. Please install it using 'pip install mypy'.")


def analyze_results(json_file):
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_files = len(data)
    compiled_files = sum(1 for file, details in data.items() if details.get("isCompiled") == True)

    print(f"Total files: {total_files}")
    print(f"Files with isCompiled == True: {compiled_files}")

def run_remove_type_on_python_files(root_dir):
    results = {}
    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Only process Python files
            if filename.endswith('.py') :
                file_path = os.path.join(dirpath, filename)
                
                original_count,modified_count,stats,isCompiled=main(file_path)
                print(original_count,modified_count)
                isCompiled_mod=True if modified_count==0 else False
                isCompiled_original=True if original_count==0 else False
                results[file_path] = {
                    "original_error_count":original_count,
                    "modified_error_count": modified_count,
                    "stats": stats,
                    "isCompiled_original":isCompiled_original,
                    "isCompiled_modified": isCompiled_mod
                }
    with open('type_analysis_results_gpt40.json', 'w') as f:
        json.dump(results, f, indent=4)
    analyze_results('type_analysis_results_gpt40.json')
    
if __name__ == "__main__":
    run_remove_type_on_python_files(os.getcwd())
