import json
import ast
import os
import shutil
from pathlib import Path

def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

def remove_type_annotations_from_ast(tree, num_to_remove):
    """Remove type annotations from AST nodes"""
    removed_count = 0
    
    def process_node(node):
        nonlocal removed_count
        if removed_count >= num_to_remove:
            return
        
        # Process function definitions
        if isinstance(node, ast.FunctionDef):
            # Remove parameter annotations
            for arg in node.args.args:
                if arg.annotation and removed_count < num_to_remove:
                    arg.annotation = None
                    removed_count += 1
                    print(f"  Removed parameter annotation: {arg.arg}")
            
            # Remove return type annotation
            if node.returns and removed_count < num_to_remove:
                node.returns = None
                removed_count += 1
                print(f"  Removed return annotation from function: {node.name}")
        
        # Process class definitions
        elif isinstance(node, ast.ClassDef):
            # Remove class variable annotations
            for i, item in enumerate(node.body):
                if isinstance(item, ast.AnnAssign) and removed_count < num_to_remove:
                    # Convert to simple assignment with proper attributes
                    if item.value:  # Only if it has a value
                        new_assign = ast.Assign(
                            targets=[item.target],
                            value=item.value
                        )
                        # Copy important attributes
                        new_assign.lineno = getattr(item, 'lineno', 0)
                        new_assign.col_offset = getattr(item, 'col_offset', 0)
                        new_assign.end_lineno = getattr(item, 'end_lineno', None)
                        new_assign.end_col_offset = getattr(item, 'end_col_offset', None)
                        
                        node.body[i] = new_assign
                        removed_count += 1
                        try:
                            target_name = ast.unparse(item.target)
                            print(f"  Removed class variable annotation: {target_name}")
                        except:
                            print(f"  Removed class variable annotation")
        
        # Process module-level annotations
        elif isinstance(node, ast.Module):
            for i, item in enumerate(node.body):
                if isinstance(item, ast.AnnAssign) and removed_count < num_to_remove:
                    # Convert to simple assignment with proper attributes
                    if item.value:  # Only if it has a value
                        new_assign = ast.Assign(
                            targets=[item.target],
                            value=item.value
                        )
                        # Copy important attributes
                        new_assign.lineno = getattr(item, 'lineno', 0)
                        new_assign.col_offset = getattr(item, 'col_offset', 0)
                        new_assign.end_lineno = getattr(item, 'end_lineno', None)
                        new_assign.end_col_offset = getattr(item, 'end_col_offset', None)
                        
                        node.body[i] = new_assign
                        removed_count += 1
                        try:
                            target_name = ast.unparse(item.target)
                            print(f"  Removed module-level annotation: {target_name}")
                        except:
                            print(f"  Removed module-level annotation")
    
    # Process all nodes in the tree
    for node in ast.walk(tree):
        process_node(node)
    
    return removed_count

def process_file(source_path, target_path, num_to_remove):
    """Process a single Python file and remove type annotations"""
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    try:
        # Parse the source code into AST
        tree = ast.parse(source_code)
        
        # Remove type annotations
        removed_count = remove_type_annotations_from_ast(tree, num_to_remove)
        
        # Convert AST back to source code
        modified_code = ast.unparse(tree)
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Write modified code to target file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        return removed_count
        
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return 0

def main():
    # Input paths (relative to ManyTypes4py_benchmarks)
    json_file = "mypy_results/split_original_files/files_with_parameter_annotations.json"
    original_files_dir = "original_files"
    output_dir = "partially_typed_files"
    
    # Load JSON data
    data = load_json_file(json_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    for filename, file_info in data.items():
        # Only process compiled files
        if not file_info.get("isCompiled", False):
            continue
        
        # Calculate how many annotations to remove (50% of annotated parameters)
        annotated_params = file_info["stats"]["parameters_with_annotations"]
        num_to_remove = annotated_params // 2
        
        if num_to_remove == 0:
            continue
        
        # Source and target paths
        source_path = os.path.join(original_files_dir, filename)
        target_path = os.path.join(output_dir, filename)
        
        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"Source file not found: {source_path}")
            continue
        
        # Process the file
        removed_count = process_file(source_path, target_path, num_to_remove)
        
        print(f"Processed {filename}: removed {removed_count}/{num_to_remove} annotations")
        processed_count += 1
    
    print(f"\nTotal files processed: {processed_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
