#!/usr/bin/env python3
"""
Script to copy Python files from specified directories and create untyped versions.
"""

import os
import shutil
import ast
import warnings
from pathlib import Path

def copy_python_files(source_dir, target_dir):
    """Copy all Python files from source directory to target directory."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified target directory: {target_path}")
    
    # Find all Python files recursively
    python_files = list(source_path.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files in {source_dir}")
    
    for py_file in python_files:
        # Calculate relative path from source directory
        rel_path = py_file.relative_to(source_path)
        target_file = target_path / rel_path
        
        # Create target directory structure
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(py_file, target_file)
        print(f"Copied: {rel_path}")

class RemoveTypeHints(ast.NodeTransformer):
    """AST transformer to remove type annotations from Python code."""
    
    def visit_ClassDef(self, node):
        # Visit body first to remove annotations inside class
        self.generic_visit(node)
        # If class body is empty after removals, insert a pass
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        # Remove return annotation
        node.returns = None
        # Remove parameter annotations
        for arg in node.args.args + getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        # Convert "x: T = value" to "x = value"
        if node.value:
            new_node = ast.Assign(
                targets=[node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset
            )
            return ast.copy_location(new_node, node)
        return None

    def visit_arg(self, node):
        node.annotation = None
        return node

def remove_type_annotations(content):
    """Remove type annotations from Python code using AST."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
        
        transformer = RemoveTypeHints()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        
        try:
            untyped_code = ast.unparse(tree)
        except AttributeError:
            # Fallback for older Python versions
            import astunparse
            untyped_code = astunparse.unparse(tree)
        
        return untyped_code
    except Exception as e:
        print(f"Error processing code with AST: {e}")
        return content  # Return original content if AST processing fails

def create_untyped_version(source_dir, target_dir):
    """Create untyped versions of Python files."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified untyped target directory: {target_path}")
    
    # Find all Python files recursively
    python_files = list(source_path.rglob("*.py"))
    
    print(f"Creating untyped versions of {len(python_files)} Python files from {source_dir}")
    
    for py_file in python_files:
        # Calculate relative path from source directory
        rel_path = py_file.relative_to(source_path)
        target_file = target_path / rel_path
        
        # Create target directory structure
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read original file
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove type annotations
        untyped_content = remove_type_annotations(content)
        
        # Write untyped version
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(untyped_content)
        
        print(f"Created untyped version: {rel_path}")

def main():
    """Main function to execute the script."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    base_dir = script_dir  # Since script is in ManyTypes4py_benchmarks
    full_repo_dir = base_dir / "Full_repo_analysis"
    
    # Directories to process
    source_dirs = [
        "catt",
        "requests-html", 
        "musicbox",
        "easyquotation",
        "databases"
    ]
    
    # Ensure base directory exists
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist!")
        return
    
    print(f"Base directory: {base_dir}")
    print(f"Full repo directory: {full_repo_dir}")
    
    # Create Full_repo_analysis directory if it doesn't exist
    full_repo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory: {full_repo_dir}")
    
    # Check if all source directories exist
    missing_dirs = []
    for source_dir in source_dirs:
        source_path = base_dir / source_dir
        print(f"Checking: {source_path} (exists: {source_path.exists()})")
        if not source_path.exists():
            missing_dirs.append(source_dir)
    
    if missing_dirs:
        print(f"Warning: The following source directories do not exist: {missing_dirs}")
        print("Skipping missing directories...")
        source_dirs = [d for d in source_dirs if d not in missing_dirs]
    
    if not source_dirs:
        print("No valid source directories found!")
        return
    
    print("Starting file copying process...")
    
    # Step 1: Copy Python files to Full_repo_analysis
    for source_dir in source_dirs:
        source_path = base_dir / source_dir
        target_path = full_repo_dir / source_dir
        
        print(f"\nProcessing {source_dir}...")
        copy_python_files(source_path, target_path)
    
    # Step 2: Create untyped_version and untyped_version_large directories
    untyped_dir = full_repo_dir / "untyped_version"
    untyped_large_dir = full_repo_dir / "untyped_version_large"
    
    print(f"\nCreating directories: {untyped_dir} and {untyped_large_dir}")
    untyped_dir.mkdir(parents=True, exist_ok=True)
    untyped_large_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Create untyped versions
    print("\nCreating untyped versions...")
    for source_dir in source_dirs:
        source_path = full_repo_dir / source_dir
        target_path = untyped_dir / source_dir
        
        print(f"\nCreating untyped version of {source_dir}...")
        create_untyped_version(source_path, target_path)
    
    print("\nScript completed successfully!")
    print(f"Files copied to: {full_repo_dir}")
    print(f"Untyped versions created in: {untyped_dir}")

if __name__ == "__main__":
    main()
