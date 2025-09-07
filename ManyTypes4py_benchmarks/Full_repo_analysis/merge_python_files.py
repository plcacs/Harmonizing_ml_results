#!/usr/bin/env python3
"""
Script to merge all Python files from each project in untyped_version 
into single files and save them in untyped_version_large.
"""

import os
from pathlib import Path

def merge_python_files_in_project(project_dir, output_file):
    """Merge all Python files in a project directory into a single file."""
    project_path = Path(project_dir)
    
    if not project_path.exists():
        print(f"Project directory {project_dir} does not exist!")
        return False
    
    # Find all Python files recursively
    python_files = list(project_path.rglob("*.py"))
    
    if not python_files:
        print(f"No Python files found in {project_dir}")
        return False
    
    print(f"Found {len(python_files)} Python files in {project_dir}")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    merged_content = []
    merged_content.append(f"# Merged Python files from {project_path.name}")
    merged_content.append(f"# Total files: {len(python_files)}")
    merged_content.append("")
    
    # Sort files for consistent ordering
    python_files.sort()
    
    for py_file in python_files:
        try:
            # Calculate relative path from project directory
            rel_path = py_file.relative_to(project_path)
            
            # Add file separator and header
            merged_content.append("")
            merged_content.append(f"# File: {rel_path}")
            merged_content.append("")
            
            # Read and add file content
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                merged_content.append(content)
                
            print(f"  Added: {rel_path}")
            
        except Exception as e:
            print(f"  Error reading {py_file}: {e}")
            merged_content.append(f"# Error reading file: {e}")
    
    # Write merged content to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        print(f"Successfully merged {len(python_files)} files into {output_file}")
        return True
    except Exception as e:
        print(f"Error writing merged file {output_file}: {e}")
        return False

def main():
    """Main function to execute the script."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    untyped_dir = script_dir / "untyped_version"
    untyped_large_dir = script_dir / "untyped_version_large"
    
    print(f"Script directory: {script_dir}")
    print(f"Untyped directory: {untyped_dir}")
    print(f"Untyped large directory: {untyped_large_dir}")
    
    # Check if untyped_version directory exists
    if not untyped_dir.exists():
        print(f"Error: untyped_version directory {untyped_dir} does not exist!")
        print("Please run the copy_and_create_untyped.py script first.")
        return
    
    # Create untyped_version_large directory if it doesn't exist
    untyped_large_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory: {untyped_large_dir}")
    
    # Find all project directories in untyped_version
    project_dirs = [d for d in untyped_dir.iterdir() if d.is_dir()]
    
    if not project_dirs:
        print("No project directories found in untyped_version!")
        return
    
    print(f"Found {len(project_dirs)} project directories:")
    for project_dir in project_dirs:
        print(f"  - {project_dir.name}")
    
    print("\nStarting file merging process...")
    
    success_count = 0
    for project_dir in project_dirs:
        project_name = project_dir.name
        output_file = untyped_large_dir / f"{project_name}_merged.py"
        
        print(f"\nProcessing project: {project_name}")
        if merge_python_files_in_project(project_dir, output_file):
            success_count += 1
    
    print(f"\nScript completed!")
    print(f"Successfully merged {success_count} out of {len(project_dirs)} projects")
    print(f"Merged files saved in: {untyped_large_dir}")

if __name__ == "__main__":
    main()
