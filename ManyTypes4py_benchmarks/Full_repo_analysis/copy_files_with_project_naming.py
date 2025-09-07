#!/usr/bin/env python3
"""
Script to copy Python files from an input directory and save them in untyped_version
with ProjectName_FileName.py format, handling multi-level subdirectories.
"""

import os
import shutil
from pathlib import Path

def copy_python_files_with_naming(input_dir, output_dir):
    """Copy Python files from input directory to output directory with project naming."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist!")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified output directory: {output_path}")
    
    # Find all Python files recursively
    python_files = list(input_path.rglob("*.py"))
    
    if not python_files:
        print(f"No Python files found in {input_dir}")
        return False
    
    print(f"Found {len(python_files)} Python files in {input_dir}")
    
    copied_count = 0
    for py_file in python_files:
        try:
            # Get relative path from input directory
            rel_path = py_file.relative_to(input_path)
            
            # Extract project name (first directory) and filename
            path_parts = rel_path.parts
            
            if len(path_parts) >= 2:
                # Get project name (first directory) and filename
                project_name = path_parts[0]
                filename = path_parts[-1]  # Last part is the filename
                
                # Create new filename with project prefix
                new_filename = f"{project_name}_{filename}"
            else:
                # If file is directly in input directory (no subdirectories)
                new_filename = py_file.name
            
            # Create output file path
            output_file = output_path / new_filename
            
            # Handle file overwriting by adding counter if file exists
            counter = 1
            original_new_filename = new_filename
            while output_file.exists():
                name, ext = original_new_filename.rsplit('.', 1)
                new_filename = f"{name}_{counter}.{ext}"
                output_file = output_path / new_filename
                counter += 1
            
            # Copy the file
            shutil.copy2(py_file, output_file)
            print(f"Copied: {rel_path} -> {new_filename}")
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {py_file}: {e}")
    
    print(f"Successfully copied {copied_count} out of {len(python_files)} files")
    return True

def main():
    """Main function to execute the script."""
    # Hardcoded input directory - change this as needed
    input_directory = "catt"
    copy_python_files_with_naming(input_directory, "original_version")
    input_directory = "databases"
    copy_python_files_with_naming(input_directory, "original_version")
    input_directory = "easyquotation"
    copy_python_files_with_naming(input_directory, "original_version")
    input_directory = "requests-html"
    copy_python_files_with_naming(input_directory, "original_version")
    input_directory = "musicbox"
    copy_python_files_with_naming(input_directory, "original_version")

if __name__ == "__main__":
    main()
