import json
import os
import shutil
import random
from pathlib import Path

def collect_random_200_files():
    # Paths
    json_file_path = "mypy_results/split_original_files/files_with_parameter_annotations.json"
    source_dir = "untyped_benchmarks"
    target_dir = "Hundrad_original_typed_benchmarks"
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Filter files where isCompiled=True
    compiled_files = [
        filename for filename, info in data.items() 
        if info.get('isCompiled', False) == True
    ]
    
    # Select 200 random files
    random.seed(42)  # For reproducibility
    selected_files = random.sample(compiled_files, min(200, len(compiled_files)))
    
    print(f"Found {len(compiled_files)} compiled files total")
    print(f"Selected {len(selected_files)} random files (isCompiled=True)")
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(exist_ok=True)
    
    # Copy files
    copied_count = 0
    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"Warning: File {filename} not found in {source_dir}")
    
    print(f"\nSuccessfully copied {copied_count} files to {target_dir}/")
    print(f"Target files list saved to {target_dir}/target_files_list.txt")
    
    # Save the list of target files
    with open(os.path.join(target_dir, "target_files_list.txt"), 'w') as f:
        for filename in selected_files:
            f.write(f"{filename}\n")

if __name__ == "__main__":
    collect_random_200_files() 