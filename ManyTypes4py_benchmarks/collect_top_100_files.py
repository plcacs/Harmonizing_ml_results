import json
import os
import shutil
from pathlib import Path

def collect_top_100_files():
    # Paths
    json_file_path = "mypy_results/split_original_files/files_with_parameter_annotations.json"
    source_dir = "untyped_benchmarks"
    target_dir = "Hundrad_original_typed_benchmarks"
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Filter files where isCompiled=True and sort by total_parameters (descending)
    compiled_files = [
        (filename, info) for filename, info in data.items() 
        if info.get('isCompiled', False) == True
    ]
    
    sorted_files = sorted(
        compiled_files, 
        key=lambda x: x[1]['stats']['total_parameters'], 
        reverse=True
    )
    
    top_100_files = [filename for filename, _ in sorted_files[:100]]
    
    print(f"Found {len(compiled_files)} compiled files total")
    print(f"Selected top {len(top_100_files)} files with highest total_parameters (isCompiled=True)")
    print(f"Top 5 files by total_parameters:")
    for i, (filename, info) in enumerate(sorted_files[:5]):
        print(f"{i+1}. {filename}: {info['stats']['total_parameters']} parameters")
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(exist_ok=True)
    
    # Copy files
    copied_count = 0
    for filename in top_100_files:
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
        for filename in top_100_files:
            f.write(f"{filename}\n")

if __name__ == "__main__":
    collect_top_100_files() 