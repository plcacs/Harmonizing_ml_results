import json
import os
import shutil
from pathlib import Path

def find_and_copy_error_files_for_model(model_name):
    """Find and copy error files for a specific model"""
    # Map directory names to JSON file names
    json_name_mapping = {
        'deep_seek': 'deepseek',
        'o1_mini': 'o1-mini', 
        'gpt4o': 'gpt4o'
    }
    
    # Paths
    json_file_path = f"mypy_results/Filtered_type_errors/merged_{json_name_mapping[model_name]}.json"
    source_dir = model_name
    target_dir = f"{model_name}_llm_only_errors"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract filenames from JSON keys
    filenames = list(data.keys())
    print(f"\n=== Processing {model_name} ===")
    print(f"Found {len(filenames)} files in JSON")
    
    # Track found and copied files
    found_files = []
    copied_files = []
    not_found_files = []
    
    # Search for each file in the model subdirectories
    for filename in filenames:
        found = False
        
        # Search through numbered subdirectories
        for subdir_num in range(1, 19):  # Based on the directory structure
            subdir_path = os.path.join(source_dir, str(subdir_num))
            
            if os.path.exists(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                
                if os.path.exists(file_path):
                    # Copy file to target directory
                    target_path = os.path.join(target_dir, filename)
                    shutil.copy2(file_path, target_path)
                    
                    found_files.append(filename)
                    copied_files.append(filename)
                    found = True
                    print(f"✓ Copied: {filename} from {subdir_path}")
                    break
        
        if not found:
            not_found_files.append(filename)
            print(f"✗ Not found: {filename}")
    
    # Summary for this model
    print(f"\n=== {model_name.upper()} SUMMARY ===")
    print(f"Total files in JSON: {len(filenames)}")
    print(f"Files found and copied: {len(copied_files)}")
    print(f"Files not found: {len(not_found_files)}")
    
    if not_found_files:
        print(f"\nFiles not found:")
        for filename in not_found_files[:10]:  # Show first 10
            print(f"  - {filename}")
        if len(not_found_files) > 10:
            print(f"  ... and {len(not_found_files) - 10} more")
    
    return {
        'model': model_name,
        'total_files': len(filenames),
        'copied_files': len(copied_files),
        'not_found_files': len(not_found_files)
    }

def find_and_copy_error_files():
    """Process all three models"""
    models = ['deep_seek', 'o1_mini', 'gpt4o']  # Fixed directory names
    results = []
    
    for model in models:
        try:
            result = find_and_copy_error_files_for_model(model)
            results.append(result)
        except FileNotFoundError as e:
            print(f"\n❌ Error processing {model}: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error processing {model}: {e}")
    
    # Overall summary
    print(f"\n{'='*50}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*50}")
    for result in results:
        print(f"{result['model']:12} | Total: {result['total_files']:4} | Copied: {result['copied_files']:4} | Not found: {result['not_found_files']:4}")

if __name__ == "__main__":
    find_and_copy_error_files()
