import json
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

def get_base_filenames() -> set:
    """Get the set of Python filenames from Hundrad_renamed_benchmarks directory."""
    base_dir = "../../Hundrad_renamed_benchmarks"
    base_filenames = set()
    
    try:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    base_filenames.add(file)
        print(f"Found {len(base_filenames)} base Python files")
        return base_filenames
    except Exception as e:
        print(f"Error reading base directory {base_dir}: {e}")
        return set()

def load_type_info(file_path: str) -> Dict:
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def calculate_file_metrics(type_info_data: Dict, filename: str) -> Tuple[int, int, int, int, float]:
    """Calculate metrics for a specific file using the same logic as calculate_any_rate.py."""
    any_slots = 0
    typed_slots = 0
    param_any = 0
    param_total = 0
    
    if filename not in type_info_data:
        return 0, 0, 0, 0, 0.0
    
    file_data = type_info_data[filename]
    if not isinstance(file_data, dict):
        return 0, 0, 0, 0, 0.0
    
    for func_name, func_data in file_data.items():
        if isinstance(func_data, list):
            for param in func_data:
                if isinstance(param, dict):
                    # Get type annotations
                    param_types = param.get("type", [])
                    if isinstance(param_types, list) and len(param_types) > 0:
                        type_str = param_types[0]
                        if isinstance(type_str, str) and type_str.strip():
                            # Count as typed slot
                            typed_slots += 1
                            category = param.get("category", "")
                            
                            # Check if it's Any
                            if type_str.strip().lower() == "any":
                                any_slots += 1
                            
                            # Count parameters specifically
                            if category == "arg":
                                param_total += 1
                                if type_str.strip().lower() == "any":
                                    param_any += 1
    
    any_ratio = (any_slots / typed_slots * 100) if typed_slots > 0 else 0.0
    return any_slots, typed_slots, param_any, param_total, any_ratio

def main():
    # Define model pairs (original and renamed versions)
    model_pairs = {
        "gpt35": {
            "original": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_gpt35_renamed_output_benchmarks.json"
        },
        "deepseek": {
            "original": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_deepseek_renamed_output_2_benchmarks.json"
        },
        "o3-mini": {
            "original": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_o3_mini_renamed_output_benchmarks.json"
        },
        "claude3-sonnet": {
            "original": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_claude_sonnet_renamed_output_benchmarks.json"
        }
    }
    
    print("Loading base filenames...")
    base_filenames = get_base_filenames()
    if not base_filenames:
        print("No base files found. Exiting.")
        return
    
    # Load all model data
    model_data = {}
    for model_name, paths in model_pairs.items():
        print(f"Loading {model_name} data...")
        model_data[model_name] = {
            "original": load_type_info(paths["original"]),
            "renamed": load_type_info(paths["renamed"])
        }
    
    # Prepare CSV data
    csv_rows = []
    csv_headers = ["Filename"]
    
    # Add headers for each model
    for model_name in model_pairs.keys():
        csv_headers.extend([
            f"{model_name}-any-count",
            f"{model_name}-renamed-any-count",
            f"{model_name}-typed-count", 
            f"{model_name}-renamed-typed-count",
            f"{model_name}-param-any-count",
            f"{model_name}-renamed-param-any-count",
            f"{model_name}-param-total-count",
            f"{model_name}-renamed-param-total-count",
            f"{model_name}-any-ratio",
            f"{model_name}-renamed-any-ratio"
        ])
    
    csv_rows.append(csv_headers)
    
    # Process each file
    print("Processing files...")
    for filename in sorted(base_filenames):
        row = [filename]
        
        for model_name in model_pairs.keys():
            # Calculate metrics for original version
            orig_any, orig_typed, orig_param_any, orig_param_total, orig_ratio = calculate_file_metrics(
                model_data[model_name]["original"], filename
            )
            
            # Calculate metrics for renamed version
            renamed_any, renamed_typed, renamed_param_any, renamed_param_total, renamed_ratio = calculate_file_metrics(
                model_data[model_name]["renamed"], filename
            )
            
            # Add to row
            row.extend([
                orig_any,
                renamed_any,
                orig_typed,
                renamed_typed,
                orig_param_any,
                renamed_param_any,
                orig_param_total,
                renamed_param_total,
                f"{orig_ratio:.2f}",
                f"{renamed_ratio:.2f}"
            ])
        
        csv_rows.append(row)
    
    # Save combined CSV
    output_file = "llm_version_comparison.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"Combined results saved to {output_file}")
    
    # Generate separate CSV for each LLM
    for model_name in model_pairs.keys():
        print(f"Generating separate CSV for {model_name}...")
        
        # Create headers for this LLM
        llm_headers = ["Filename", 
                       f"{model_name}-any-count", f"{model_name}-renamed-any-count",
                       f"{model_name}-typed-count", f"{model_name}-renamed-typed-count",
                       f"{model_name}-param-any-count", f"{model_name}-renamed-param-any-count",
                       f"{model_name}-param-total-count", f"{model_name}-renamed-param-total-count",
                       f"{model_name}-any-ratio", f"{model_name}-renamed-any-ratio"]
        
        # Create data rows for this LLM
        llm_csv_rows = [llm_headers]
        for filename in sorted(base_filenames):
            row = [filename]
            
            # Calculate metrics for original version
            orig_any, orig_typed, orig_param_any, orig_param_total, orig_ratio = calculate_file_metrics(
                model_data[model_name]["original"], filename
            )
            
            # Calculate metrics for renamed version
            renamed_any, renamed_typed, renamed_param_any, renamed_param_total, renamed_ratio = calculate_file_metrics(
                model_data[model_name]["renamed"], filename
            )
            
            # Add to row
            row.extend([
                orig_any,
                renamed_any,
                orig_typed,
                renamed_typed,
                orig_param_any,
                renamed_param_any,
                orig_param_total,
                renamed_param_total,
                f"{orig_ratio:.2f}",
                f"{renamed_ratio:.2f}"
            ])
            
            llm_csv_rows.append(row)
        
        # Save separate CSV for this LLM
        llm_output_file = f"{model_name}_comparison.csv"
        with open(llm_output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(llm_csv_rows)
        
        print(f"  {model_name} results saved to {llm_output_file}")
    
    print(f"\nProcessed {len(base_filenames)} files for {len(model_pairs)} model pairs")

if __name__ == "__main__":
    main()