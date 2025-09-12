import json
import os
import csv
import re
from collections import defaultdict


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_any_score(type_str):
    """Calculate Any score for a type annotation.
    Any = 0, Non-Any = 1"""
    if not isinstance(type_str, str):
        return 0
    
    type_str = type_str.strip().lower()
    
    # Check if it's Any or empty
    if type_str in ["any", "tp.any", "typing.any", ""]:
        return 0
    
    # Any non-Any type gets score 1
    return 1


def extract_file_level_slots(run1_data, run2_data):
    """Extract type slots organized by file for both runs."""
    run1_file_slots = defaultdict(dict)
    run2_file_slots = defaultdict(dict)
    
    # Extract slots from run1
    for filename, functions in run1_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            param_types = param.get("type", [])
                            
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    slot_key = f"{func_name}:{category}:{param_name}"
                                    run1_file_slots[filename][slot_key] = type_str.strip()
    
    # Extract slots from run2
    for filename, functions in run2_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            param_types = param.get("type", [])
                            
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    slot_key = f"{func_name}:{category}:{param_name}"
                                    run2_file_slots[filename][slot_key] = type_str.strip()
    
    return run1_file_slots, run2_file_slots


def analyze_file_any_score(run1_file_slots, run2_file_slots):
    """Analyze file-level Any score comparison."""
    file_results = {}
    
    for filename in run1_file_slots:
        if filename not in run2_file_slots:
            continue
            
        run1_slots = run1_file_slots[filename]
        run2_slots = run2_file_slots[filename]
        
        # Find common slots for this file
        common_slots = set(run1_slots.keys()) & set(run2_slots.keys())
        
        if not common_slots:
            continue
        
        # Calculate total precision scores for the file
        run1_precision_score = 0  # Count of static types (non-Any)
        run2_precision_score = 0  # Count of static types (non-Any)
        
        for slot in common_slots:
            run1_type = run1_slots[slot]
            run2_type = run2_slots[slot]
            
            run1_precision_score += get_any_score(run1_type)
            run2_precision_score += get_any_score(run2_type)
        
        # Determine file-level category based on total precision scores
        if run1_precision_score > run2_precision_score:
            category = "A"  # 1st run more precise (more static types)
        elif run2_precision_score > run1_precision_score:
            category = "B"  # 2nd run more precise (more static types)
        else:
            category = "C"  # equal precision
        
        file_results[filename] = {
            "category": category,
            "total_slots": len(common_slots),
            "run1_precision_score": run1_precision_score,
            "run2_precision_score": run2_precision_score
        }
    
    return file_results


def main():
    # Define model pairs (1st and 2nd runs)
    model_pairs = {
        "GPT35": {
            "run1": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json"
        },
        "GPT4o": {
            "run1": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt4o_2nd_run_benchmarks.json"
        },
        "O1-mini": {
            "run1": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o1_mini_2nd_run_benchmarks.json"
        },
        "O3-mini": {
            "run1": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o3_mini_2nd_run_benchmarks.json"
        },
        "DeepSeek": {
            "run1": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_deep_seek_2nd_run_benchmarks.json"
        },
        "Claude3-Sonnet": {
            "run1": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_claude3_sonnet_2nd_run_benchmarks.json"
        }
    }
    
    results = {}
    
    for model_name, file_paths in model_pairs.items():
        # Load data for both runs
        run1_data = load_type_info(file_paths["run1"])
        run2_data = load_type_info(file_paths["run2"])
        
        if not run1_data or not run2_data:
            continue
        
        # Extract file-level slots
        run1_file_slots, run2_file_slots = extract_file_level_slots(run1_data, run2_data)
        
        # Analyze file-level Any score
        file_results = analyze_file_any_score(run1_file_slots, run2_file_slots)
        
        # Aggregate results
        category_a_count = 0
        category_b_count = 0
        category_c_count = 0
        total_files = len(file_results)
        
        category_c_details = {
            "run1_precision_score": 0,
            "run2_precision_score": 0
        }
        
        for filename, file_data in file_results.items():
            if file_data["category"] == "A":
                category_a_count += 1
            elif file_data["category"] == "B":
                category_b_count += 1
            elif file_data["category"] == "C":
                category_c_count += 1
                category_c_details["run1_precision_score"] += file_data["run1_precision_score"]
                category_c_details["run2_precision_score"] += file_data["run2_precision_score"]
        
        results[model_name] = {
            "total_files": total_files,
            "category_a": category_a_count,
            "category_b": category_b_count,
            "category_c": category_c_count,
            "category_c_details": category_c_details
        }
    
    # Print summary table
    print(f"{'Model':<15} {'Total Files':<12} {'Category A':<12} {'Category B':<12} {'Category C':<12}")
    print("-" * 80)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['total_files']:<12} {data['category_a']:<12} "
              f"{data['category_b']:<12} {data['category_c']:<12}")
    
    # Print Category C details
    print(f"\n{'Model':<15} {'Category C Files':<15} {'1st Run Score':<15} {'2nd Run Score':<15}")
    print("-" * 90)
    
    for model_name, data in results.items():
        if data['category_c'] > 0:
            print(f"{model_name:<15} {data['category_c']:<15} "
                  f"{data['category_c_details']['run1_precision_score']:<15} "
                  f"{data['category_c_details']['run2_precision_score']:<15} "
                  )


if __name__ == "__main__":
    main()
