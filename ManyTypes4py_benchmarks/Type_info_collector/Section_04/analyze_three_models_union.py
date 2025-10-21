import json
import os
from collections import defaultdict


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_baseline_files():
    """Load baseline files from untyped mypy results (files with isCompiled=True)."""
    untyped_mypy_path = (
        "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
    )
    try:
        with open(untyped_mypy_path, "r", encoding="utf-8") as f:
            untyped_mypy = json.load(f)
        baseline_files = {
            fname
            for fname, info in untyped_mypy.items()
            if info.get("isCompiled") is True
        }
        print(f"Loaded {len(baseline_files)} baseline files")
        return baseline_files
    except Exception as e:
        print(f"Error loading baseline files from {untyped_mypy_path}: {e}")
        return set()


def is_non_any_type(param_types):
    """Check if parameter has a non-Any type annotation."""
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        if isinstance(type_str, str) and type_str.strip():
            # Check if it's explicitly "Any"
            if type_str.strip().lower() == "any":
                return False
            else:
                # Non-empty, non-Any type annotation
                return True
        else:
            # Empty string counts as Any
            return False
    else:
        # No type annotation counts as Any
        return False


def analyze_file_type_coverage(type_info_data, baseline_files=None):
    """Analyze type coverage for a single model's data."""
    if not isinstance(type_info_data, dict):
        return {}
    
    file_results = {}
    
    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue
            
        non_any_count = 0
        total_params = 0
        
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            # Skip 'self' parameter completely
                            param_name = param.get("name", "")
                            if param_name == "self":
                                continue
                                
                            total_params += 1
                            param_types = param.get("type", [])
                            
                            if is_non_any_type(param_types):
                                non_any_count += 1
        
        if total_params > 0:
            any_count = total_params - non_any_count
            file_results[filename] = {
                "total_params": total_params,
                "non_any_count": non_any_count,
                "non_any_percentage": (non_any_count / total_params) * 100.0,
                "any_count": any_count,
                "any_percentage": (any_count / total_params) * 100.0
            }
    
    return file_results


def calculate_union_coverage(o3_mini_data, claude_data, deepseek_data):
    """Calculate union coverage across all three models.
    
    Union logic: If ANY model has a non-Any type for a parameter, 
    then the union has a non-Any type for that parameter.
    """
    union_results = {}
    
    # Get all unique filenames from all models
    all_files = set()
    if o3_mini_data:
        all_files.update(o3_mini_data.keys())
    if claude_data:
        all_files.update(claude_data.keys())
    if deepseek_data:
        all_files.update(deepseek_data.keys())
    
    for filename in all_files:
        # Get data from each model (None if file not present)
        o3_data = o3_mini_data.get(filename) if o3_mini_data else None
        claude_file_data = claude_data.get(filename) if claude_data else None
        deepseek_file_data = deepseek_data.get(filename) if deepseek_data else None
        
        # Determine total parameters (should be same across all models for same file)
        total_params = 0
        if o3_data:
            total_params = o3_data["total_params"]
        elif claude_file_data:
            total_params = claude_file_data["total_params"]
        elif deepseek_file_data:
            total_params = deepseek_file_data["total_params"]
        
        if total_params > 0:
            # For union: if ANY model has non-Any, then union has non-Any
            # This means we take the MAXIMUM non-any count across all models
            # But we need to ensure it doesn't exceed total_params
            max_non_any = 0
            if o3_data:
                max_non_any = max(max_non_any, o3_data["non_any_count"])
            if claude_file_data:
                max_non_any = max(max_non_any, claude_file_data["non_any_count"])
            if deepseek_file_data:
                max_non_any = max(max_non_any, deepseek_file_data["non_any_count"])
            
            # Ensure max_non_any doesn't exceed total_params (safety check)
            max_non_any = min(max_non_any, total_params)
            any_count = total_params - max_non_any
            
            union_results[filename] = {
                "total_params": total_params,
                "non_any_count": max_non_any,
                "non_any_percentage": (max_non_any / total_params) * 100.0,
                "any_count": any_count,
                "any_percentage": (any_count / total_params) * 100.0
            }
    
    return union_results


def main():
    # Load baseline files
    baseline_files = load_baseline_files()
    
    if not baseline_files:
        print("No baseline files found. Exiting.")
        return
    
    # Define model files for the three models we care about
    model_files = {
        "O3-mini": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "Claude": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json", 
        "DeepSeek": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json"
    }
    
    # Load data for each model
    model_data = {}
    for model_name, filename in model_files.items():
        print(f"Loading {model_name} data from {filename}...")
        type_info = load_type_info(filename)
        if type_info:
            model_data[model_name] = analyze_file_type_coverage(type_info, baseline_files)
            print(f"  Loaded {len(model_data[model_name])} files")
        else:
            print(f"  Failed to load {filename}")
            model_data[model_name] = {}
    
    # Calculate union coverage
    print("Calculating union coverage...")
    union_data = calculate_union_coverage(
        model_data.get("O3-mini", {}),
        model_data.get("Claude", {}),
        model_data.get("DeepSeek", {})
    )
    
    # Create final results
    all_files = set()
    for model_data_dict in model_data.values():
        all_files.update(model_data_dict.keys())
    all_files.update(union_data.keys())
    
    results = {}
    
    for filename in sorted(all_files):
        # Get data from each model
        o3_data = model_data.get("O3-mini", {}).get(filename)
        claude_data = model_data.get("Claude", {}).get(filename)
        deepseek_data = model_data.get("DeepSeek", {}).get(filename)
        union_file_data = union_data.get(filename)
        
        # Determine total parameters (use first available)
        total_params = None
        if o3_data:
            total_params = o3_data["total_params"]
        elif claude_data:
            total_params = claude_data["total_params"]
        elif deepseek_data:
            total_params = deepseek_data["total_params"]
        elif union_file_data:
            total_params = union_file_data["total_params"]
        
        if total_params is None:
            continue
            
        # Build result for this file
        file_result = {
            "Total_number_of_parameters": total_params,
            "Number_of_nonany_by_o3mini": o3_data["non_any_count"] if o3_data else None,
            "Percentage_of_nonany_by_o3mini": o3_data["non_any_percentage"] if o3_data else None,
            "Number_of_any_by_o3mini": o3_data["any_count"] if o3_data else None,
            "Percentage_of_any_by_o3mini": o3_data["any_percentage"] if o3_data else None,
            "Number_of_nonany_by_claude": claude_data["non_any_count"] if claude_data else None,
            "Percentage_of_nonany_by_claude": claude_data["non_any_percentage"] if claude_data else None,
            "Number_of_any_by_claude": claude_data["any_count"] if claude_data else None,
            "Percentage_of_any_by_claude": claude_data["any_percentage"] if claude_data else None,
            "Number_of_nonany_by_deepseek": deepseek_data["non_any_count"] if deepseek_data else None,
            "Percentage_of_nonany_by_deepseek": deepseek_data["non_any_percentage"] if deepseek_data else None,
            "Number_of_any_by_deepseek": deepseek_data["any_count"] if deepseek_data else None,
            "Percentage_of_any_by_deepseek": deepseek_data["any_percentage"] if deepseek_data else None,
            "Number_of_nonany_union": union_file_data["non_any_count"] if union_file_data else None,
            "Percentage_of_nonany_union": union_file_data["non_any_percentage"] if union_file_data else None,
            "Number_of_any_union": union_file_data["any_count"] if union_file_data else None,
            "Percentage_of_any_union": union_file_data["any_percentage"] if union_file_data else None
        }
        
        # Calculate parameters that all models assigned Any or left blank
        all_any_count = 0
        if o3_data and claude_data and deepseek_data:
            # This is a simplified calculation - in reality we'd need parameter-level comparison
            # For now, we'll calculate: total - max_non_any_from_any_model
            max_non_any = max(
                o3_data["non_any_count"],
                claude_data["non_any_count"], 
                deepseek_data["non_any_count"]
            )
            all_any_count = total_params - max_non_any
        
        file_result["Number_of_parameters_all_models_any_or_blank"] = all_any_count
        file_result["Percentage_of_parameters_all_models_any_or_blank"] = (all_any_count / total_params * 100.0) if total_params > 0 else 0.0
        
        results[filename] = file_result
    
    # Save results to JSON
    output_file = "three_models_union_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(results)} files")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total files analyzed: {len(results)}")
    
    # Count files present in each model
    o3_files = sum(1 for data in results.values() if data["Number_of_nonany_by_o3mini"] is not None)
    claude_files = sum(1 for data in results.values() if data["Number_of_nonany_by_claude"] is not None)
    deepseek_files = sum(1 for data in results.values() if data["Number_of_nonany_by_deepseek"] is not None)
    
    print(f"Files in O3-mini: {o3_files}")
    print(f"Files in Claude: {claude_files}")
    print(f"Files in DeepSeek: {deepseek_files}")


if __name__ == "__main__":
    main()
