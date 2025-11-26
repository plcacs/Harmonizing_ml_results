import json
import os


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


def load_mypy_results(mypy_file_path):
    """Load mypy results and return compilation status for each file."""
    try:
        with open(mypy_file_path, "r", encoding="utf-8") as f:
            mypy_data = json.load(f)
        return {fname: info.get("isCompiled", False) for fname, info in mypy_data.items()}
    except Exception as e:
        print(f"Error loading mypy results from {mypy_file_path}: {e}")
        return {}


def load_mypy_results(mypy_file_path):
    """Load mypy results and return compilation status for each file."""
    try:
        with open(mypy_file_path, "r", encoding="utf-8") as f:
            mypy_data = json.load(f)
        return {fname: info.get("isCompiled", False) for fname, info in mypy_data.items()}
    except Exception as e:
        print(f"Error loading mypy results from {mypy_file_path}: {e}")
        return {}


def is_any_type(param_types):
    """Check if parameter type is Any or empty, matching Table_03_any_empty_rate.py logic."""
    is_any = False
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        if isinstance(type_str, str) and type_str.strip():
            # Check if it's explicitly "Any"
            if type_str.strip().lower() == "any":
                is_any = True
        else:
            # Empty string counts as Any
            is_any = True
    else:
        # No type annotation counts as Any
        is_any = True
    return is_any


def get_type_string(param_types):
    """Extract type string from param_types list."""
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        if isinstance(type_str, str) and type_str.strip():
            return type_str.strip()
    return ""


def get_precision_score(type_str):
    """Simple precision score: higher = more precise. Any/empty = 0."""
    if not type_str or type_str.lower() == "any":
        return 0
    # Simple scoring: basic types get higher scores
    if type_str.lower() in ["str", "int", "float", "bool"]:
        return 10
    return 5  # Other types


def analyze_not_comparable(not_comparable_pairs):
    """Analyze why types are not comparable - show most common type pairs."""
    if not not_comparable_pairs:
        return {}
    
    # Count frequency of each type pair
    pair_counts = {}
    for pair in not_comparable_pairs:
        # Normalize pair (sort to avoid duplicates like (a,b) vs (b,a))
        normalized = tuple(sorted(pair))
        pair_counts[normalized] = pair_counts.get(normalized, 0) + 1
    
    # Get top 10 most common pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_pairs": len(not_comparable_pairs),
        "unique_pairs": len(pair_counts),
        "top_pairs": sorted_pairs
    }


def extract_common_type_slots(run1_data, run2_data, allowed_files):
    """Extract common parameter and return type slots between two runs.
    Only includes slots from allowed files."""
    run1_slots = {}
    run2_slots = {}
    
    # Extract slots from run1
    for filename, functions in run1_data.items():
        # Only process files that are in allowed_files
        if filename not in allowed_files:
            continue
            
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            
                            # Skip 'self' parameter
                            if param_name == "self":
                                continue
                                
                            param_types = param.get("type", [])
                            slot_key = f"{filename}:{func_name}:{category}:{param_name}"
                            run1_slots[slot_key] = param_types
    
    # Extract slots from run2
    for filename, functions in run2_data.items():
        # Only process files that are in allowed_files
        if filename not in allowed_files:
            continue
            
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            
                            # Skip 'self' parameter
                            if param_name == "self":
                                continue
                                
                            param_types = param.get("type", [])
                            slot_key = f"{filename}:{func_name}:{category}:{param_name}"
                            run2_slots[slot_key] = param_types
    
    # Find common slots
    common_slots = set(run1_slots.keys()) & set(run2_slots.keys())
    
    return common_slots, run1_slots, run2_slots


def analyze_any_truth_table(common_slots, run1_slots, run2_slots):
    """Analyze Any predictions in truth table format.
    Divides Any Neither into 4 groups based on precision."""
    any_both = 0
    any_run1_only = 0
    any_run2_only = 0
    neither_run1_more_precise = 0
    neither_run2_more_precise = 0
    neither_both_identical = 0
    neither_not_comparable = 0
    
    # Track not comparable type pairs for analysis
    not_comparable_pairs = []
    
    for slot in common_slots:
        run1_param_types = run1_slots.get(slot, [])
        run2_param_types = run2_slots.get(slot, [])
        
        run1_is_any = is_any_type(run1_param_types)
        run2_is_any = is_any_type(run2_param_types)
        
        if run1_is_any and run2_is_any:
            any_both += 1
        elif run1_is_any and not run2_is_any:
            any_run1_only += 1
        elif not run1_is_any and run2_is_any:
            any_run2_only += 1
        else:  # neither is any - categorize by precision
            run1_type = get_type_string(run1_param_types)
            run2_type = get_type_string(run2_param_types)
            
            # Check if identical
            if run1_type.lower() == run2_type.lower():
                neither_both_identical += 1
            else:
                # Compare precision scores
                run1_score = get_precision_score(run1_type)
                run2_score = get_precision_score(run2_type)
                
                if run1_score > run2_score:
                    neither_run1_more_precise += 1
                elif run2_score > run1_score:
                    neither_run2_more_precise += 1
                else:
                    # Same score but different types = not comparable
                    neither_not_comparable += 1
                    # Store the type pair for analysis
                    type_pair = (run1_type.lower(), run2_type.lower())
                    not_comparable_pairs.append(type_pair)
    
    return (any_both, any_run1_only, any_run2_only, 
            neither_run1_more_precise, neither_run2_more_precise, 
            neither_both_identical, neither_not_comparable, not_comparable_pairs)


def main():
    # Load baseline files (files with isCompiled=True)
    baseline_files = load_baseline_files()

    if not baseline_files:
        print("No baseline files found. Exiting.")
        return

    # Define model pairs (1st and 2nd runs)
    model_pairs = {
        "GPT35": {
            "run1": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json"
        },
        "GPT4o": {
            "run1": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt4o_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json"
        },
        "O1-mini": {
            "run1": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o1_mini_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json"
        },
        "O3-mini": {
            "run1": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o3_mini_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json"
        },
        "DeepSeek": {
            "run1": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_deep_seek_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json"
        },
        "Claude3-Sonnet": {
            "run1": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_claude3_sonnet_2nd_run_benchmarks.json",
            "mypy_run1": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
            "mypy_run2": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json"
        }
    }
    
    print("=" * 80)
    print("RUN-TO-RUN COMPARISON OF ANY PREDICTIONS (TRUTH TABLE)")
    print(f"BASELINE FILES: {len(baseline_files)} files with isCompiled=True")
    print("FILTER: Only files that pass mypy type checking in BOTH runs are analyzed")
    print("=" * 80)
    
    results = {}
    
    for model_name, file_paths in model_pairs.items():
        # Load data for both runs
        run1_data = load_type_info(file_paths["run1"])
        run2_data = load_type_info(file_paths["run2"])
        
        if not run1_data or not run2_data:
            print(f"Failed to load data for {model_name}")
            continue
        
        # Load mypy results for both runs
        mypy_run1 = load_mypy_results(file_paths["mypy_run1"])
        mypy_run2 = load_mypy_results(file_paths["mypy_run2"])
        
        if not mypy_run1 or not mypy_run2:
            print(f"Failed to load mypy results for {model_name}")
            continue
        
        # Filter: only files that pass mypy in both runs
        mypy_successful_files = set()
        for filename in baseline_files:
            if filename in mypy_run1 and filename in mypy_run2:
                if mypy_run1[filename] and mypy_run2[filename]:
                    mypy_successful_files.add(filename)
        
        print(f"{model_name}: {len(mypy_successful_files)} files pass mypy in both runs")
        
        # Extract common slots (filtered by mypy successful files)
        common_slots, run1_slots, run2_slots = extract_common_type_slots(
            run1_data, run2_data, mypy_successful_files
        )
        
        if not common_slots:
            print(f"No common slots found for {model_name}")
            continue
        
        # Analyze Any predictions in truth table format
        (any_both, any_run1_only, any_run2_only, 
         neither_run1_more_precise, neither_run2_more_precise, 
         neither_both_identical, neither_not_comparable, not_comparable_pairs) = analyze_any_truth_table(
            common_slots, run1_slots, run2_slots
        )
        
        total_common = len(common_slots)
        any_neither_total = (neither_run1_more_precise + neither_run2_more_precise + 
                             neither_both_identical + neither_not_comparable)
        
        # Analyze not comparable pairs
        not_comparable_analysis = analyze_not_comparable(not_comparable_pairs)
        
        results[model_name] = {
            "total_common": total_common,
            "any_both": any_both,
            "any_run1_only": any_run1_only,
            "any_run2_only": any_run2_only,
            "any_neither_total": any_neither_total,
            "neither_run1_more_precise": neither_run1_more_precise,
            "neither_run2_more_precise": neither_run2_more_precise,
            "neither_both_identical": neither_both_identical,
            "neither_not_comparable": neither_not_comparable,
            "not_comparable_analysis": not_comparable_analysis
        }
    
    # Print truth table
    print(f"\n{'Model':<15} {'Total':<8} {'Any Both':<12} {'Any 1st':<12} {'Any 2nd':<12} {'Neither Total':<15}")
    print("-" * 90)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['total_common']:<8} {data['any_both']:<12} "
              f"{data['any_run1_only']:<12} {data['any_run2_only']:<12} {data['any_neither_total']:<15}")
    
    # Print breakdown of Any Neither
    print(f"\n{'Model':<15} {'Neither Total':<15} {'1st Precise':<15} {'2nd Precise':<15} {'Identical':<15} {'Not Comparable':<15}")
    print("-" * 90)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['any_neither_total']:<15} "
              f"{data['neither_run1_more_precise']:<15} {data['neither_run2_more_precise']:<15} "
              f"{data['neither_both_identical']:<15} {data['neither_not_comparable']:<15}")
    
    # Print analysis of not comparable cases
    print(f"\n{'='*80}")
    print("ANALYSIS: Why types are NOT COMPARABLE")
    print("(Same precision score but different types)")
    print(f"{'='*80}")
    
    for model_name, data in results.items():
        analysis = data.get("not_comparable_analysis", {})
        if analysis and analysis.get("total_pairs", 0) > 0:
            print(f"\n{model_name}:")
            print(f"  Total not comparable: {analysis['total_pairs']}")
            print(f"  Unique type pairs: {analysis['unique_pairs']}")
            print(f"  Top 10 most common pairs:")
            for (type1, type2), count in analysis.get("top_pairs", []):
                print(f"    '{type1}' vs '{type2}': {count} occurrences")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

