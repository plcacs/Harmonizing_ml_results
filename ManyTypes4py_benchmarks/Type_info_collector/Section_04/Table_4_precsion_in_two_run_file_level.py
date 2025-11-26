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


def is_any_type(param_types):
    """Check if parameter type is Any or empty, matching Table_03_any_empty_rate.py logic."""
    is_any = False
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        if isinstance(type_str, str) and type_str.strip():
            if type_str.strip().lower() == "any":
                is_any = True
        else:
            is_any = True
    else:
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


def extract_file_slots(filename, functions, baseline_files):
    """Extract all parameter/return type slots for a file."""
    if filename not in baseline_files:
        return {}
    
    file_slots = {}
    
    if isinstance(functions, dict):
        for func_name, func_data in functions.items():
            if isinstance(func_data, list):
                for param in func_data:
                    if isinstance(param, dict):
                        param_name = param.get("name", "")
                        if param_name == "self":
                            continue
                        
                        category = param.get("category", "")
                        param_types = param.get("type", [])
                        slot_key = f"{func_name}:{category}:{param_name}"
                        file_slots[slot_key] = param_types
    
    return file_slots


def extract_file_level_slots(run_data, baseline_files):
    """Extract file-level slots for all files."""
    file_slots_dict = {}
    
    for filename, functions in run_data.items():
        file_slots = extract_file_slots(filename, functions, baseline_files)
        if file_slots:
            file_slots_dict[filename] = file_slots
    
    return file_slots_dict


def compare_file_precision(run1_slots, run2_slots):
    """Compare two files by calculating overall precision scores.
    Returns: 'run1_more_precise', 'run2_more_precise', 'identical', or 'incomparable'
    """
    # Find common slots
    common_slots = set(run1_slots.keys()) & set(run2_slots.keys())
    
    if not common_slots:
        return 'incomparable'
    
    # Calculate overall precision scores for each file
    run1_total_score = 0
    run2_total_score = 0
    
    for slot in common_slots:
        run1_param_types = run1_slots[slot]
        run2_param_types = run2_slots[slot]
        
        run1_type = get_type_string(run1_param_types)
        run2_type = get_type_string(run2_param_types)
        
        run1_score = get_precision_score(run1_type)
        run2_score = get_precision_score(run2_type)
        
        run1_total_score += run1_score
        run2_total_score += run2_score
    
    # Compare overall scores
    if run1_total_score > run2_total_score:
        return 'run1_more_precise'
    elif run2_total_score > run1_total_score:
        return 'run2_more_precise'
    else:
        # Scores are equal - compare parameters directly
        # If any parameter differs (either type or missing), mark as incomparable
        if (len(run1_slots) != len(run2_slots)) or (
            len(common_slots) != len(run1_slots)
        ):
            return 'incomparable'
        
        for slot in common_slots:
            run1_type = get_type_string(run1_slots[slot])
            run2_type = get_type_string(run2_slots[slot])
            
            if run1_type.lower() != run2_type.lower():
                return 'incomparable'
        
        # All parameters match exactly
        return 'identical'


def analyze_file_level_truth_table(run1_file_slots, run2_file_slots):
    """Analyze file-level precision comparison.
    A file P1 is more precise than P2 if each parameter annotation in P1 is more precise than the corresponding annotation in P2.
    """
    any_both = 0
    any_run1_only = 0
    any_run2_only = 0
    neither_run1_more_precise = 0
    neither_run2_more_precise = 0
    neither_both_identical = 0
    neither_not_comparable = 0
    
    # Find common files
    common_files = set(run1_file_slots.keys()) & set(run2_file_slots.keys())
    
    not_comparable_files = []
    
    for filename in common_files:
        run1_slots = run1_file_slots[filename]
        run2_slots = run2_file_slots[filename]
        
        # Check if files are "Any-heavy" (all parameters are Any)
        run1_all_any = all(is_any_type(run1_slots.get(slot, [])) for slot in run1_slots.keys())
        run2_all_any = all(is_any_type(run2_slots.get(slot, [])) for slot in run2_slots.keys())
        
        if run1_all_any and run2_all_any:
            any_both += 1
        elif run1_all_any and not run2_all_any:
            any_run1_only += 1
        elif not run1_all_any and run2_all_any:
            any_run2_only += 1
        else:  # neither is all-any - compare precision
            comparison = compare_file_precision(run1_slots, run2_slots)
            
            if comparison == 'run1_more_precise':
                neither_run1_more_precise += 1
            elif comparison == 'run2_more_precise':
                neither_run2_more_precise += 1
            elif comparison == 'identical':
                neither_both_identical += 1
            else:  # incomparable
                neither_not_comparable += 1
                not_comparable_files.append(filename)
    
    return (any_both, any_run1_only, any_run2_only, 
            neither_run1_more_precise, neither_run2_more_precise, 
            neither_both_identical, neither_not_comparable, not_comparable_files)


def analyze_not_comparable(not_comparable_files):
    """Analyze why files are not comparable."""
    if not not_comparable_files:
        return {}
    
    return {
        "total_files": len(not_comparable_files),
        "note": "Files where some parameters are more precise in run1, others in run2"
    }


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
    mypy_files = {
        "Human": "../../mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json",
        "GPT35_1st_run": "../../mypy_results/mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
        "GPT35_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
        "GPT4o_1st_run": "../../mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
        "GPT4o_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json",
        "O1-mini_1st_run": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
        "O1-mini_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json",
        "O3-mini_1st_run": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "O3-mini_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json",
        "DeepSeek_1st_run": "../../mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
        "DeepSeek_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json",
        "Claude3-Sonnet_1st_run": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        "Claude3-Sonnet_2nd_run": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json",
    }
    
    print("=" * 80)
    print("FILE-LEVEL RUN-TO-RUN COMPARISON (TRUTH TABLE)")
    print(f"BASELINE FILES: {len(baseline_files)} files with isCompiled=True")
    print("FILTER: Only files that pass mypy type checking in BOTH runs are analyzed")
    print("Note: Files where ALL parameters are Any are considered 'Any'")
    print("Precision: File P1 is more precise than P2 if each parameter in P1 is more precise than corresponding parameter in P2")
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
        
        # Extract file-level slots (filtered by mypy successful files)
        run1_file_slots = extract_file_level_slots(run1_data, mypy_successful_files)
        run2_file_slots = extract_file_level_slots(run2_data, mypy_successful_files)
        
        if not run1_file_slots or not run2_file_slots:
            print(f"No file slots found for {model_name}")
            continue
        
        # Analyze file-level truth table
        (any_both, any_run1_only, any_run2_only, 
         neither_run1_more_precise, neither_run2_more_precise, 
         neither_both_identical, neither_not_comparable, not_comparable_files) = analyze_file_level_truth_table(
            run1_file_slots, run2_file_slots
        )
        
        common_files = len(set(run1_file_slots.keys()) & set(run2_file_slots.keys()))
        any_neither_total = (neither_run1_more_precise + neither_run2_more_precise + 
                             neither_both_identical + neither_not_comparable)
        
        # Analyze not comparable files
        not_comparable_analysis = analyze_not_comparable(not_comparable_files)
        
        results[model_name] = {
            "common_files": common_files,
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
    print(f"\n{'Model':<15} {'Files':<8} {'Any Both':<12} {'Any 1st':<12} {'Any 2nd':<12} {'Neither Total':<15}")
    print("-" * 90)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['common_files']:<8} {data['any_both']:<12} "
              f"{data['any_run1_only']:<12} {data['any_run2_only']:<12} {data['any_neither_total']:<15}")
    
    # Print breakdown of Any Neither
    print(f"\n{'Model':<15} {'Neither Total':<15} {'1st Precise':<15} {'2nd Precise':<15} {'Identical':<15} {'Not Comparable':<15}")
    print("-" * 90)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['any_neither_total']:<15} "
              f"{data['neither_run1_more_precise']:<15} {data['neither_run2_more_precise']:<15} "
              f"{data['neither_both_identical']:<15} {data['neither_not_comparable']:<15}")
    
    # Print analysis of not comparable cases
    """
    print(f"\n{'='*80}")
    print("ANALYSIS: Why files are INCOMPARABLE")
    print("(Files where some parameters are more precise in run1, others in run2)")
    print(f"{'='*80}")
    
    for model_name, data in results.items():
        analysis = data.get("not_comparable_analysis", {})
        if analysis and analysis.get("total_files", 0) > 0:
            print(f"\n{model_name}:")
            print(f"  Total incomparable: {analysis['total_files']}")
            if "note" in analysis:
                print(f"  Note: {analysis['note']}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")"""


if __name__ == "__main__":
    main()

