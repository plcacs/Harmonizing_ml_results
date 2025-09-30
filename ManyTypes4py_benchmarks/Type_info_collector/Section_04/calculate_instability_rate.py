import json
import os
import csv
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


def is_any_type(param_types):
    """Check if a parameter type is considered 'Any' (including empty types)."""
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        if isinstance(type_str, str) and type_str.strip():
            # Check if it's explicitly "Any"
            return type_str.strip().lower() == "any"
        else:
            # Empty string counts as Any
            return True
    else:
        # No type annotation counts as Any
        return True


def is_imprecise_type(type_str):
    """Check if a type is imprecise (contains Any or Optional with Any)."""
    if not type_str:
        return True
    
    type_str = type_str.strip()
    
    # Check for explicit Any
    if type_str.lower() == "any":
        return True
    
    # Check for List[Any], Dict[Any, Any], etc.
    if "any" in type_str.lower() and ("[" in type_str or "]" in type_str):
        return True
    
    # Check for Optional[Any]
    if "optional[any]" in type_str.lower():
        return True
    
    # Check for Union[Any, ...]
    if "union" in type_str.lower() and "any" in type_str.lower():
        return True
    
    return False


def extract_parameter_signatures(type_info_data, baseline_files=None):
    """Extract parameter signatures from type info data."""
    signatures = {}

    if not isinstance(type_info_data, dict):
        return signatures

    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    # Create a signature key for this function
                    signature_key = f"{filename}::{func_name}"
                    param_info = []

                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_types = param.get("type", [])
                            param_name = param.get("name", "")

                            param_info.append(
                                {
                                    "category": category,
                                    "name": param_name,
                                    "is_any": is_any_type(param_types),
                                }
                            )

                    signatures[signature_key] = param_info

    return signatures


def extract_parameter_signatures_with_types(type_info_data, baseline_files=None):
    """Extract parameter signatures with actual type information from type info data."""
    signatures = {}

    if not isinstance(type_info_data, dict):
        return signatures

    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    # Create a signature key for this function
                    signature_key = f"{filename}::{func_name}"
                    param_info = []

                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_types = param.get("type", [])
                            param_name = param.get("name", "")

                            # Extract actual type string
                            type_str = ""
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = str(param_types[0]) if param_types[0] else ""

                            param_info.append(
                                {
                                    "category": category,
                                    "name": param_name,
                                    "is_any": is_any_type(param_types),
                                    "type_str": type_str.strip(),
                                    "is_imprecise": is_imprecise_type(type_str.strip()),
                                }
                            )

                    signatures[signature_key] = param_info

    return signatures


def calculate_instability_rate(
    model_name, first_run_data, second_run_data, baseline_files=None
):
    """Calculate instability rate for a model between first and second runs."""
    # Extract signatures from both runs (filtered to baseline files)
    first_signatures = extract_parameter_signatures_with_types(first_run_data, baseline_files)
    second_signatures = extract_parameter_signatures_with_types(second_run_data, baseline_files)

    # Find common parameters (exist in both runs)
    common_signatures = set(first_signatures.keys()) & set(second_signatures.keys())

    total_params = 0
    any_both = 0
    any_first = 0
    any_second = 0
    any_neither = 0
    type_consistent = 0  # Parameters with same type in both runs
    
    # Subcategories for any_neither (typed in both runs)
    first_more_precise = 0  # First run more precise, second run uses Any/Optional
    second_more_precise = 0  # Second run more precise, first run uses Any/Optional
    neither_precise = 0  # Neither is precise or both equal precision

    for signature_key in common_signatures:
        first_params = first_signatures[signature_key]
        second_params = second_signatures[signature_key]

        # Match parameters by category and name
        first_param_map = {
            (p["category"], p["name"]): {"is_any": p["is_any"], "type_str": p["type_str"], "is_imprecise": p["is_imprecise"]} 
            for p in first_params
        }
        second_param_map = {
            (p["category"], p["name"]): {"is_any": p["is_any"], "type_str": p["type_str"], "is_imprecise": p["is_imprecise"]} 
            for p in second_params
        }

        # Find common parameters between the two runs
        common_param_keys = set(first_param_map.keys()) & set(second_param_map.keys())

        for param_key in common_param_keys:
            total_params += 1
            first_data = first_param_map[param_key]
            second_data = second_param_map[param_key]
            
            first_is_any = first_data["is_any"]
            second_is_any = second_data["is_any"]
            first_is_imprecise = first_data["is_imprecise"]
            second_is_imprecise = second_data["is_imprecise"]

            if first_is_any and second_is_any:
                any_both += 1
            elif first_is_any and not second_is_any:
                any_first += 1
            elif not first_is_any and second_is_any:
                any_second += 1
            else:  # not first_is_any and not second_is_any
                any_neither += 1
                # Check if the actual types are the same
                if first_data["type_str"] == second_data["type_str"]:
                    type_consistent += 1
                
                # Categorize precision differences within any_neither
                if not first_is_imprecise and second_is_imprecise:
                    first_more_precise += 1
                elif first_is_imprecise and not second_is_imprecise:
                    second_more_precise += 1
                else:  # Both imprecise or both precise
                    neither_precise += 1

    # Calculate instability rate
    instability_rate = (
        100 * ((any_first + any_second) / total_params) if total_params > 0 else 0
    )

    return {
        "model": model_name,
        "total_params": total_params,
        "any_both": any_both,
        "any_first": any_first,
        "any_second": any_second,
        "any_neither": any_neither,
        "type_consistent": type_consistent,
        "first_more_precise": first_more_precise,
        "second_more_precise": second_more_precise,
        "neither_precise": neither_precise,
        "instability_rate": instability_rate,
    }


def main():
    # Load baseline files (files with isCompiled=True)
    baseline_files = load_baseline_files()
    if not baseline_files:
        print("No baseline files found. Exiting.")
        return

    # Define model pairs (first and second runs)
    model_pairs = {
        "GPT35": {
            "first": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
        },
        "GPT4o": {
            "first": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_gpt4o_2nd_run_benchmarks.json",
        },
        "O1-mini": {
            "first": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_o1_mini_2nd_run_benchmarks.json",
        },
        "O3-mini": {
            "first": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_o3_mini_2nd_run_benchmarks.json",
        },
        "DeepSeek": {
            "first": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_deep_seek_2nd_run_benchmarks.json",
        },
        "Claude3-Sonnet": {
            "first": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "second": "../Type_info_LLMS/Type_info_claude3_sonnet_2nd_run_benchmarks.json",
        },
    }

    print("=" * 100)
    print(
        "INSTABILITY RATE ANALYSIS: Comparing Any-type consistency between 1st and 2nd runs"
    )
    print(f"BASELINE FILES: {len(baseline_files)} files with isCompiled=True")
    print("=" * 100)

    results = []

    for model_name, file_paths in model_pairs.items():
        print(f"\nProcessing {model_name}...")

        # Load data from both runs
        first_run_data = load_type_info(file_paths["first"])
        second_run_data = load_type_info(file_paths["second"])

        if not first_run_data or not second_run_data:
            print(f"  Failed to load data for {model_name}")
            continue

        # Calculate instability rate (filtered to baseline files)
        result = calculate_instability_rate(
            model_name, first_run_data, second_run_data, baseline_files
        )
        results.append(result)

        print(f"  Total common parameters: {result['total_params']:,}")
        print(f"  Any in both runs: {result['any_both']:,}")
        print(f"  Any in 1st run only: {result['any_first']:,}")
        print(f"  Any in 2nd run only: {result['any_second']:,}")
        print(f"  Not Any in both runs: {result['any_neither']:,}")
        print(f"    - Same type in both runs: {result['type_consistent']:,}")
        print(f"    - 1st run more precise: {result['first_more_precise']:,}")
        print(f"    - 2nd run more precise: {result['second_more_precise']:,}")
        print(f"    - Neither precise: {result['neither_precise']:,}")
        print(f"  Instability rate: {result['instability_rate']:.2f}%")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Model':<15} {'Total Params':<12} {'Any Both':<10} {'Any 1st':<10} {'Any 2nd':<10} {'Any Neither':<12} {'Same Type':<10} {'1st Precise':<11} {'2nd Precise':<11} {'Neither Precise':<15} {'Instability Rate':<15}"
    )
    print("-" * 150)

    for result in results:
        print(
            f"{result['model']:<15} {result['total_params']:<12,} {result['any_both']:<10,} {result['any_first']:<10,} {result['any_second']:<10,} {result['any_neither']:<12,} {result['type_consistent']:<10,} {result['first_more_precise']:<11,} {result['second_more_precise']:<11,} {result['neither_precise']:<15,} {result['instability_rate']:<15.2f}%"
        )

    # Save results to CSV
    output_file = "./instability_rate_results_baseline.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "Model",
                "Total_Parameters",
                "Any_Both",
                "Any_1st",
                "Any_2nd",
                "Any_Neither",
                "Same_Type",
                "First_More_Precise",
                "Second_More_Precise",
                "Neither_Precise",
                "Instability_Rate",
            ]
        )

        # Write data rows
        for result in results:
            writer.writerow(
                [
                    result["model"],
                    result["total_params"],
                    result["any_both"],
                    result["any_first"],
                    result["any_second"],
                    result["any_neither"],
                    result["type_consistent"],
                    result["first_more_precise"],
                    result["second_more_precise"],
                    result["neither_precise"],
                    f"{result['instability_rate']:.2f}%",
                ]
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
