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


def load_mypy_results(mypy_file_path):
    """Load mypy results and return compilation status for each file."""
    try:
        with open(mypy_file_path, "r", encoding="utf-8") as f:
            mypy_data = json.load(f)
        return {fname: info.get("isCompiled", False) for fname, info in mypy_data.items()}
    except Exception as e:
        print(f"Error loading mypy results from {mypy_file_path}: {e}")
        return {}


def get_filtered_files(baseline_files, mypy_compilation_status):
    """Split baseline files into successful and failed based on mypy compilation status."""
    successful_files = set()
    failed_files = set()
    
    for filename in baseline_files:
        if filename in mypy_compilation_status:
            if mypy_compilation_status[filename]:
                successful_files.add(filename)
            else:
                failed_files.add(filename)
    
    return successful_files, failed_files


def calculate_type_annotation_stats(type_info_data, baseline_files=None):
    """Calculate comprehensive type annotation statistics."""
    stats = {
        'any_slots': 0,
        'total_slots': 0,
        'type_annotated_slots': 0,
        'param_any_slots': 0,
        'param_total_slots': 0,
        'return_any_slots': 0,
        'return_total_slots': 0,
        'files_processed': 0,
        'functions_processed': 0
    }

    if not isinstance(type_info_data, dict):
        return stats

    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue

        stats['files_processed'] += 1

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    stats['functions_processed'] += 1
                    for param in func_data:
                        if isinstance(param, dict):
                            # Skip 'self' parameter completely
                            param_name = param.get("name", "")
                            if param_name == "self":
                                continue

                            category = param.get("category", "")
                            param_types = param.get("type", [])

                            # Count all slots
                            stats['total_slots'] += 1

                            # Check if it's Any or empty
                            is_any = False
                            is_type_annotated = False

                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    if type_str.strip().lower() == "any":
                                        is_any = True
                                    else:
                                        # Has a non-empty, non-Any type annotation
                                        is_type_annotated = True
                                else:
                                    # Empty string counts as Any
                                    is_any = True
                            else:
                                # No type annotation counts as Any
                                is_any = True

                            if is_any:
                                stats['any_slots'] += 1
                                if category == "arg":
                                    stats['param_any_slots'] += 1
                                elif category == "return":
                                    stats['return_any_slots'] += 1
                            elif is_type_annotated:
                                stats['type_annotated_slots'] += 1

                            # Count by category
                            if category == "arg":
                                stats['param_total_slots'] += 1
                            elif category == "return":
                                stats['return_total_slots'] += 1

    return stats


def calculate_rates(stats):
    """Calculate rates from statistics."""
    rates = {}
    
    # Overall rates
    rates['any_rate'] = stats['any_slots'] / stats['total_slots'] if stats['total_slots'] > 0 else 0
    rates['type_annotated_rate'] = stats['type_annotated_slots'] / stats['total_slots'] if stats['total_slots'] > 0 else 0
    
    # Parameter rates
    rates['param_any_rate'] = stats['param_any_slots'] / stats['param_total_slots'] if stats['param_total_slots'] > 0 else 0
    
    # Return type rates
    rates['return_any_rate'] = stats['return_any_slots'] / stats['return_total_slots'] if stats['return_total_slots'] > 0 else 0
    
    return rates


def print_detailed_analysis(model_name, stats, rates, file_type=""):
    """Print detailed analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR: {model_name} - {file_type}")
    print(f"{'='*60}")
    
    print(f"Files processed: {stats['files_processed']:,}")
    print(f"Functions processed: {stats['functions_processed']:,}")
    print(f"Total type slots: {stats['total_slots']:,}")
    
    print(f"\nOVERALL TYPE ANNOTATION QUALITY:")
    print(f"  Any slots: {stats['any_slots']:,} ({rates['any_rate']*100:.1f}%)")
    print(f"  Type annotated: {stats['type_annotated_slots']:,} ({rates['type_annotated_rate']*100:.1f}%)")
    
    print(f"\nPARAMETER TYPE ANNOTATIONS:")
    print(f"  Any parameters: {stats['param_any_slots']:,} / {stats['param_total_slots']:,} ({rates['param_any_rate']*100:.1f}%)")
    
    print(f"\nRETURN TYPE ANNOTATIONS:")
    print(f"  Any returns: {stats['return_any_slots']:,} / {stats['return_total_slots']:,} ({rates['return_any_rate']*100:.1f}%)")


def print_comparison_table(results, title):
    """Print comparison table across all models."""
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}")
    print(f"{'Model':<20} {'Total':<8} {'Any Slots':<12} {'Any Rate':<12} {'Param Any Ratio':<18} {'Return Any Ratio':<18}")
    print("-" * 120)
    
    for model_name, data in results.items():
        print(f"{model_name:<20} {data['total_slots']:<8,} {data['any_slots']:<12,} {data['any_rate']:<12.3f} {data['param_any_rate']:<18.3f} {data['return_any_rate']:<18.3f}")


def print_delta_analysis(results, title):
    """Print delta analysis compared to Human baseline."""
    if "Human" not in results:
        print(f"\nNo Human baseline found for delta analysis in {title}.")
        return
        
    human_data = results["Human"]
    print(f"\n{'='*80}")
    print(f"DELTA ANALYSIS (vs Human baseline) - {title}")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Any Rate Δ':<15} {'Type Annotated Δ':<18} {'Param Any Δ':<15} {'Return Any Δ':<15}")
    print("-" * 80)
    print(f"{'Human (baseline)':<20} {'—':<15} {'—':<18} {'—':<15} {'—':<15}")
    
    for model_name, data in results.items():
        if model_name != "Human":
            any_delta = data['any_rate'] - human_data['any_rate']
            type_annotated_delta = data['type_annotated_rate'] - human_data['type_annotated_rate']
            param_delta = data['param_any_rate'] - human_data['param_any_rate']
            return_delta = data['return_any_rate'] - human_data['return_any_rate']
            
            print(f"{model_name:<20} {any_delta:<+15.3f} {type_annotated_delta:<+18.3f} {param_delta:<+15.3f} {return_delta:<+15.3f}")


def analyze_model_files(model_name, type_info_data, file_filter, file_type):
    """Analyze a single model with given file filter."""
    print(f"\nProcessing {model_name} - {file_type}...")
    
    # Calculate statistics with file filter
    stats = calculate_type_annotation_stats(type_info_data, file_filter)
    rates = calculate_rates(stats)
    
    # Print detailed analysis
    #print_detailed_analysis(model_name, stats, rates, file_type)
    
    return {**stats, **rates}


def main():
    # Load baseline files (files with isCompiled=True)
    baseline_files = load_baseline_files()

    if not baseline_files:
        print("No baseline files found. Exiting.")
        return

    # Define model files and mypy results
    model_files = {
        "Human": "../Type_info_LLMS/Type_info_original_files.json",
        "GPT35_1st_run": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
        "GPT35_2nd_run": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
        "GPT4o_1st_run": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
        "GPT4o_2nd_run": "../Type_info_LLMS/Type_info_gpt4o_2nd_run_benchmarks.json",
        "O1-mini_1st_run": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
        "O1-mini_2nd_run": "../Type_info_LLMS/Type_info_o1_mini_2nd_run_benchmarks.json",
        "O3-mini_1st_run": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "O3-mini_2nd_run": "../Type_info_LLMS/Type_info_o3_mini_2nd_run_benchmarks.json",
        "DeepSeek_1st_run": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        "DeepSeek_2nd_run": "../Type_info_LLMS/Type_info_deep_seek_2nd_run_benchmarks.json",
        "Claude3-Sonnet_1st_run": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
        "Claude3-Sonnet_2nd_run": "../Type_info_LLMS/Type_info_claude3_sonnet_2nd_run_benchmarks.json",
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
    print("TWO-PART TYPE ANNOTATION QUALITY ANALYSIS")
    print("Any-rate = #Any_slots / #total_params (including empty types as Any)")
    print(f"BASELINE FILES: {len(baseline_files)} files with isCompiled=True")
    print("=" * 80)

    # Initialize results storage
    successful_results = {}
    failed_results = {}

    # Process each model for both successful and failed files
    for model_name in model_files.keys():
        # Load type info data
        type_info_data = load_type_info(model_files[model_name])
        if not type_info_data:
            print(f"Failed to load type info for {model_name}")
            continue

        # Load mypy results for this model
        mypy_compilation_status = load_mypy_results(mypy_files[model_name])
        if not mypy_compilation_status:
            print(f"Failed to load mypy results for {model_name}")
            continue

        # Split baseline files into successful and failed
        successful_files, failed_files = get_filtered_files(baseline_files, mypy_compilation_status)
        
        print(f"\n{model_name}: {len(successful_files)} successful, {len(failed_files)} failed files")

        # Analyze successful files
        if successful_files:
            successful_results[model_name] = analyze_model_files(
                model_name, type_info_data, successful_files, "SUCCESSFUL FILES"
            )

        # Analyze failed files
        if failed_files:
            failed_results[model_name] = analyze_model_files(
                model_name, type_info_data, failed_files, "FAILED FILES"
            )

    # Print comparison tables
    if successful_results:
        print_comparison_table(successful_results, "SUCCESSFULLY TYPE-CHECKED FILES")
        #print_delta_analysis(successful_results, "SUCCESSFULLY TYPE-CHECKED FILES")
    
    if failed_results:
        print_comparison_table(failed_results, "FAILED TYPE-CHECKED FILES")
        #print_delta_analysis(failed_results, "FAILED TYPE-CHECKED FILES")
    
    print(f"\n{'='*80}")
    print("TWO-PART ANALYSIS COMPLETE - No files saved (console output only)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
