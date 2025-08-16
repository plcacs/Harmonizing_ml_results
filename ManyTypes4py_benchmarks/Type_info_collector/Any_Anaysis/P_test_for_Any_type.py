import json
import pandas as pd
import csv
from scipy import stats
from collections import defaultdict


def load_json_file(file_path):
    """Load JSON file and handle potential errors."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_any_counts(type_info_data):
    """Extract 'Any' type counts per file from type info data."""
    any_counts = defaultdict(int)

    if isinstance(type_info_data, dict):
        for filename, functions in type_info_data.items():
            if isinstance(functions, dict):
                for func_name, func_data in functions.items():
                    if isinstance(func_data, list):
                        for param in func_data:
                            if isinstance(param, dict):
                                category = param.get("category", "")
                                if (
                                    category == "arg"
                                ):  # Only count parameter annotations
                                    param_types = param.get("type", [])
                                    if isinstance(param_types, list):
                                        for type_annotation in param_types:
                                            if (
                                                isinstance(type_annotation, str)
                                                and type_annotation == "Any"
                                            ):
                                                any_counts[filename] += 1

    return any_counts


def extract_mypy_results_with_stats(mypy_data):
    """Extract compilation status from mypy results."""
    results = {}

    if isinstance(mypy_data, dict):
        for filename, data in mypy_data.items():
            if isinstance(data, dict):
                results[filename] = {
                    "isCompiled": data.get("isCompiled", False),
                    "parameters_with_annotations": data.get("stats", {}).get(
                        "parameters_with_annotations", 0
                    ),
                }

    return results


def main():
    # Define model files
    typed_info_files = {
        "Human": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_original_files.json",
        "GPT4O": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
        "O1-mini": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_o1_mini_benchmarks.json",
        "O3-mini": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "DeepSeek": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        "Claude3-Sonnet": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
    }

    mypy_files = {
        "Human": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json",
        "GPT4O": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
        "O1-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
        "O3-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "DeepSeek": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
        "Claude3-Sonnet": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    }

    print("=" * 60)
    print("SIMPLE CORRELATION: Any Count vs Compilation Success")
    print("=" * 60)

    results = []

    for model_name in typed_info_files.keys():
        print(f"\nProcessing {model_name}...")

        # Load data
        type_info_data = load_json_file(typed_info_files[model_name])
        mypy_data = load_json_file(mypy_files[model_name])

        if not type_info_data or not mypy_data:
            print(f"  Failed to load data for {model_name}")
            continue

        # Extract data
        mypy_results = extract_mypy_results_with_stats(mypy_data)
        any_counts = extract_any_counts(type_info_data)

        # Prepare data
        any_counts_list = []
        compilation_success_list = []

        for filename in mypy_results.keys():
            if filename in any_counts:
                any_count = any_counts[filename]
                parameters = mypy_results[filename]["parameters_with_annotations"]

                # Only include valid data
                if any_count <= parameters:
                    any_counts_list.append(any_count)
                    compilation_success_list.append(
                        1 if mypy_results[filename]["isCompiled"] else 0
                    )

        if len(any_counts_list) < 10:
            print(f"  Insufficient data for {model_name}: {len(any_counts_list)} files")
            continue

        # Perform Point-Biserial correlation test
        correlation, p_value = stats.pointbiserialr(
            compilation_success_list, any_counts_list
        )

        # Calculate basic stats
        success_rate = sum(compilation_success_list) / len(compilation_success_list)
        mean_any = sum(any_counts_list) / len(any_counts_list)

        results.append(
            {
                "Model": model_name,
                "Files": len(any_counts_list),
                "Success_Rate": success_rate,
                "Mean_Any": mean_any,
                "Correlation": correlation,
                "P_Value": p_value,
            }
        )

        print(f"  Files: {len(any_counts_list)}")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Mean Any Count: {mean_any:.2f}")
        print(f"  Correlation: r={correlation:.3f}, p={p_value:.6f}")

    # Save to CSV
    output_file = "simple_correlation_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Files", "Success_Rate", "Mean_Any", "Correlation", "P_Value"]
        )

        for result in results:
            writer.writerow(
                [
                    result["Model"],
                    result["Files"],
                    f"{result['Success_Rate']:.6f}",
                    f"{result['Mean_Any']:.6f}",
                    f"{result['Correlation']:.6f}",
                    f"{result['P_Value']:.6f}",
                ]
            )

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Model':<15} {'Files':<8} {'Success':<8} {'Any_Mean':<10} {'Correlation':<12} {'P_Value':<10}"
    )
    print("-" * 70)

    for result in results:
        print(
            f"{result['Model']:<15} {result['Files']:<8} {result['Success_Rate']:<8.3f} {result['Mean_Any']:<10.2f} {result['Correlation']:<12.3f} {result['P_Value']:<10.6f}"
        )


if __name__ == "__main__":
    main()
