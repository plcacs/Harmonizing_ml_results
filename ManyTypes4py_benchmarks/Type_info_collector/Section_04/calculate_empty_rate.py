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


def calculate_empty_rate(type_info_data):
    """Calculate Empty-rate: #empty_slots / #total_slots."""
    empty_slots = 0
    total_slots = 0

    if not isinstance(type_info_data, dict):
        return 0, 0, 0

    for filename, functions in type_info_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            # Count as total slot
                            total_slots += 1

                            # Get type annotations
                            param_types = param.get("type", [])
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    # Has type annotation, not empty
                                    pass
                                else:
                                    # Empty type annotation
                                    empty_slots += 1
                            else:
                                # No type annotation at all
                                empty_slots += 1

    empty_rate = empty_slots / total_slots if total_slots > 0 else 0
    return empty_slots, total_slots, empty_rate


def calculate_empty_rate_by_category(type_info_data):
    """Calculate Empty-rate separately for parameters and return types."""
    param_empty_slots = 0
    param_total_slots = 0
    return_empty_slots = 0
    return_total_slots = 0

    if not isinstance(type_info_data, dict):
        return 0, 0, 0, 0, 0, 0

    for filename, functions in type_info_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_types = param.get("type", [])

                            if category == "arg":  # Parameter
                                param_total_slots += 1
                                if isinstance(param_types, list) and len(param_types) > 0:
                                    type_str = param_types[0]
                                    if not (isinstance(type_str, str) and type_str.strip()):
                                        param_empty_slots += 1
                                else:
                                    param_empty_slots += 1
                            elif category == "return":  # Return type
                                return_total_slots += 1
                                if isinstance(param_types, list) and len(param_types) > 0:
                                    type_str = param_types[0]
                                    if not (isinstance(type_str, str) and type_str.strip()):
                                        return_empty_slots += 1
                                else:
                                    return_empty_slots += 1

    param_empty_rate = param_empty_slots / param_total_slots if param_total_slots > 0 else 0
    return_empty_rate = (
        return_empty_slots / return_total_slots if return_total_slots > 0 else 0
    )

    return (
        param_empty_slots,
        param_total_slots,
        param_empty_rate,
        return_empty_slots,
        return_total_slots,
        return_empty_rate,
    )


def main():
    # Define model files (using relative paths) - ordered by preference
    model_files = {
        "Human": "../Type_info_LLMS/Type_info_original_files.json",
        "GPT35_1st_run": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
        "GPT35_2nd_run": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
        "GPT4o_1st_run": "../Type_info_LLMS/Type_info_gpt4o_1st_run_benchmarks.json",
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

    print("=" * 80)
    print("EMPTY-RATE ANALYSIS: Empty_rate = #empty_slots / #total_slots")
    print("=" * 80)

    results = {}

    for model_name, filename in model_files.items():
        print(f"\nProcessing {model_name}...")

        # Load data
        type_info_data = load_type_info(filename)
        if not type_info_data:
            print(f"  Failed to load {filename}")
            continue

        # Calculate overall Empty-rate
        empty_slots, total_slots, empty_rate = calculate_empty_rate(type_info_data)

        # Calculate category-specific Empty-rates
        param_empty, param_total, param_rate, return_empty, return_total, return_rate = (
            calculate_empty_rate_by_category(type_info_data)
        )

        results[model_name] = {
            "empty_slots": empty_slots,
            "total_slots": total_slots,
            "empty_rate": empty_rate,
            "param_empty": param_empty,
            "param_total": param_total,
            "param_rate": param_rate,
            "return_empty": return_empty,
            "return_total": return_total,
            "return_rate": return_rate,
        }

        print(
            f"  Overall: {empty_slots:,} empty / {total_slots:,} total = {empty_rate*100:.1f}%"
        )
        print(
            f"  Parameters: {param_empty:,} empty / {param_total:,} total = {param_rate*100:.1f}%"
        )
        print(
            f"  Returns: {return_empty:,} empty / {return_total:,} total = {return_rate*100:.1f}%"
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<15} {'Empty-rate':<12} {'Param Empty':<14} {'Return Empty':<14}")
    print("-" * 80)

    for model_name, data in results.items():
        print(
            f"{model_name:<15} {data['empty_rate']*100:<12.1f}% {data['param_rate']*100:<14.1f}% {data['return_rate']*100:<14.1f}%"
        )

    # Calculate delta vs human
    if "Human" in results:
        human_rate = results["Human"]["empty_rate"]
        print(f"\n{'Model':<15} {'Empty-rate':<12} {'Δ vs Human':<12}")
        print("-" * 40)
        print(f"{'Human':<15} {human_rate*100:<12.1f}% {'—':<12}")

        for model_name, data in results.items():
            if model_name != "Human":
                delta = data["empty_rate"] - human_rate
                print(f"{model_name:<15} {data['empty_rate']*100:<12.1f}% {delta*100:<+12.1f}%")

    # Save results to CSV
    output_file = "./empty_rate_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "Model",
                "Empty_slots",
                "Total_slots",
                "Empty_rate",
                "Param_empty",
                "Param_total",
                "Param_rate",
                "Return_empty",
                "Return_total",
                "Return_rate",
                "Delta_vs_human",
            ]
        )

        # Write data rows
        human_rate = results.get("Human", {}).get("empty_rate", 0)
        for model_name, data in results.items():
            delta = data["empty_rate"] - human_rate if model_name != "Human" else 0
            writer.writerow(
                [
                    model_name,
                    data["empty_slots"],
                    data["total_slots"],
                    f"{data['empty_rate']*100:.2f}%",
                    data["param_empty"],
                    data["param_total"],
                    f"{data['param_rate']*100:.2f}%",
                    data["return_empty"],
                    data["return_total"],
                    f"{data['return_rate']*100:.2f}%",
                    f"{delta*100:+.2f}%" if model_name != "Human" else "—",
                ]
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
