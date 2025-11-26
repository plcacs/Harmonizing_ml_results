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


def calculate_any_rate(type_info_data, baseline_files=None):
    """Calculate Any-rate: #Any_slots / #total_params (including empty types as Any)."""
    any_slots = 0
    total_params = 0

    if not isinstance(type_info_data, dict):
        return 0, 0, 0

    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            # Skip 'self' parameter completely
                            param_name = param.get("name", "")
                            if param_name == "self":
                                continue
                            # Count all parameters
                            total_params += 1

                            # Get type annotations
                            param_types = param.get("type", [])

                            # Check if it's Any or empty
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

                            if is_any:
                                any_slots += 1

    any_rate = any_slots / total_params if total_params > 0 else 0
    return any_slots, total_params, any_rate


def calculate_any_rate_by_category(type_info_data, baseline_files=None):
    """Calculate Any-rate separately for parameters and return types (including empty types as Any)."""
    param_any_slots = 0
    param_total_slots = 0
    return_any_slots = 0
    return_total_slots = 0

    if not isinstance(type_info_data, dict):
        return 0, 0, 0, 0, 0, 0

    for filename, functions in type_info_data.items():
        # Only process files that are in baseline_files (if provided)
        if baseline_files is not None and filename not in baseline_files:
            continue

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")

                            # Skip 'self' parameter completely
                            if param_name == "self":
                                continue
                            param_types = param.get("type", [])

                            # Check if it's Any or empty
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

                            if category == "arg":  # Parameter
                                param_total_slots += 1
                                if is_any:
                                    param_any_slots += 1
                            elif category == "return":  # Return type
                                return_total_slots += 1
                                if is_any:
                                    return_any_slots += 1

    param_any_rate = param_any_slots / param_total_slots if param_total_slots > 0 else 0
    return_any_rate = (
        return_any_slots / return_total_slots if return_total_slots > 0 else 0
    )

    return (
        param_any_slots,
        param_total_slots,
        param_any_rate,
        return_any_slots,
        return_total_slots,
        return_any_rate,
    )


def compute_per_file_any_percentage(type_info_data, baseline_files=None):
    """Compute per-file Any percentage.
    Returns dict: { filename: {"any_slots": int, "total_slots": int, "type_annotationed_slots": int, "any_percentage": float, "type_annotationed_percentage": float} }
    """
    results = {}

    if not isinstance(type_info_data, dict):
        return results

    for filename, functions in type_info_data.items():
        if baseline_files is not None and filename not in baseline_files:
            continue

        any_slots = 0
        total_slots = 0
        type_annotated_slots = 0

        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            # Skip 'self' parameter completely
                            param_name = param.get("name", "")
                            if param_name == "self":
                                continue

                            total_slots += 1

                            param_types = param.get("type", [])

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
                                any_slots += 1
                            elif is_type_annotated:
                                type_annotated_slots += 1

        any_percentage = (any_slots / total_slots * 100.0) if total_slots > 0 else 0.0
        type_annotated_percentage = (
            (type_annotated_slots / total_slots * 100.0) if total_slots > 0 else 0.0
        )

        results[filename] = {
            "total_slots": total_slots,
            "any_slots": any_slots,
            "type_annotationed_slots": type_annotated_slots,
            "type_annotationed_percentage": type_annotated_percentage,
            "any_percentage": any_percentage,
        }

    return results


def save_per_run_file_any_percentages(
    model_files, output_root="./per_file_any_percentage", baseline_files=None
):
    """Save per-file Any percentages for each run/model into separate subfolders.
    - model_files: dict like in main() mapping model_name -> path
    - output_root: directory where subfolders per model_name will be created
    - baseline_files: optional set of filenames to filter
    """
    os.makedirs(output_root, exist_ok=True)

    for model_name, filepath in model_files.items():
        type_info = load_type_info(filepath)
        if not type_info:
            continue

        per_file = compute_per_file_any_percentage(type_info, baseline_files)

        model_dir = os.path.join(output_root, model_name)
        os.makedirs(model_dir, exist_ok=True)

        out_path = os.path.join(model_dir, "per_file_any_percentage.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_file, f, ensure_ascii=False, indent=2, sort_keys=True)

    return True


def main():
    # Load baseline files (files with isCompiled=True)
    baseline_files = load_baseline_files()
    baseline = load_baseline_files()

    if not baseline_files:
        print("No baseline files found. Exiting.")
        return

    # Define model files (using relative paths) - ordered by preference
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
        "GPT5_1st_run": "../Type_info_LLMS/Type_info_gpt5_1st_run_benchmarks.json",
    }
    
    save_per_run_file_any_percentages(model_files, baseline_files=baseline)
    print("=" * 80)
    print(
        "ANY-RATE ANALYSIS: Any_rate = #Any_slots / #total_params (including empty types as Any)"
    )
    print(f"BASELINE FILES: {len(baseline_files)} files with isCompiled=True")
    print("=" * 80)

    results = {}

    for model_name, filename in model_files.items():
        print(f"\nProcessing {model_name}...")

        # Load data
        type_info_data = load_type_info(filename)
        if not type_info_data:
            print(f"  Failed to load {filename}")
            continue

        # Calculate overall Any-rate (filtered to baseline files)
        any_slots, total_params, any_rate = calculate_any_rate(
            type_info_data, baseline_files
        )

        # Calculate category-specific Any-rates (filtered to baseline files)
        param_any, param_total, param_rate, return_any, return_total, return_rate = (
            calculate_any_rate_by_category(type_info_data, baseline_files)
        )

        results[model_name] = {
            "any_slots": any_slots,
            "total_params": total_params,
            "any_rate": any_rate,
            "param_any": param_any,
            "param_total": param_total,
            "param_rate": param_rate,
            "return_any": return_any,
            "return_total": return_total,
            "return_rate": return_rate,
        }

        print(
            f"  Overall: {any_slots:,} Any / {total_params:,} total = {any_rate:.3f} ({any_rate*100:.1f}%)"
        )
        print(
            f"  Parameters: {param_any:,} Any / {param_total:,} total = {param_rate:.3f} ({param_rate*100:.1f}%)"
        )
        print(
            f"  Returns: {return_any:,} Any / {return_total:,} total = {return_rate:.3f} ({return_rate*100:.1f}%)"
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<15} {'Any-rate':<10} {'Param Any':<12} {'Return Any':<12}")
    print("-" * 80)

    for model_name, data in results.items():
        print(
            f"{model_name:<15} {data['any_rate']:<10.3f} {data['param_rate']:<12.3f} {data['return_rate']:<12.3f}"
        )

    # Calculate delta vs human
    if "Human" in results:
        human_rate = results["Human"]["any_rate"]
        print(f"\n{'Model':<15} {'Any-rate':<10} {'Δ vs Human':<12}")
        print("-" * 40)
        print(f"{'Human':<15} {human_rate:<10.3f} {'—':<12}")

        for model_name, data in results.items():
            if model_name != "Human":
                delta = data["any_rate"] - human_rate
                print(f"{model_name:<15} {data['any_rate']:<10.3f} {delta:<+12.3f}")

    # Save results to CSV
    output_file = "./any_empty_rate_results_baseline.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "Model",
                "Any_slots",
                "Total_params",
                "Any_rate",
                "Param_any",
                "Param_total",
                "Param_rate",
                "Return_any",
                "Return_total",
                "Return_rate",
                "Delta_vs_human",
            ]
        )

        # Write data rows
        human_rate = results.get("Human", {}).get("any_rate", 0)
        for model_name, data in results.items():
            delta = data["any_rate"] - human_rate if model_name != "Human" else 0
            writer.writerow(
                [
                    model_name,
                    data["any_slots"],
                    data["total_params"],
                    f"{data['any_rate']*100:.2f}%",
                    data["param_any"],
                    data["param_total"],
                    f"{data['param_rate']*100:.2f}%",
                    data["return_any"],
                    data["return_total"],
                    f"{data['return_rate']*100:.2f}%",
                    f"{delta*100:+.2f}%" if model_name != "Human" else "—",
                ]
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
