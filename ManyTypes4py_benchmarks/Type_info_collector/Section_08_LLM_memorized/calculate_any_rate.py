import json
import os
import csv
from collections import defaultdict


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


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_any_rate(type_info_data):
    """Calculate Any-rate: #Any_slots / #typed_slots."""
    any_slots = 0
    typed_slots = 0

    if not isinstance(type_info_data, dict):
        return 0, 0, 0

    for filename, functions in type_info_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
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

                                    # Check if it's Any
                                    if type_str.strip().lower() == "any":
                                        any_slots += 1

    any_rate = any_slots / typed_slots if typed_slots > 0 else 0
    return any_slots, typed_slots, any_rate


def calculate_any_rate_by_category(type_info_data):
    """Calculate Any-rate separately for parameters and return types."""
    param_any_slots = 0
    param_typed_slots = 0
    return_any_slots = 0
    return_typed_slots = 0

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

                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    is_any = type_str.strip().lower() == "any"

                                    if category == "arg":  # Parameter
                                        param_typed_slots += 1
                                        if is_any:
                                            param_any_slots += 1
                                    elif category == "return":  # Return type
                                        return_typed_slots += 1
                                        if is_any:
                                            return_any_slots += 1

    param_any_rate = param_any_slots / param_typed_slots if param_typed_slots > 0 else 0
    return_any_rate = (
        return_any_slots / return_typed_slots if return_typed_slots > 0 else 0
    )

    return (
        param_any_slots,
        param_typed_slots,
        param_any_rate,
        return_any_slots,
        return_typed_slots,
        return_any_rate,
    )


def main():
    # Define model files (using relative paths) - ordered by preference
    model_files = {
        "o3-mini": {
            "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        },
        "o3-mini-renamed": {
            "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_renamed_output_benchmarks.json",
        },
        "deepseek": {
            "type_info_path": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        },
        "deepseek-renamed": {
            "type_info_path": "../Type_info_LLMS/Type_info_deepseek_renamed_output_2_benchmarks.json",
        },
        "claude3-sonnet": {
            "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
        },
        
        
        "claude3-sonnet-renamed": {
            "type_info_path": "../Type_info_LLMS/Type_info_claude_sonnet_renamed_output_benchmarks.json",
        },
        "Human": {
            "type_info_path": "../Type_info_LLMS/Type_info_original_files.json",
        },
        "gpt35": {
            "type_info_path": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
        },
        "gpt35-renamed": {
            "type_info_path": "../Type_info_LLMS/Type_info_gpt35_renamed_output_benchmarks.json",
        },
    }

    print("=" * 80)
    print("ANY-RATE ANALYSIS: Any_rate = #Any_slots / #typed_slots")
    print("=" * 80)

    # Get base filenames to filter by
    base_filenames = get_base_filenames()
    if not base_filenames:
        print("No base files found. Exiting.")
        return

    results = {}

    for model_name, config in model_files.items():
        print(f"\nProcessing {model_name}...")

        # Load data
        type_info_data = load_type_info(config["type_info_path"])
        if not type_info_data:
            print(f"  Failed to load {config['type_info_path']}")
            continue

        # Filter to only include files that exist in base directory
        filtered_type_info = {k: v for k, v in type_info_data.items() if k in base_filenames}
        print(f"  Filtered to {len(filtered_type_info)} files from {len(type_info_data)} total files")

        # Calculate overall Any-rate
        any_slots, typed_slots, any_rate = calculate_any_rate(filtered_type_info)

        # Calculate category-specific Any-rates
        param_any, param_typed, param_rate, return_any, return_typed, return_rate = (
            calculate_any_rate_by_category(filtered_type_info)
        )

        results[model_name] = {
            "any_slots": any_slots,
            "typed_slots": typed_slots,
            "any_rate": any_rate,
            "param_any": param_any,
            "param_typed": param_typed,
            "param_rate": param_rate,
            "return_any": return_any,
            "return_typed": return_typed,
            "return_rate": return_rate,
        }

        print(
            f"  Overall: {any_slots:,} Any / {typed_slots:,} typed = {any_rate:.3f} ({any_rate*100:.1f}%)"
        )
        print(
            f"  Parameters: {param_any:,} Any / {param_typed:,} typed = {param_rate:.3f} ({param_rate*100:.1f}%)"
        )
        print(
            f"  Returns: {return_any:,} Any / {return_typed:,} typed = {return_rate:.3f} ({return_rate*100:.1f}%)"
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
    output_file = "./any_rate_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "Model",
                "Any_slots",
                "Typed_slots",
                "Any_rate",
                "Param_any",
                "Param_typed",
                "Param_rate",
                "Return_any",
                "Return_typed",
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
                    data["typed_slots"],
                    f"{data['any_rate']*100:.2f}%",
                    data["param_any"],
                    data["param_typed"],
                    f"{data['param_rate']*100:.2f}%",
                    data["return_any"],
                    data["return_typed"],
                    f"{data['return_rate']*100:.2f}%",
                    f"{delta*100:+.2f}%" if model_name != "Human" else "—",
                ]
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
