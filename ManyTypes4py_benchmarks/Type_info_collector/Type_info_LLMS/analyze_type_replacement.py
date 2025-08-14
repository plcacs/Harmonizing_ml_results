import json
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


def is_concrete_type(type_str):
    """Check if type is concrete (not Any, None, or empty)."""
    if not type_str or not isinstance(type_str, str):
        return False
    type_str = type_str.strip().lower()
    return type_str not in ["any", "none", ""]


def analyze_type_replacements(human_data, llm_data):
    """Analyze how LLMs replace human concrete types with Any."""
    replacements = 0
    total_concrete = 0
    preserved = 0

    # Find common files
    common_files = set(human_data.keys()) & set(llm_data.keys())

    for filename in common_files:
        human_functions = human_data[filename]
        llm_functions = llm_data[filename]

        # Find common functions
        common_funcs = set(human_functions.keys()) & set(llm_functions.keys())

        for func_name in common_funcs:
            human_params = human_functions[func_name]
            llm_params = llm_functions[func_name]

            # Match parameters by category and name
            human_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in human_params
            }
            llm_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in llm_params
            }

            common_keys = set(human_dict.keys()) & set(llm_dict.keys())

            for key in common_keys:
                human_param = human_dict[key]
                llm_param = llm_dict[key]

                human_type = (
                    human_param.get("type", [""])[0] if human_param.get("type") else ""
                )
                llm_type = (
                    llm_param.get("type", [""])[0] if llm_param.get("type") else ""
                )

                # Check if human had concrete type
                if is_concrete_type(human_type):
                    total_concrete += 1

                    # Check if LLM replaced with Any
                    if llm_type.strip().lower() == "any":
                        replacements += 1
                    else:
                        preserved += 1

    replacement_rate = replacements / total_concrete if total_concrete > 0 else 0
    return replacements, total_concrete, replacement_rate


def main():
    # Load human data
    human_data = load_type_info("./Type_info_original_files.json")
    if not human_data:
        print("Failed to load human data")
        return

    # Define LLM files
    llm_files = {
        "GPT4o": "./Type_info_gpt4o_benchmarks.json",
        "O1-mini": "./Type_info_o1_mini_benchmarks.json",
        "O3-mini": "./Type_info_o3_mini_1st_run_benchmarks.json",
        "DeepSeek": "./Type_info_deep_seek_benchmarks.json",
        "Claude3-Sonnet": "./Type_info_claude3_sonnet_1st_run_benchmarks.json",
    }

    print("=" * 60)
    print("TYPE REPLACEMENT ANALYSIS: Human Concrete → LLM Any")
    print("=" * 60)

    results = []

    for model_name, filename in llm_files.items():
        print(f"\nProcessing {model_name}...")

        llm_data = load_type_info(filename)
        if not llm_data:
            print(f"  Failed to load {filename}")
            continue

        replacements, total_concrete, replacement_rate = analyze_type_replacements(
            human_data, llm_data
        )

        results.append(
            {
                "Model": model_name,
                "Replacements": replacements,
                "Total_Concrete": total_concrete,
                "Replacement_Rate": replacement_rate,
                "Preserved": total_concrete - replacements,
            }
        )

        print(
            f"  Replacements: {replacements:,} / {total_concrete:,} = {replacement_rate:.3f} ({replacement_rate*100:.1f}%)"
        )

    # Save to CSV
    output_file = "type_replacement_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Replacements", "Total_Concrete", "Replacement_Rate", "Preserved"]
        )

        for result in results:
            writer.writerow(
                [
                    result["Model"],
                    result["Replacements"],
                    result["Total_Concrete"],
                    f"{result['Replacement_Rate']:.6f}",
                    result["Preserved"],
                ]
            )

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Replacements':<12} {'Rate':<8}")
    print("-" * 40)
    for result in results:
        print(
            f"{result['Model']:<15} {result['Replacements']:<12,} {result['Replacement_Rate']:<8.3f}"
        )


if __name__ == "__main__":
    main()
