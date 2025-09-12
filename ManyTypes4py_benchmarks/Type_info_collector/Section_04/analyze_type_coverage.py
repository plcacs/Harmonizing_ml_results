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


def is_typed_slot(param):
    """Check if a parameter has a type annotation."""
    if not isinstance(param, dict):
        return False

    param_types = param.get("type", [])
    if isinstance(param_types, list) and len(param_types) > 0:
        type_str = param_types[0]
        return isinstance(type_str, str) and type_str.strip() != ""
    return False


def analyze_type_coverage(human_data, llm_data):
    """Analyze type slot coverage between human and LLM annotations."""
    both_typed = 0
    both_untyped = 0
    only_human_typed = 0
    only_llm_typed = 0
    total_slots = 0

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

                human_typed = is_typed_slot(human_param)
                llm_typed = is_typed_slot(llm_param)

                total_slots += 1

                if human_typed and llm_typed:
                    both_typed += 1
                elif not human_typed and not llm_typed:
                    both_untyped += 1
                elif human_typed and not llm_typed:
                    only_human_typed += 1
                elif not human_typed and llm_typed:
                    only_llm_typed += 1

    return {
        "both_typed": both_typed,
        "both_untyped": both_untyped,
        "only_human_typed": only_human_typed,
        "only_llm_typed": only_llm_typed,
        "total_slots": total_slots,
    }


def analyze_type_coverage_by_category(human_data, llm_data):
    """Analyze type slot coverage separately for parameters and return types."""
    categories = {
        "parameters": {
            "both_typed": 0,
            "both_untyped": 0,
            "only_human": 0,
            "only_llm": 0,
            "total": 0,
        },
        "returns": {
            "both_typed": 0,
            "both_untyped": 0,
            "only_human": 0,
            "only_llm": 0,
            "total": 0,
        },
    }

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

                human_typed = is_typed_slot(human_param)
                llm_typed = is_typed_slot(llm_param)
                category = human_param.get("category", "")

                if category == "arg":  # Parameter
                    cat_key = "parameters"
                elif category == "return":  # Return type
                    cat_key = "returns"
                else:
                    continue

                categories[cat_key]["total"] += 1

                if human_typed and llm_typed:
                    categories[cat_key]["both_typed"] += 1
                elif not human_typed and not llm_typed:
                    categories[cat_key]["both_untyped"] += 1
                elif human_typed and not llm_typed:
                    categories[cat_key]["only_human"] += 1
                elif not human_typed and llm_typed:
                    categories[cat_key]["only_llm"] += 1

    return categories


def main():
    # Load human data
    human_data = load_type_info("../Type_info_LLMS/Type_info_original_files.json")
    if not human_data:
        print("Failed to load human data")
        return

    # Define LLM files
    llm_files = {
        "GPT4o": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
        "O1-mini": "../Type_info_LLMS/Type_info_o1_mini_benchmarks.json",
        "O3-mini": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "DeepSeek": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        "Claude3-Sonnet": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
        "DeepSeek-User-Annotated": "../Type_info_LLMS/Type_info_deepseek_user_annotated_benchmarks.json",
        "O3-Mini-User-Annotated": "../Type_info_LLMS/Type_info_o3_mini_user_annotated_benchmarks.json",
        "Claude3-Sonnet-User-Annotated": "../Type_info_LLMS/Type_info_claude3_sonnet_user_annotated_benchmarks.json",
    }

    print("=" * 80)
    print("TYPE SLOT COVERAGE ANALYSIS: Human vs LLM")
    print("=" * 80)

    results = []

    for model_name, filename in llm_files.items():
        print(f"\nProcessing {model_name}...")

        llm_data = load_type_info(filename)
        if not llm_data:
            print(f"  Failed to load {filename}")
            continue

        # Overall analysis
        coverage = analyze_type_coverage(human_data, llm_data)

        # Category-specific analysis
        category_coverage = analyze_type_coverage_by_category(human_data, llm_data)

        results.append(
            {
                "Model": model_name,
                "Both_Typed": coverage["both_typed"],
                "Both_Typed_%": coverage["both_typed"] / coverage["total_slots"] * 100,
                "Both_Untyped": coverage["both_untyped"],
                "Both_Untyped_%": coverage["both_untyped"]
                / coverage["total_slots"]
                * 100,
                "Only_Human_Typed": coverage["only_human_typed"],
                "Only_Human_Typed_%": coverage["only_human_typed"]
                / coverage["total_slots"]
                * 100,
                "Only_LLM_Typed": coverage["only_llm_typed"],
                "Only_LLM_Typed_%": coverage["only_llm_typed"]
                / coverage["total_slots"]
                * 100,
                "Total_Slots": coverage["total_slots"],
            }
        )

        print(f"  Total slots: {coverage['total_slots']:,}")
        print(
            f"  Both typed: {coverage['both_typed']:,} ({coverage['both_typed']/coverage['total_slots']*100:.1f}%)"
        )
        print(
            f"  Both untyped: {coverage['both_untyped']:,} ({coverage['both_untyped']/coverage['total_slots']*100:.1f}%)"
        )
        print(
            f"  Only human typed: {coverage['only_human_typed']:,} ({coverage['only_human_typed']/coverage['total_slots']*100:.1f}%)"
        )
        print(
            f"  Only LLM typed: {coverage['only_llm_typed']:,} ({coverage['only_llm_typed']/coverage['total_slots']*100:.1f}%)"
        )

    # Save to CSV
    output_file = "type_coverage_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Both_Typed",
                "Both_Typed_%",
                "Both_Untyped",
                "Both_Untyped_%",
                "Only_Human_Typed",
                "Only_Human_Typed_%",
                "Only_LLM_Typed",
                "Only_LLM_Typed_%",
                "Total_Slots",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["Model"],
                    result["Both_Typed"],
                    f"{result['Both_Typed_%']:.2f}",
                    result["Both_Untyped"],
                    f"{result['Both_Untyped_%']:.2f}",
                    result["Only_Human_Typed"],
                    f"{result['Only_Human_Typed_%']:.2f}",
                    result["Only_LLM_Typed"],
                    f"{result['Only_LLM_Typed_%']:.2f}",
                    result["Total_Slots"],
                ]
            )

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Both':<8} {'Both':<8} {'Only':<8} {'Only':<8}")
    print(f"{'':<15} {'Typed':<8} {'Untyped':<8} {'Human':<8} {'LLM':<8}")
    print("-" * 80)
    for result in results:
        print(
            f"{result['Model']:<15} {result['Both_Typed']:<8,} {result['Both_Untyped']:<8,} {result['Only_Human_Typed']:<8,} {result['Only_LLM_Typed']:<8,}"
        )


if __name__ == "__main__":
    main()
