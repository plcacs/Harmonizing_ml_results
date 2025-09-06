import json
import csv
import os
from pathlib import Path
from collections import defaultdict


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_function_mappings(mappings_file):
    """Load function signature mappings from JSON file."""
    try:
        with open(mappings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading function mappings: {e}")
        return None


def normalize_type(type_str):
    """Normalize type string for semantic comparison."""
    if not type_str or not isinstance(type_str, str):
        return ""

    type_str = type_str.strip().lower()

    # Handle common type aliases and variations
    type_mappings = {
        "tp.any": "any",
        "typing.any": "any",
        "tp.optional": "optional",
        "typing.optional": "optional",
        "tp.list": "list",
        "typing.list": "list",
        "tp.dict": "dict",
        "typing.dict": "dict",
        "tp.union": "union",
        "typing.union": "union",
        "tp.tuple": "tuple",
        "typing.tuple": "tuple",
        "tp.set": "set",
        "typing.set": "set",
        "tp.sequence": "sequence",
        "typing.sequence": "sequence",
        "tp.iterable": "iterable",
        "typing.iterable": "iterable",
        "tp.callable": "callable",
        "typing.callable": "callable",
    }

    for old, new in type_mappings.items():
        type_str = type_str.replace(old, new)

    return type_str


def types_semantically_match(type1, type2):
    """Check if two types are semantically equivalent."""
    norm1 = normalize_type(type1)
    norm2 = normalize_type(type2)

    # Exact match
    if norm1 == norm2:
        return True

    # Handle empty/None cases
    if not norm1 and not norm2:
        return True

    # Handle Any vs concrete types (Any is considered to match everything)
    if norm1 == "any" or norm2 == "any":
        return True

    # Handle None vs empty
    if (norm1 == "none" and not norm2) or (norm2 == "none" and not norm1):
        return True

    # Handle generic types with same base
    if "[" in norm1 and "[" in norm2:
        base1 = norm1.split("[")[0]
        base2 = norm2.split("[")[0]
        if base1 == base2:
            return True

    return False


def analyze_semantic_matching(
    human_data, original_data, renamed_data, function_mappings, base_files
):
    """Analyze semantic type matching between human, original, and renamed annotations."""

    results = {
        "match_both": 0,  # Match human in both original and renamed
        "match_original_only": 0,  # Match human only in original
        "match_renamed_only": 0,  # Match human only in renamed
        "match_neither": 0,  # Don't match human in either
        "total_comparisons": 0,
    }

    detailed_results = []

    print(f"Analyzing {len(base_files)} base files...")

    for filename in base_files:
        if filename not in human_data:
            print(f"  Skipping {filename} - not found in human data")
            continue

        if filename not in original_data:
            print(f"  Skipping {filename} - not found in original data")
            continue

        if filename not in renamed_data:
            print(f"  Skipping {filename} - not found in renamed data")
            continue

        # Get function mappings for this file
        file_mappings = function_mappings.get(filename, [])
        if not file_mappings:
            print(f"  Skipping {filename} - no function mappings found")
            continue

        print(f"  Processing {filename} with {len(file_mappings)} function mappings")

        human_functions = human_data[filename]
        original_functions = original_data[filename]
        renamed_functions = renamed_data[filename]

        for mapping in file_mappings:
            original_func_key = mapping["original_func"]
            renamed_func_key = mapping["renamed_func"]

            # Check if all three versions exist
            if original_func_key not in human_functions:
                continue
            if original_func_key not in original_functions:
                continue
            if renamed_func_key not in renamed_functions:
                continue

            human_params = human_functions[original_func_key]
            original_params = original_functions[original_func_key]
            renamed_params = renamed_functions[renamed_func_key]

            # Create parameter dictionaries for easier matching
            human_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in human_params
            }
            original_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in original_params
            }
            renamed_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in renamed_params
            }

            # Find common parameters
            common_keys = (
                set(human_dict.keys())
                & set(original_dict.keys())
                & set(renamed_dict.keys())
            )

            for key in common_keys:
                human_param = human_dict[key]
                original_param = original_dict[key]
                renamed_param = renamed_dict[key]

                human_type = (
                    human_param.get("type", [""])[0] if human_param.get("type") else ""
                )
                original_type = (
                    original_param.get("type", [""])[0]
                    if original_param.get("type")
                    else ""
                )
                renamed_type = (
                    renamed_param.get("type", [""])[0]
                    if renamed_param.get("type")
                    else ""
                )

                # Skip if human type is empty/None (no ground truth)
                if not human_type or human_type.strip().lower() in ["", "none"]:
                    continue

                results["total_comparisons"] += 1

                # Check semantic matches
                original_matches = types_semantically_match(human_type, original_type)
                renamed_matches = types_semantically_match(human_type, renamed_type)

                # Categorize the result
                if original_matches and renamed_matches:
                    results["match_both"] += 1
                    category = "match_both"
                elif original_matches and not renamed_matches:
                    results["match_original_only"] += 1
                    category = "match_original_only"
                elif not original_matches and renamed_matches:
                    results["match_renamed_only"] += 1
                    category = "match_renamed_only"
                else:
                    results["match_neither"] += 1
                    category = "match_neither"

                # Store detailed result for analysis
                detailed_results.append(
                    {
                        "filename": filename,
                        "function": original_func_key,
                        "parameter": key,
                        "human_type": human_type,
                        "original_type": original_type,
                        "renamed_type": renamed_type,
                        "category": category,
                        "original_matches": original_matches,
                        "renamed_matches": renamed_matches,
                    }
                )

    return results, detailed_results


def main():
    # Load all required data
    print("Loading data files...")

    # Load human annotations (ground truth)
    human_data = load_type_info("../Type_info_LLMS/Type_info_original_files.json")
    if not human_data:
        print("Failed to load human type annotations")
        return

    # Load function mappings
    function_mappings = load_function_mappings("function_signature_mappings.json")
    if not function_mappings:
        print("Failed to load function mappings")
        return

    # Get base files from renamed benchmarks directory
    renamed_dir = "../../Hundrad_renamed_benchmarks"
    base_files = [f.name for f in Path(renamed_dir).glob("*.py")]
    print(f"Found {len(base_files)} base files in {renamed_dir}")

    # Define model pairs for comparison
    model_pairs = {
        "deepseek": {
            "original": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_deepseek_renamed_output_2_benchmarks.json",
        },
        "o3-mini": {
            "original": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_o3_mini_renamed_output_benchmarks.json",
        },
        "claude3-sonnet": {
            "original": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_claude_sonnet_renamed_output_benchmarks.json",
        },
        "gpt35": {
            "original": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_gpt35_renamed_output_benchmarks.json",
        },
    }

    print("=" * 80)
    print("SEMANTIC TYPE MATCHING ANALYSIS: Human vs Original vs Renamed")
    print("=" * 80)

    all_results = []
    all_detailed_results = []

    for model_name, file_paths in model_pairs.items():
        print(f"\nProcessing {model_name}...")

        original_data = load_type_info(file_paths["original"])
        renamed_data = load_type_info(file_paths["renamed"])

        if not original_data:
            print(f"  Failed to load original: {file_paths['original']}")
            continue
        if not renamed_data:
            print(f"  Failed to load renamed: {file_paths['renamed']}")
            continue

        results, detailed_results = analyze_semantic_matching(
            human_data, original_data, renamed_data, function_mappings, base_files
        )

        # Calculate percentages
        total = results["total_comparisons"]
        if total > 0:
            results["match_both_pct"] = (results["match_both"] / total) * 100
            results["match_original_only_pct"] = (
                results["match_original_only"] / total
            ) * 100
            results["match_renamed_only_pct"] = (
                results["match_renamed_only"] / total
            ) * 100
            results["match_neither_pct"] = (results["match_neither"] / total) * 100

        results["model"] = model_name
        all_results.append(results)

        # Add model info to detailed results
        for detail in detailed_results:
            detail["model"] = model_name
        all_detailed_results.extend(detailed_results)

        print(f"  Total comparisons: {total:,}")
        print(
            f"  Match both: {results['match_both']:,} ({results.get('match_both_pct', 0):.1f}%)"
        )
        print(
            f"  Match original only: {results['match_original_only']:,} ({results.get('match_original_only_pct', 0):.1f}%)"
        )
        print(
            f"  Match renamed only: {results['match_renamed_only']:,} ({results.get('match_renamed_only_pct', 0):.1f}%)"
        )
        print(
            f"  Match neither: {results['match_neither']:,} ({results.get('match_neither_pct', 0):.1f}%)"
        )

    # Save summary results to CSV
    summary_file = "semantic_type_matching_summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Total_Comparisons",
                "Match_Both",
                "Match_Both_%",
                "Match_Original_Only",
                "Match_Original_Only_%",
                "Match_Renamed_Only",
                "Match_Renamed_Only_%",
                "Match_Neither",
                "Match_Neither_%",
            ]
        )

        for result in all_results:
            writer.writerow(
                [
                    result["model"],
                    result["total_comparisons"],
                    result["match_both"],
                    f"{result.get('match_both_pct', 0):.2f}",
                    result["match_original_only"],
                    f"{result.get('match_original_only_pct', 0):.2f}",
                    result["match_renamed_only"],
                    f"{result.get('match_renamed_only_pct', 0):.2f}",
                    result["match_neither"],
                    f"{result.get('match_neither_pct', 0):.2f}",
                ]
            )

    # Save detailed results to CSV
    detailed_file = "semantic_type_matching_detailed.csv"
    with open(detailed_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Filename",
                "Function",
                "Parameter",
                "Human_Type",
                "Original_Type",
                "Renamed_Type",
                "Category",
                "Original_Matches",
                "Renamed_Matches",
            ]
        )

        for detail in all_detailed_results:
            writer.writerow(
                [
                    detail["model"],
                    detail["filename"],
                    detail["function"],
                    str(detail["parameter"]),
                    detail["human_type"],
                    detail["original_type"],
                    detail["renamed_type"],
                    detail["category"],
                    detail["original_matches"],
                    detail["renamed_matches"],
                ]
            )

    print(f"\nResults saved to:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed: {detailed_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<15} {'Total':<8} {'Both':<8} {'Orig':<8} {'Ren':<8} {'None':<8}")
    print("-" * 60)
    for result in all_results:
        print(
            f"{result['model']:<15} {result['total_comparisons']:<8,} "
            f"{result.get('match_both_pct', 0):<8.1f}% "
            f"{result.get('match_original_only_pct', 0):<8.1f}% "
            f"{result.get('match_renamed_only_pct', 0):<8.1f}% "
            f"{result.get('match_neither_pct', 0):<8.1f}%"
        )


if __name__ == "__main__":
    main()
