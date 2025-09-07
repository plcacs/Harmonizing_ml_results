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


def normalize_type(type_str):
    """Normalize type string for analysis."""
    if not type_str or not isinstance(type_str, str):
        return ""

    type_str = type_str.strip().lower()

    # Remove quotes/backticks
    if (
        (type_str.startswith("'") and type_str.endswith("'"))
        or (type_str.startswith('"') and type_str.endswith('"'))
        or (type_str.startswith("`") and type_str.endswith("`"))
    ):
        type_str = type_str[1:-1]

    type_str = type_str.replace("'", "").replace('"', "").replace("`", "")

    # Handle common type aliases
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

    # Strip module prefixes
    for prefix in [
        "typing.",
        "tp.",
        "pd.",
        "pandas.",
        "builtins.",
        "collections.",
    ]:
        type_str = type_str.replace(prefix, "")

    # Remove spaces
    type_str = type_str.replace(" ", "")

    # Handle Union with None -> Optional
    if "|" in type_str:
        parts = [p for p in type_str.split("|") if p]
        if "none" in parts and len(parts) == 2:
            other = parts[0] if parts[1] == "none" else parts[1]
            type_str = f"optional[{other}]"

    if type_str.startswith("union[") and type_str.endswith("]"):
        inner = type_str[len("union[") : -1]
        union_parts = [p for p in inner.split(",") if p]
        if "none" in union_parts and len(union_parts) == 2:
            other = union_parts[0] if union_parts[1] == "none" else union_parts[1]
            type_str = f"optional[{other}]"

    return type_str


def is_any_type(type_str):
    """Check if type is Any or equivalent."""
    normalized = normalize_type(type_str)
    return normalized == "any"


def is_concrete_type(type_str):
    """Check if type is concrete (not Any, None, or empty)."""
    if not type_str or not isinstance(type_str, str):
        return False

    normalized = normalize_type(type_str)
    return normalized not in ["any", "none", ""]


def analyze_single_dataset(data, model_name):
    """Analyze type patterns in a single dataset."""

    results = {
        "model": model_name,
        "total_parameters": 0,
        "any_count": 0,
        "concrete_count": 0,
        "empty_count": 0,
        "total_functions": 0,
        "total_files": len(data),
    }

    detailed_results = []

    print(f"Analyzing {model_name} with {len(data)} files...")

    for filename, functions in data.items():
        results["total_functions"] += len(functions)

        for func_key, params in functions.items():
            for param in params:
                type_str = param.get("type", [""])[0] if param.get("type") else ""

                results["total_parameters"] += 1

                # Analyze type categories
                if is_any_type(type_str):
                    results["any_count"] += 1
                elif is_concrete_type(type_str):
                    results["concrete_count"] += 1
                else:
                    results["empty_count"] += 1

                # Store detailed result
                detailed_results.append(
                    {
                        "filename": filename,
                        "function": func_key,
                        "parameter": (param.get("category", ""), param.get("name", "")),
                        "type": type_str,
                        "normalized_type": normalize_type(type_str),
                        "is_any": is_any_type(type_str),
                        "is_concrete": is_concrete_type(type_str),
                    }
                )

    return results, detailed_results


def calculate_percentages(results):
    """Calculate percentages for the results."""
    total = results["total_parameters"]

    if total == 0:
        results["any_pct"] = 0.0
        results["concrete_pct"] = 0.0
        results["empty_pct"] = 0.0
        return results

    # Calculate percentages
    results["any_pct"] = (results["any_count"] / total) * 100
    results["concrete_pct"] = (results["concrete_count"] / total) * 100
    results["empty_pct"] = (results["empty_count"] / total) * 100

    return results


def main():
    """Main analysis function."""

    # Define model pairs for comparison
    model_pairs = {
        "o3-mini": {
            "original": "Type_info_collector/Type_info_o3_mini_output_benchmarks.json",
            "large": "Type_info_collector/Type_info_o3_mini_output_large_benchmarks.json",
        },
        "deepseek": {
            "original": "Type_info_collector/Type_info_deepseek_output_benchmarks.json",
            "large": "Type_info_collector/Type_info_deepseek_output_large_benchmarks.json",
        },
    }

    print("=" * 80)
    print("LLM VERSION COMPARISON ANALYSIS")
    print("=" * 80)

    all_results = []

    for model_name, file_paths in model_pairs.items():
        print(f"\nProcessing {model_name}...")

        # Load data
        original_data = load_type_info(file_paths["original"])
        large_data = load_type_info(file_paths["large"])

        if not original_data:
            print(f"  Failed to load original: {file_paths['original']}")
            continue
        if not large_data:
            print(f"  Failed to load large: {file_paths['large']}")
            continue

        # Analyze original dataset
        print(f"\n  Analyzing {model_name} original version...")
        original_results, _ = analyze_single_dataset(
            original_data, f"{model_name}_original"
        )
        original_results = calculate_percentages(original_results)
        all_results.append(original_results)

        # Analyze large dataset
        print(f"\n  Analyzing {model_name} large version...")
        large_results, _ = analyze_single_dataset(large_data, f"{model_name}_large")
        large_results = calculate_percentages(large_results)
        all_results.append(large_results)

        # Print summary
        print(f"\n  {model_name} Original:")
        print(f"    Total parameters: {original_results['total_parameters']:,}")
        print(
            f"    Any usage: {original_results['any_pct']:.1f}% ({original_results['any_count']:,})"
        )
        print(
            f"    Concrete types: {original_results['concrete_pct']:.1f}% ({original_results['concrete_count']:,})"
        )
        print(
            f"    Empty/None: {original_results['empty_pct']:.1f}% ({original_results['empty_count']:,})"
        )

        print(f"\n  {model_name} Large:")
        print(f"    Total parameters: {large_results['total_parameters']:,}")
        print(
            f"    Any usage: {large_results['any_pct']:.1f}% ({large_results['any_count']:,})"
        )
        print(
            f"    Concrete types: {large_results['concrete_pct']:.1f}% ({large_results['concrete_count']:,})"
        )
        print(
            f"    Empty/None: {large_results['empty_pct']:.1f}% ({large_results['empty_count']:,})"
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("ANY USAGE COMPARISON: Original vs Large")
    print("=" * 80)
    print(
        f"{'Model':<12} {'Original_Any_%':<15} {'Large_Any_%':<15} {'Difference':<12}"
    )
    print("-" * 60)

    # Group results by model
    model_groups = {}
    for result in all_results:
        model_base = result["model"].replace("_original", "").replace("_large", "")
        if model_base not in model_groups:
            model_groups[model_base] = {}

        if "_original" in result["model"]:
            model_groups[model_base]["original"] = result
        elif "_large" in result["model"]:
            model_groups[model_base]["large"] = result

    for model_name, versions in model_groups.items():
        if "original" in versions and "large" in versions:
            orig_any = versions["original"]["any_pct"]
            large_any = versions["large"]["any_pct"]
            difference = large_any - orig_any

            print(
                f"{model_name:<12} {orig_any:<15.1f}% {large_any:<15.1f}% {difference:+.1f}%"
            )
        else:
            print(
                f"{model_name:<12} {'Missing data':<15} {'Missing data':<15} {'N/A':<12}"
            )


if __name__ == "__main__":
    main()
