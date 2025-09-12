import json
import csv
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration for 6 LLMs
LLM_CONFIGS = {
    "gpt-3.5": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
    },
    "gpt-4o": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt4o_1st_run_benchmarks.json",
    },
    "o1-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
    },
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deepseek_1st_run_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
    },
}

# Baseline for comparison
UNTYPED_MYPY_PATH = (
    "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
)

# Output files
OUTPUT_DIR = "./precision_results"
OUTPUT_CSV = f"{OUTPUT_DIR}/llm_precision_comparison.csv"
DETAILED_CSV = f"{OUTPUT_DIR}/llm_precision_detailed.csv"
RANKING_CSV = f"{OUTPUT_DIR}/llm_precision_ranking.csv"


def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def get_type_precision_score(type_str: str) -> int:
    """Calculate precision score for a type annotation.
    Higher score = more precise type. Empty types are treated as 'Any' (score 0)."""
    if not isinstance(type_str, str) or not type_str.strip():
        return 0  # Empty or None types are treated as 'Any'

    type_str = type_str.strip().lower()

    # Base precision scores
    if type_str == "any":
        return 0
    elif type_str in ["object", "typing.any"]:
        return 1
    elif type_str in ["str", "int", "float", "bool", "bytes", "complex"]:
        return 10
    elif type_str in ["list", "dict", "set", "tuple"]:
        return 5
    elif type_str in ["none", "nonetype"]:
        return 1

    # Handle generic types with type parameters
    if type_str.startswith("list["):
        inner_type = type_str[5:-1]
        inner_score = get_type_precision_score(inner_type)
        return 10 + inner_score

    elif type_str.startswith("dict["):
        if "," in type_str:
            key_type = type_str[5 : type_str.find(",")]
            value_type = type_str[type_str.find(",") + 1 : -1]
            key_score = get_type_precision_score(key_type)
            value_score = get_type_precision_score(value_type)
            return 10 + key_score + value_score
        else:
            return 5

    elif type_str.startswith("set["):
        inner_type = type_str[4:-1]
        inner_score = get_type_precision_score(inner_type)
        return 10 + inner_score

    elif type_str.startswith("tuple["):
        if "," in type_str:
            inner_types = type_str[6:-1].split(",")
            total_score = 10
            for inner_type in inner_types:
                total_score += get_type_precision_score(inner_type.strip())
            return total_score
        else:
            inner_type = type_str[6:-1]
            return 10 + get_type_precision_score(inner_type)

    elif type_str.startswith("union["):
        if "," in type_str:
            inner_types = type_str[6:-1].split(",")
            total_score = 2
            for inner_type in inner_types:
                total_score += get_type_precision_score(inner_type.strip())
            return total_score
        else:
            inner_type = type_str[6:-1]
            return 2 + get_type_precision_score(inner_type)

    elif type_str.startswith("optional["):
        inner_type = type_str[9:-1]
        return 8 + get_type_precision_score(inner_type)

    elif "typing." in type_str:
        return 8
    
    # Handle custom class types (e.g., "ClassA", "MyClass", "User")
    # Custom classes are more specific than generic types but less than built-in primitives
    elif type_str.replace("_", "").replace(".", "").isalnum() and not type_str.startswith(("list", "dict", "set", "tuple", "union", "optional")):
        # Check if it looks like a class name (starts with capital letter or contains dots for modules)
        if type_str[0].isupper() or "." in type_str:
            return 15  # Custom classes get high precision score
        else:
            return 8   # Other identifiers get medium score
    
    return 5  # Default fallback


def analyze_file_precision(type_info: Dict) -> Dict:
    """Analyze precision metrics for a single file. Includes all parameters (empty types treated as 'Any')."""
    total_slots = 0
    total_precision_score = 0
    total_functions = 0

    if not isinstance(type_info, dict):
        return {
            "total_slots": 0,
            "total_precision_score": 0,
            "total_functions": 0,
            "avg_precision_score": 0,
        }

    for func_name, items in type_info.items():
        if not isinstance(items, list):
            continue

        total_functions += 1
        arg_count = 0

        for entry in items:
            if not isinstance(entry, dict):
                continue

            category = entry.get("category", "")
            tlist = entry.get("type", [])

            if category == "arg":
                arg_count += 1
                # Skip 'self' parameter
                if arg_count == 1:
                    continue

            # Count all parameters (including those without type annotations)
            total_slots += 1
            
            # Get type string (empty if no type annotation)
            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                type_str = t0 if isinstance(t0, str) else ""
            else:
                type_str = ""  # No type annotation
            
            precision_score = get_type_precision_score(type_str)
            total_precision_score += precision_score

    avg_precision_score = total_precision_score / total_slots if total_slots > 0 else 0

    return {
        "total_slots": total_slots,
        "total_precision_score": total_precision_score,
        "total_functions": total_functions,
        "avg_precision_score": avg_precision_score,
    }


def build_baseline_files(untyped_mypy: Dict) -> Set[str]:
    """Build set of files that compile successfully without types."""
    return {
        fname for fname, info in untyped_mypy.items()
    }


def main():
    """Main analysis function comparing 6 LLMs based on Type Precision Score only."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading baseline data...")
    untyped_mypy = load_json(UNTYPED_MYPY_PATH)
    baseline_files = build_baseline_files(untyped_mypy)
    print(f"Found {len(baseline_files)} baseline files")

    # Load all LLM data
    print("Loading LLM data...")
    llm_data = {}
    for llm_name, config in LLM_CONFIGS.items():
        type_info = load_json(config["type_info_path"])
        llm_data[llm_name] = type_info
        print(f"Loaded {llm_name}: {len(type_info)} files")

    # Analyze each LLM and collect per-file data
    llm_results = {}
    file_data = defaultdict(dict)  # filename -> {llm_name: precision_score}

    for llm_name, type_info in llm_data.items():
        print(f"Analyzing {llm_name}...")

        total_files = 0
        total_slots = 0
        total_precision_score = 0
        total_functions = 0

        for filename in baseline_files:
            if filename not in type_info:
                # Assign score of 0 if file is not available for this LLM
                file_data[filename][llm_name] = 0.0
                continue

            file_analysis = analyze_file_precision(type_info[filename])

            if file_analysis["total_slots"] > 0:
                total_files += 1
                total_slots += file_analysis["total_slots"]
                total_precision_score += file_analysis["total_precision_score"]
                total_functions += file_analysis["total_functions"]

                # Store per-file precision score for this LLM
                file_data[filename][llm_name] = file_analysis["avg_precision_score"]
            else:
                # Assign score of 0 if file has no valid type slots
                file_data[filename][llm_name] = 0.0

        # Calculate overall metrics
        avg_precision_score = (
            total_precision_score / total_slots if total_slots > 0 else 0
        )

        llm_results[llm_name] = {
            "total_files": total_files,
            "total_slots": total_slots,
            "total_precision_score": total_precision_score,
            "total_functions": total_functions,
            "avg_precision_score": avg_precision_score,
        }

    # Sort LLMs by average precision score
    ranked_llms = sorted(
        llm_results.items(), key=lambda x: x[1]["avg_precision_score"], reverse=True
    )

    # Write results to CSV files
    print("Writing results...")

    # Main comparison CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "LLM",
                "Total Files",
                "Total Slots",
                "Total Precision Score",
                "Avg Precision Score",
            ]
        )

        for llm_name, results in ranked_llms:
            writer.writerow(
                [
                    llm_name,
                    results["total_files"],
                    results["total_slots"],
                    results["total_precision_score"],
                    f"{results['avg_precision_score']:.3f}",
                ]
            )

    # Detailed per-file CSV
    with open(DETAILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "winners",
                "points_each",
                "scores...",
            ]
        )

        for filename, llm_scores in file_data.items():
            if not llm_scores:
                continue

            # Find the LLM(s) with the highest precision score for this file
            max_score = max(llm_scores.values())
            winners = [llm for llm, score in llm_scores.items() if score == max_score]
            split = 1.0 / len(winners)

            # Create scores string
            scores_str = ";".join(
                [f"{llm}:{score:.3f}" for llm, score in sorted(llm_scores.items())]
            )

            writer.writerow([filename, ";".join(winners), f"{split:.3f}", scores_str])

    # Ranking CSV
    with open(RANKING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Rank",
                "LLM",
                "Avg Precision Score",
                "Total Files",
                "Total Slots",
            ]
        )

        for rank, (llm_name, results) in enumerate(ranked_llms, 1):
            writer.writerow(
                [
                    rank,
                    llm_name,
                    f"{results['avg_precision_score']:.3f}",
                    results["total_files"],
                    results["total_slots"],
                ]
            )

    # Print summary
    print(f"\nResults written to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {DETAILED_CSV}")
    print(f"  {RANKING_CSV}")

    print(f"\nLLM Precision Ranking (by Type Precision Score):")
    print("-" * 60)
    for rank, (llm_name, results) in enumerate(ranked_llms, 1):
        print(
            f"{rank}. {llm_name:<15} Avg Precision Score: {results['avg_precision_score']:.3f} "
            f"(Files: {results['total_files']}, Slots: {results['total_slots']})"
        )


if __name__ == "__main__":
    main()
