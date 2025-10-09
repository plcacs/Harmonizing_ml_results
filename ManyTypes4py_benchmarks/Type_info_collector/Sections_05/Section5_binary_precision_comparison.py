import json
import csv
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration for 6 LLMs
LLM_CONFIGS = {
    "gpt-3.5": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
    },
    "gpt-4o": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
    },
    "o1-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o1_mini_benchmarks.json",
    },
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
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
DETAILED_CSV = f"{OUTPUT_DIR}/llm_binary_precision_detailed.csv"


def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def is_any_type(type_str: str) -> bool:
    """Check if a type is considered 'Any' (including empty types).
    Returns True if type is Any/empty, False if it's a specific type."""
    if not isinstance(type_str, str) or not type_str.strip():
        return True  # Empty or None types are treated as 'Any'

    type_str = type_str.strip().lower()
    return type_str == "any"


def get_binary_precision_score(type_str: str) -> int:
    """Calculate binary precision score for a type annotation.
    Returns 0 for Any/empty types, 1 for specific types."""
    return 0 if is_any_type(type_str) else 1


def analyze_file_binary_precision(type_info: Dict) -> Dict:
    """Analyze binary precision metrics for a single file.
    Includes all parameters (empty types treated as 'Any' = 0 points)."""
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
            param_name = entry.get("name", "")
            tlist = entry.get("type", [])

            # Skip 'self' parameter completely
            if param_name == "self":
                continue

            # Count all parameters (including those without type annotations)
            total_slots += 1

            # Get type string (empty if no type annotation)
            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                type_str = t0 if isinstance(t0, str) else ""
            else:
                type_str = ""  # No type annotation

            precision_score = get_binary_precision_score(type_str)
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
        fname for fname, info in untyped_mypy.items() if info.get("isCompiled") is True
    }


def main():
    """Main analysis function comparing 6 LLMs based on Binary Precision Score (Any vs Non-Any)."""
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

            file_analysis = analyze_file_binary_precision(type_info[filename])

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

    # Write detailed per-file CSV
    print("Writing detailed results...")
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

    # Print summary
    print(f"\nDetailed results written to: {DETAILED_CSV}")

    print(f"\nLLM Binary Precision Ranking (Any vs Non-Any):")
    print("-" * 80)
    for rank, (llm_name, results) in enumerate(ranked_llms, 1):
        non_any_rate = results["avg_precision_score"] * 100

        description = ""

        print(
            f"{rank}. {llm_name:<20} Non-Any Rate: {non_any_rate:.1f}% "
            f"(Files: {results['total_files']}, Slots: {results['total_slots']}){description}"
        )


if __name__ == "__main__":
    main()
