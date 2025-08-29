import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration for 6 LLMs with mypy results
LLM_CONFIGS = {
    "gpt-3.5": {
        "type_info_path": "./Type_info_gpt35_2nd_run_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
    },
    "gpt-4o": {
        "type_info_path": "./Type_info_gpt4o_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_gpt4o.json",
    },
    "o1-mini": {
        "type_info_path": "./Type_info_o1_mini_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
    },
    "o3-mini": {
        "type_info_path": "./Type_info_o3_mini_1st_run_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
    },
    "deepseek": {
        "type_info_path": "./Type_info_deep_seek_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
    },
    "claude3-sonnet": {
        "type_info_path": "./Type_info_claude3_sonnet_1st_run_benchmarks.json",
        "mypy_results_path": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    },
}

# Baseline for comparison
UNTYPED_MYPY_PATH = (
    "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
)

# Output files
OUTPUT_DIR = "./precision_results"
OUTPUT_CSV = f"{OUTPUT_DIR}/llm_precision_comparison_with_plot.csv"
DETAILED_CSV = f"{OUTPUT_DIR}/llm_precision_detailed_with_plot.csv"
RANKING_CSV = f"{OUTPUT_DIR}/llm_precision_ranking_with_plot.csv"
PLOT_PATH = f"{OUTPUT_DIR}/llm_precision_plot_strict_vs_baseline.pdf"


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
    Higher score = more precise type."""
    if not isinstance(type_str, str):
        return 0

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
        return 8

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

    return 5


def analyze_file_precision(type_info: Dict) -> Dict:
    """Analyze precision metrics for a single file."""
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

            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                if isinstance(t0, str) and t0.strip():
                    total_slots += 1
                    precision_score = get_type_precision_score(t0)
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


def build_strict_baseline_files(llm_mypy_results: Dict[str, Dict]) -> Set[str]:
    """Build set of files where ALL LLMs succeeded (mypy success)."""
    if not llm_mypy_results:
        return set()

    # Start with the first LLM's successful files
    strict_baseline = set()
    first_llm = list(llm_mypy_results.keys())[0]

    for fname, info in llm_mypy_results[first_llm].items():
        if info.get("isCompiled") is True:
            strict_baseline.add(fname)

    # Intersect with all other LLMs
    for llm_name, mypy_results in llm_mypy_results.items():
        if llm_name == first_llm:
            continue

        llm_success_files = {
            fname
            for fname, info in mypy_results.items()
            if info.get("isCompiled") is True
        }
        strict_baseline = strict_baseline.intersection(llm_success_files)

    return strict_baseline


def analyze_llm_performance(
    llm_data: Dict, baseline_files: Set[str], llm_name: str
) -> Dict:
    """Analyze performance for a single LLM on given baseline files."""
    type_info = llm_data[llm_name]

    total_files = 0
    total_slots = 0
    total_precision_score = 0
    total_functions = 0
    file_scores = {}

    for filename in baseline_files:
        if filename not in type_info:
            file_scores[filename] = 0.0
            continue

        file_analysis = analyze_file_precision(type_info[filename])

        if file_analysis["total_slots"] > 0:
            total_files += 1
            total_slots += file_analysis["total_slots"]
            total_precision_score += file_analysis["total_precision_score"]
            total_functions += file_analysis["total_functions"]
            file_scores[filename] = file_analysis["avg_precision_score"]
        else:
            file_scores[filename] = 0.0

    avg_precision_score = total_precision_score / total_slots if total_slots > 0 else 0

    return {
        "total_files": total_files,
        "total_slots": total_slots,
        "total_precision_score": total_precision_score,
        "total_functions": total_functions,
        "avg_precision_score": avg_precision_score,
        "file_scores": file_scores,
    }


def group_files_by_precision_winners(
    llm_data: Dict, baseline_files: Set[str], llm_names: List[str]
) -> Dict[int, Set[str]]:
    """Group files by how many LLMs are tied for highest precision score."""
    grouped_files = defaultdict(set)

    for filename in baseline_files:
        # Calculate precision scores for all LLMs on this file
        llm_scores = {}

        for llm_name in llm_names:
            if filename in llm_data[llm_name]:
                file_analysis = analyze_file_precision(llm_data[llm_name][filename])
                llm_scores[llm_name] = file_analysis["avg_precision_score"]
            else:
                llm_scores[llm_name] = 0.0

        # Find winners (LLMs with highest precision score)
        if llm_scores:
            max_score = max(llm_scores.values())
            winners = [llm for llm, score in llm_scores.items() if score == max_score]
            winner_count = len(winners)

            # Group by winner count (1-5, excluding 6 as requested)
            if winner_count <= 5:
                grouped_files[winner_count].add(filename)

    return grouped_files


def create_precision_plot(
    llm_results_baseline: Dict,
    llm_results_strict: Dict,
    grouped_files: Dict[int, Set[str]],
    llm_names: List[str],
    llm_data: Dict,
    llm_mypy_results: Dict,
    strict_baseline_files: Set[str],
):
    """Create the winner count comparison plot with grouped bars."""

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Colors for each LLM (consistent across both evaluation methods)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Group sizes to plot (excluding size 6 as requested)
    group_sizes = [1, 2, 3, 4, 5]

    # Set up bar positions - each group will have 12 bars (6 LLMs × 2 methods)
    x = np.arange(len(group_sizes))
    width = 0.03  # Much smaller width to fit 12 bars per group
    # Create positions for 12 bars: 6 baseline (left) + 6 strict (right)
    bar_positions = np.concatenate(
        [
            np.arange(-0.15, 0.0, 0.025),  # 6 positions for baseline bars
            np.arange(0.025, 0.16, 0.025),  # 6 positions for strict bars
        ]
    )

    # Plot data for each group size
    for i, group_size in enumerate(group_sizes):
        if group_size not in grouped_files:
            continue

        files_in_group = grouped_files[group_size]
        if not files_in_group:
            continue

        # Calculate winner counts for each LLM in this group
        baseline_winner_counts = []
        strict_winner_counts = []

        for llm_name in llm_names:
            # Count files where this LLM is the winner using baseline evaluation
            baseline_winner_count = count_llm_winners_in_group(
                llm_data, files_in_group, llm_names, llm_name
            )
            baseline_winner_counts.append(baseline_winner_count)

            # Count files where this LLM is the winner using strict evaluation
            # First, group strict files by precision winners
            strict_grouped_files = group_files_by_precision_winners(
                llm_data, strict_baseline_files, llm_names
            )

            # Then count winners in the corresponding strict group
            if group_size in strict_grouped_files:
                strict_winner_count = count_llm_winners_in_group(
                    llm_data, strict_grouped_files[group_size], llm_names, llm_name
                )
            else:
                strict_winner_count = 0
            strict_winner_counts.append(strict_winner_count)

        # Plot baseline bars (left side)
        for j, (llm_name, count) in enumerate(zip(llm_names, baseline_winner_counts)):
            ax.bar(
                x[i] + bar_positions[j],
                count,
                width,
                color=colors[j],
                label=f"{llm_name} (Baseline)" if i == 0 else "",
            )

        # Plot strict bars (right side)
        for j, (llm_name, count) in enumerate(zip(llm_names, strict_winner_counts)):
            ax.bar(
                x[i] + bar_positions[j + len(llm_names)],
                count,
                width,
                color=colors[j],
                alpha=0.6,
                label=f"{llm_name} (Strict)" if i == 0 else "",
            )

    # Customize the plot
    ax.set_xlabel("Winner Group Size (Number of LLMs Tied)")
    ax.set_ylabel("Number of Files Won")
    ax.set_title("LLM Winner Count Comparison: Baseline vs Strict Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{size} LLM{"s" if size > 1 else ""} tied' for size in group_sizes]
    )

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = []
    for j, llm_name in enumerate(llm_names):
        legend_elements.append(Patch(color=colors[j], label=f"{llm_name} (Baseline)"))
        legend_elements.append(
            Patch(color=colors[j], alpha=0.3, label=f"{llm_name} (Strict)")
        )

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.show()


def count_llm_winners_in_group(
    llm_data: Dict, files_in_group: Set[str], llm_names: List[str], target_llm: str
) -> int:
    """Count how many files in the group this LLM wins (has highest precision score)."""
    winner_count = 0

    for filename in files_in_group:
        # Calculate precision scores for all LLMs on this file
        llm_scores = {}

        for llm_name in llm_names:
            if filename in llm_data[llm_name]:
                file_analysis = analyze_file_precision(llm_data[llm_name][filename])
                llm_scores[llm_name] = file_analysis["avg_precision_score"]
            else:
                llm_scores[llm_name] = 0.0

        # Find the winner (LLM with highest precision score)
        if llm_scores:
            max_score = max(llm_scores.values())
            winners = [llm for llm, score in llm_scores.items() if score == max_score]

            # If target LLM is among the winners, count it
            if target_llm in winners:
                winner_count += 1

    return winner_count


def calculate_group_precision_score(
    type_info: Dict, files_in_group: Set[str], llm_name: str
) -> float:
    """Calculate precision score for a specific group of files."""
    total_slots = 0
    total_precision_score = 0

    for filename in files_in_group:
        if filename not in type_info:
            continue

        file_analysis = analyze_file_precision(type_info[filename])
        total_slots += file_analysis["total_slots"]
        total_precision_score += file_analysis["total_precision_score"]

    return total_precision_score / total_slots if total_slots > 0 else 0


def main():
    """Main analysis function comparing 6 LLMs using two baseline sets."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading baseline data...")
    untyped_mypy = load_json(UNTYPED_MYPY_PATH)
    baseline_files = build_baseline_files(untyped_mypy)
    print(f"Found {len(baseline_files)} baseline files")

    # Load all LLM data and mypy results
    print("Loading LLM data and mypy results...")
    llm_data = {}
    llm_mypy_results = {}

    for llm_name, config in LLM_CONFIGS.items():
        type_info = load_json(config["type_info_path"])
        mypy_results = load_json(config["mypy_results_path"])

        llm_data[llm_name] = type_info
        llm_mypy_results[llm_name] = mypy_results
        print(
            f"Loaded {llm_name}: {len(type_info)} files, {len(mypy_results)} mypy results"
        )

    # Build strict baseline (files where ALL LLMs succeeded)
    strict_baseline_files = build_strict_baseline_files(llm_mypy_results)
    print(
        f"Found {len(strict_baseline_files)} strict baseline files (all LLMs succeeded)"
    )

    # Group files by precision winners
    grouped_files = group_files_by_precision_winners(
        llm_data, baseline_files, list(LLM_CONFIGS.keys())
    )
    print("Files grouped by precision winners:")
    for count, files in sorted(grouped_files.items()):
        print(f"  {count} LLM(s) tied for highest precision: {len(files)} files")

    # Analyze performance using baseline files
    print("Analyzing performance using baseline files...")
    llm_results_baseline = {}
    for llm_name in LLM_CONFIGS.keys():
        llm_results_baseline[llm_name] = analyze_llm_performance(
            llm_data, baseline_files, llm_name
        )

    # Analyze performance using strict baseline files
    print("Analyzing performance using strict baseline files...")
    llm_results_strict = {}
    for llm_name in LLM_CONFIGS.keys():
        llm_results_strict[llm_name] = analyze_llm_performance(
            llm_data, strict_baseline_files, llm_name
        )

    # Create the plot
    print("Creating precision comparison plot...")
    create_precision_plot(
        llm_results_baseline,
        llm_results_strict,
        grouped_files,
        list(LLM_CONFIGS.keys()),
        llm_data,
        llm_mypy_results,
        strict_baseline_files,
    )

    # Write results to CSV files
    print("Writing results...")

    # Main comparison CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "LLM",
                "Baseline_Total_Files",
                "Baseline_Avg_Precision",
                "Strict_Total_Files",
                "Strict_Avg_Precision",
            ]
        )

        for llm_name in LLM_CONFIGS.keys():
            baseline_results = llm_results_baseline[llm_name]
            strict_results = llm_results_strict[llm_name]

            writer.writerow(
                [
                    llm_name,
                    baseline_results["total_files"],
                    f"{baseline_results['avg_precision_score']:.3f}",
                    strict_results["total_files"],
                    f"{strict_results['avg_precision_score']:.3f}",
                ]
            )

    # Detailed per-file CSV
    with open(DETAILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "success_count", "baseline_scores", "strict_scores"]
        )

        for filename in baseline_files:
            success_count = 0
            for llm_name, mypy_results in llm_mypy_results.items():
                if (
                    filename in mypy_results
                    and mypy_results[filename].get("isCompiled") is True
                ):
                    success_count += 1

            baseline_scores = []
            strict_scores = []

            for llm_name in LLM_CONFIGS.keys():
                baseline_score = llm_results_baseline[llm_name]["file_scores"].get(
                    filename, 0.0
                )
                strict_score = llm_results_strict[llm_name]["file_scores"].get(
                    filename, 0.0
                )

                baseline_scores.append(f"{llm_name}:{baseline_score:.3f}")
                strict_scores.append(f"{llm_name}:{strict_score:.3f}")

            writer.writerow(
                [
                    filename,
                    success_count,
                    ";".join(baseline_scores),
                    ";".join(strict_scores),
                ]
            )

    # Ranking CSV
    with open(RANKING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Rank",
                "LLM",
                "Baseline_Avg_Precision",
                "Strict_Avg_Precision",
                "Baseline_Files",
                "Strict_Files",
            ]
        )

        # Sort by baseline precision score
        ranked_llms = sorted(
            llm_results_baseline.items(),
            key=lambda x: x[1]["avg_precision_score"],
            reverse=True,
        )

        for rank, (llm_name, baseline_results) in enumerate(ranked_llms, 1):
            strict_results = llm_results_strict[llm_name]

            writer.writerow(
                [
                    rank,
                    llm_name,
                    f"{baseline_results['avg_precision_score']:.3f}",
                    f"{strict_results['avg_precision_score']:.3f}",
                    baseline_results["total_files"],
                    strict_results["total_files"],
                ]
            )

    # Print summary
    print(f"\nResults written to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {DETAILED_CSV}")
    print(f"  {RANKING_CSV}")
    print(f"  {PLOT_PATH}")

    print(f"\nLLM Precision Ranking (Baseline vs Strict):")
    print("-" * 80)
    for rank, (llm_name, baseline_results) in enumerate(ranked_llms, 1):
        strict_results = llm_results_strict[llm_name]
        print(
            f"{rank}. {llm_name:<15} Baseline: {baseline_results['avg_precision_score']:.3f} "
            f"({baseline_results['total_files']} files) | "
            f"Strict: {strict_results['avg_precision_score']:.3f} "
            f"({strict_results['total_files']} files)"
        )


if __name__ == "__main__":
    main()
