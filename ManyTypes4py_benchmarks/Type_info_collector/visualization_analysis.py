import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


def load_comparison_data(file_path):
    """Load JSON comparison data"""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_type_categories(data):
    """Extract human and LLM type categories from comparison data"""
    human_types = []
    llm_types = []
    match_status = []

    for filename, functions in data.items():
        for func_sig, annotations in functions.items():
            for annotation in annotations:
                human_type = annotation["Human"]
                llm_type = annotation["LLM"]
                match = annotation["match"]

                # Clean and categorize types
                human_category = get_type_category(human_type)
                llm_category = get_type_category(llm_type)

                human_types.append(human_category)
                llm_types.append(llm_category)
                match_status.append(match)

    return human_types, llm_types, match_status


def get_type_category(type_str):
    """Extract basic type category from type string"""
    if not type_str or type_str.strip() == "":
        return "empty"

    # Remove quotes and normalize
    clean_type = type_str.replace("'", "").replace('"', "").strip()

    # Handle common patterns
    if clean_type.startswith("List["):
        return "List"
    elif clean_type.startswith("Dict["):
        return "Dict"
    elif clean_type.startswith("Optional["):
        return "Optional"
    elif clean_type.startswith("Union["):
        return "Union"
    elif " | " in clean_type:
        return "Union"
    elif clean_type in ["str", "string"]:
        return "str"
    elif clean_type in ["int", "integer"]:
        return "int"
    elif clean_type in ["float", "double"]:
        return "float"
    elif clean_type in ["bool", "boolean"]:
        return "bool"
    elif clean_type in ["None", "NoneType"]:
        return "None"
    elif clean_type.lower() == "any":
        return "Any"
    else:
        # Try to extract base type from complex types
        if "[" in clean_type:
            base = clean_type.split("[")[0]
            return base
        else:
            return clean_type


def create_type_category_heatmap(data, llm_name):
    """Create heatmap showing match rates by type categories"""
    human_types, llm_types, match_status = extract_type_categories(data)

    # Create DataFrame
    df = pd.DataFrame(
        {"Human_Type": human_types, "LLM_Type": llm_types, "Match": match_status}
    )

    # Create pivot table for heatmap
    heatmap_data = (
        df.groupby(["Human_Type", "LLM_Type"])
        .agg({"Match": ["count", "sum"]})
        .reset_index()
    )

    heatmap_data.columns = ["Human_Type", "LLM_Type", "Total", "Matches"]
    heatmap_data["Match_Rate"] = heatmap_data["Matches"] / heatmap_data["Total"]

    # Create pivot for heatmap
    pivot_data = heatmap_data.pivot(
        index="Human_Type", columns="LLM_Type", values="Match_Rate"
    )

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"label": "Match Rate"},
    )
    plt.title(f"Type Category Match Heatmap - {llm_name}")
    plt.xlabel("LLM Predicted Type")
    plt.ylabel("Human Type")
    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(
        f"visualizations/type_category_heatmap_{llm_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_progressive_match_analysis(data, llm_name):
    """Create progressive match analysis showing exact vs top-level matches"""
    # Load both exact and top-level data
    exact_data = data  # This is the exact match data

    # For this example, we'll simulate top-level data
    # In practice, you'd load the top_level_comparison file
    top_level_file = f"top_level_comparison_{llm_name}.json"
    try:
        with open(top_level_file, "r") as f:
            top_level_data = json.load(f)
    except FileNotFoundError:
        print(f"Top-level comparison file not found: {top_level_file}")
        return

    # Count matches
    exact_matches = 0
    top_level_matches = 0
    total_comparisons = 0

    for filename, functions in exact_data.items():
        for func_sig, annotations in functions.items():
            for annotation in annotations:
                total_comparisons += 1
                if annotation["match"]:
                    exact_matches += 1

    for filename, functions in top_level_data.items():
        for func_sig, annotations in functions.items():
            for annotation in annotations:
                if annotation["match"]:
                    top_level_matches += 1

    # Calculate percentages
    exact_rate = exact_matches / total_comparisons * 100
    top_level_rate = top_level_matches / total_comparisons * 100
    no_match_rate = 100 - top_level_rate

    # Create stacked bar chart
    categories = ["Exact Match", "Top-Level Only", "No Match"]
    values = [exact_rate, top_level_rate - exact_rate, no_match_rate]
    colors = ["#2E8B57", "#FFD700", "#DC143C"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors)
    plt.title(f"Progressive Match Analysis - {llm_name}")
    plt.ylabel("Percentage (%)")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(
        f"visualizations/progressive_match_analysis_{llm_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_error_pattern_analysis(data, llm_name):
    """Create analysis showing which human types are most commonly mistyped"""
    human_type_errors = defaultdict(int)

    for filename, functions in data.items():
        for func_sig, annotations in functions.items():
            for annotation in annotations:
                if not annotation["match"]:  # Only look at mismatches
                    human_type = get_type_category(annotation["Human"])
                    human_type_errors[human_type] += 1

    # Get top mistyped human types
    top_errors = sorted(human_type_errors.items(), key=lambda x: x[1], reverse=True)[
        :15
    ]

    # Create bar chart of most mistyped human types
    human_types = [error[0] for error in top_errors]
    frequencies = [error[1] for error in top_errors]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(human_types)), frequencies)
    plt.yticks(range(len(human_types)), human_types)
    plt.xlabel("Number of Times Mismatched")
    plt.title(f"Most Commonly Mismatched Human Types - {llm_name}")
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(freq),
            va="center",
        )

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(
        f"visualizations/mismatch_analysis_{llm_name}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def create_same_pattern_analysis(data, llm_name):
    """Create analysis showing which human types are most commonly correctly predicted"""
    human_type_correct = defaultdict(int)

    for filename, functions in data.items():
        for func_sig, annotations in functions.items():
            for annotation in annotations:
                if annotation["match"]:  # Only look at correct matches
                    human_type = get_type_category(annotation["Human"])
                    human_type_correct[human_type] += 1

    # Get top correctly predicted human types
    top_correct = sorted(human_type_correct.items(), key=lambda x: x[1], reverse=True)[
        :15
    ]

    # Create bar chart of most correctly predicted human types
    human_types = [correct[0] for correct in top_correct]
    frequencies = [correct[1] for correct in top_correct]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(human_types)), frequencies, color="green")
    plt.yticks(range(len(human_types)), human_types)
    plt.xlabel("Number of Times Correctly Predicted")
    plt.title(f"Most Commonly Correctly Predicted Human Types - {llm_name}")
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(freq),
            va="center",
        )

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(
        f"visualizations/correct_analysis_{llm_name}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def create_error_pattern_analysis_combined(all_data):
    """Create single figure with grouped bars showing top 10 ranking positions"""
    # Get top 10 types for each LLM separately
    llm_top_types = {}

    for llm_name, data in all_data.items():
        human_type_errors = defaultdict(int)
        for filename, functions in data.items():
            for func_sig, annotations in functions.items():
                for annotation in annotations:
                    if not annotation["match"]:
                        human_type = get_type_category(annotation["Human"])
                        human_type_errors[human_type] += 1

        # Get top 10 for this LLM
        top_10 = sorted(human_type_errors.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        llm_top_types[llm_name] = top_10

    # Prepare data for plotting
    positions = list(range(1, 11))  # Position 1 to 10
    x = np.arange(len(positions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 8))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    llm_names = list(all_data.keys())

    for i, (llm_name, color) in enumerate(zip(llm_names, colors)):
        counts = []
        labels = []

        for position in range(10):
            if position < len(llm_top_types[llm_name]):
                type_name, count = llm_top_types[llm_name][position]
                counts.append(count)
                labels.append(f"{type_name}")
            else:
                counts.append(0)
                labels.append("")

        bars = ax.bar(x + i * width, counts, width, label=llm_name, color=color)

        # Add value labels on bars
        for bar, count, label in zip(bars, counts, labels):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{count}\n{label}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Ranking Position")
    ax.set_ylabel("Number of Times Mismatched")
    ax.set_title(
        "Top 10 Mismatched Types by Ranking Position (Each LLM has its own top 10)"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Position {i}" for i in positions])
    ax.legend()

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/mismatch_analysis_combined.pdf", bbox_inches="tight")
    plt.show()


def create_same_pattern_analysis_combined(all_data):
    """Create single figure with grouped bars showing top 10 ranking positions"""
    # Get top 10 types for each LLM separately
    llm_top_types = {}

    for llm_name, data in all_data.items():
        human_type_correct = defaultdict(int)
        for filename, functions in data.items():
            for func_sig, annotations in functions.items():
                for annotation in annotations:
                    if annotation["match"]:
                        human_type = get_type_category(annotation["Human"])
                        human_type_correct[human_type] += 1

        # Get top 10 for this LLM
        top_10 = sorted(human_type_correct.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        llm_top_types[llm_name] = top_10

    # Prepare data for plotting
    positions = list(range(1, 11))  # Position 1 to 10
    x = np.arange(len(positions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 8))

    colors = ["#2E8B57", "#32CD32", "#228B22"]  # Different shades of green
    llm_names = list(all_data.keys())

    for i, (llm_name, color) in enumerate(zip(llm_names, colors)):
        counts = []
        labels = []

        for position in range(10):
            if position < len(llm_top_types[llm_name]):
                type_name, count = llm_top_types[llm_name][position]
                counts.append(count)
                labels.append(f"{type_name}")
            else:
                counts.append(0)
                labels.append("")

        bars = ax.bar(x + i * width, counts, width, label=llm_name, color=color)

        # Add value labels on bars
        for bar, count, label in zip(bars, counts, labels):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{count}\n{label}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Ranking Position")
    ax.set_ylabel("Number of Times Correctly Predicted")
    ax.set_title(
        "Top 10 Correctly Predicted Types by Ranking Position (Each LLM has its own top 10)"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Position {i}" for i in positions])
    ax.legend()

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/correct_analysis_combined.pdf", bbox_inches="tight")
    plt.show()


def create_llm_comparison_analysis(all_data):
    """Create comparison analysis showing performance of each LLM side by side"""
    llm_performance = {}

    for llm_name, data in all_data.items():
        total_matches = 0
        total_comparisons = 0

        for filename, functions in data.items():
            for func_sig, annotations in functions.items():
                for annotation in annotations:
                    total_comparisons += 1
                    if annotation["match"]:
                        total_matches += 1

        if total_comparisons > 0:
            match_rate = (total_matches / total_comparisons) * 100
            llm_performance[llm_name] = match_rate

    # Create bar chart comparing LLMs
    llm_names = list(llm_performance.keys())
    match_rates = list(llm_performance.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(llm_names, match_rates, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    plt.ylabel("Match Rate (%)")
    plt.title("Type Prediction Performance Comparison Across LLMs")
    plt.ylim(0, 100)

    # Add value labels on bars
    for bar, rate in zip(bars, match_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Create visualizations folder if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/llm_comparison_analysis.pdf", bbox_inches="tight")
    plt.show()


def main():
    """Main function to generate all visualizations"""
    llm_names = ["gpt4o", "o1-mini", "deepseek"]
    all_data = {}

    # Load all comparison data
    for llm_name in llm_names:
        print(f"Loading {llm_name}...")
        file_path = f"type_comparison_{llm_name}.json"
        try:
            data = load_comparison_data(file_path)
            all_data[llm_name] = data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    if all_data:
        print("Generating combined visualizations...")
        # Generate combined visualizations
        create_error_pattern_analysis_combined(all_data)
        create_same_pattern_analysis_combined(all_data)
        # create_llm_comparison_analysis(all_data)
    else:
        print("No data files found!")


if __name__ == "__main__":
    main()
