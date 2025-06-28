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
import matplotlib.patches as patches


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


def count_color_buckets(all_data):
    """Count and print items in each color bucket"""
    colors = {
        "any": "#808080",  # Gray for Any/Dyn
        "match": "#1f77b4",  # Blue for exact matches
        "any_mismatch": "#9467bd",  # Purple for Any vs specific
        "different": "#d62728",  # Red for different specific types
        "empty": "#ffffff",  # White for empty annotations
        "primitive": "#ff7f0e",  # Orange for primitives
        "container": "#2ca02c",  # Green for containers
        "user_class": "#e377c2",  # Pink for user classes
    }

    color_counts = {color_name: 0 for color_name in colors.keys()}

    for llm_name, data in all_data.items():
        for filename, functions in data.items():
            for func_sig, annotations in functions.items():
                for annotation in annotations:
                    human_type = annotation["Human"]
                    llm_type = annotation["LLM"]
                    match = annotation["match"]

                    human_color, llm_color = get_comparison_colors(
                        human_type, llm_type, match, colors
                    )

                    # Count human color
                    for color_name, color_code in colors.items():
                        if human_color == color_code:
                            color_counts[color_name] += 1
                            break

                    # Count LLM color
                    for color_name, color_code in colors.items():
                        if llm_color == color_code:
                            color_counts[color_name] += 1
                            break

    print("\nColor Bucket Counts:")
    print("=" * 40)
    for color_name, count in color_counts.items():
        print(f"{color_name:15}: {count:5} items")
    print("=" * 40)

    return color_counts


def count_color_buckets_for_file(file_data, colors):
    """Count and print items in each color bucket for a specific file"""
    color_counts = {
        "both_any": 0,  # Both Any/Dyn: Gray + Gray
        "both_match": 0,  # Both match (not Any): Blue + Blue
        "both_different": 0,  # Both different (not Any): Red + Red
        "any_mismatch": 0,  # One Any, one specific: Gray + Purple
    }

    for llm_name, params in file_data.items():
        for param in params:
            human_type = param["human_type"]
            llm_type = param["llm_type"]
            match = param["match"]

            human_color, llm_color = get_comparison_colors(
                human_type, llm_type, match, colors
            )

            # Count based on the 4 scenarios
            if human_color == colors["any"] and llm_color == colors["any"]:
                color_counts["both_any"] += 1
            elif human_color == colors["match"] and llm_color == colors["match"]:
                color_counts["both_match"] += 1
            elif (
                human_color == colors["different"] and llm_color == colors["different"]
            ):
                color_counts["both_different"] += 1
            elif (
                human_color == colors["any"] and llm_color == colors["any_mismatch"]
            ) or (human_color == colors["any_mismatch"] and llm_color == colors["any"]):
                color_counts["any_mismatch"] += 1

    print(f"\nColor Bucket Counts for file:")
    print("=" * 50)
    print(f"{'Both Any/Dyn':20}: {color_counts['both_any']:5} items")
    print(f"{'Both match (not Any)':20}: {color_counts['both_match']:5} items")
    print(f"{'Both different (not Any)':20}: {color_counts['both_different']:5} items")
    print(f"{'One Any, one specific':20}: {color_counts['any_mismatch']:5} items")
    print("=" * 50)

    return color_counts


def create_rectangle_comparison_visualization(all_data):
    """
    Create rectangle-based visualization comparing human vs LLM type annotations.
    Each project/file gets two rectangles (human and LLM) divided by parameters.
    Shows top-5 most interesting files (those with data in all four color buckets).
    """
    # Color scheme
    colors = {
        "any": "#808080",  # Gray for Any/Dyn
        "match": "#1f77b4",  # Blue for exact matches
        "any_mismatch": "#9467bd",  # Purple for Any vs specific
        "different": "#d62728",  # Red for different specific types
        "empty": "#ffffff",  # White for empty annotations
        "primitive": "#ff7f0e",  # Orange for primitives
        "container": "#2ca02c",  # Green for containers
        "user_class": "#e377c2",  # Pink for user classes
    }

    # Create output directories
    output_dir = "rectangle_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Process data by project/file
    project_data = {}

    for llm_name, data in all_data.items():
        for filename, functions in data.items():
            if filename not in project_data:
                project_data[filename] = {}

            project_data[filename][llm_name] = []

            for func_sig, annotations in functions.items():
                for annotation in annotations:
                    human_type = annotation["Human"]
                    llm_type = annotation["LLM"]
                    match = annotation["match"]

                    project_data[filename][llm_name].append(
                        {"human_type": human_type, "llm_type": llm_type, "match": match}
                    )

    # Calculate color counts for all files and find most interesting ones
    file_interest_scores = {}
    number_of_files_with_all_4_buckets = 0
    for filename, llm_data in project_data.items():
        if len(llm_data) == 0:
            continue

        # Calculate color counts for this file
        color_counts = {
            "both_any": 0,  # Both Any/Dyn: Gray + Gray
            "both_match": 0,  # Both match (not Any): Blue + Blue
            "both_different": 0,  # Both different (not Any): Red + Red
            "any_mismatch": 0,  # One Any, one specific: Gray + Purple
        }

        for llm_name, params in llm_data.items():
            for param in params:
                human_type = param["human_type"]
                llm_type = param["llm_type"]
                match = param["match"]

                human_color, llm_color = get_comparison_colors(
                    human_type, llm_type, match, colors
                )

                # Count based on the 4 scenarios
                if human_color == colors["any"] and llm_color == colors["any"]:
                    color_counts["both_any"] += 1
                elif human_color == colors["match"] and llm_color == colors["match"]:
                    color_counts["both_match"] += 1
                elif (
                    human_color == colors["different"]
                    and llm_color == colors["different"]
                ):
                    color_counts["both_different"] += 1
                elif (
                    human_color == colors["any"] and llm_color == colors["any_mismatch"]
                ) or (
                    human_color == colors["any_mismatch"] and llm_color == colors["any"]
                ):
                    color_counts["any_mismatch"] += 1

        # Calculate interest score: prioritize files with all 4 buckets, then by total items
        buckets_with_data = sum(1 for count in color_counts.values() if count > 0)
        total_items = sum(color_counts.values())

        # Prioritize files with data in all 4 buckets, then by total number of items
        if buckets_with_data == 4:
            number_of_files_with_all_4_buckets += 1
            interest_score = 10000 + total_items  # Highest priority for 4-bucket files
        else:
            interest_score = (
                buckets_with_data * 1000 + total_items
            )  # Lower priority for fewer buckets

        file_interest_scores[filename] = {
            "interest_score": interest_score,
            "buckets_with_data": buckets_with_data,
            "total_items": total_items,
            "color_counts": color_counts,
        }

    # Sort by interest score and take top-5
    top_files = sorted(
        file_interest_scores.items(), key=lambda x: x[1]["interest_score"], reverse=True
    )[:5]
    top_filenames = [filename for filename, info in top_files]

    print(f"Number of files with all 4 buckets: {number_of_files_with_all_4_buckets}")
    print(f"Top-5 most interesting files:")
    for i, (filename, info) in enumerate(top_files, 1):
        print(
            f"{i}. {filename}: {info['total_items']} items, {info['buckets_with_data']}/4 buckets with data"
        )
        print(f"   Color counts: {info['color_counts']}")

    # Create visualizations for top-5 files only
    figures_created = 0

    for filename in top_filenames:
        llm_data = project_data[filename]
        if len(llm_data) == 0:
            continue

        # Count color buckets for this specific file
        count_color_buckets_for_file(llm_data, colors)

        # Calculate total items across all LLMs for this file
        max_items = max(len(params) for params in llm_data.values())
        if max_items == 0:
            continue

        # Create figure
        fig, axes = plt.subplots(len(llm_data), 2, figsize=(15, 3 * len(llm_data)))
        if len(llm_data) == 1:
            axes = axes.reshape(1, -1)

        llm_names = list(llm_data.keys())

        for i, llm_name in enumerate(llm_names):
            params = llm_data[llm_name]

            # Human rectangle (top)
            ax_human = axes[i, 0]
            ax_llm = axes[i, 1]

            # Draw rectangles
            rect_width = 0.8
            rect_height = 0.6

            # Human annotations rectangle
            human_rect = patches.Rectangle(
                (0.1, 0.2),
                rect_width,
                rect_height,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax_human.add_patch(human_rect)

            # LLM annotations rectangle
            llm_rect = patches.Rectangle(
                (0.1, 0.2),
                rect_width,
                rect_height,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax_llm.add_patch(llm_rect)

            # Divide rectangles into parameter sections
            param_width = rect_width / max_items

            for j, param in enumerate(params):
                human_type = param["human_type"]
                llm_type = param["llm_type"]
                match = param["match"]

                # Determine colors based on comparison logic
                human_color, llm_color = get_comparison_colors(
                    human_type, llm_type, match, colors
                )

                # Draw parameter sections
                x_pos = 0.1 + j * param_width

                # Human parameter section
                human_param_rect = patches.Rectangle(
                    (x_pos, 0.2),
                    param_width,
                    rect_height,
                    facecolor=human_color,
                    alpha=0.7,
                )
                ax_human.add_patch(human_param_rect)

                # LLM parameter section
                llm_param_rect = patches.Rectangle(
                    (x_pos, 0.2),
                    param_width,
                    rect_height,
                    facecolor=llm_color,
                    alpha=0.7,
                )
                ax_llm.add_patch(llm_param_rect)

                # Add parameter number labels
                ax_human.text(
                    x_pos + param_width / 2,
                    0.1,
                    str(j + 1),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax_llm.text(
                    x_pos + param_width / 2,
                    0.1,
                    str(j + 1),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            # Set titles and labels
            ax_human.set_title(f"{llm_name} - Human Annotations")
            ax_llm.set_title(f"{llm_name} - LLM Annotations")
            ax_human.set_xlim(0, 1)
            ax_human.set_ylim(0, 1)
            ax_llm.set_xlim(0, 1)
            ax_llm.set_ylim(0, 1)
            ax_human.axis("off")
            ax_llm.axis("off")

        # Add overall title
        fig.suptitle(f"Type Annotation Comparison: {filename}", fontsize=14)

        # Add legend
        legend_elements = [
            patches.Patch(color=colors["any"], label="Any/Dyn"),
            patches.Patch(color=colors["match"], label="Exact Match"),
            patches.Patch(color=colors["any_mismatch"], label="Any vs Specific"),
            patches.Patch(color=colors["different"], label="Different Types"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

        plt.tight_layout()
        figures_created += 1

        # Save the figure to file
        safe_filename = filename.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = os.path.join(
            output_dir, f"rectangle_comparison_{safe_filename}.pdf"
        )
        # plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

        # Show the figure
        plt.show()

    return figures_created


def normalize_type(type_str: str) -> str:
    """Normalize type string like in the original function"""
    if not type_str:
        return "dyn"
    # Remove quotes and spaces, then convert to lowercase
    return type_str.replace(" ", "").replace("'", "").replace('"', "").lower()


def get_comparison_colors(human_type, llm_type, match, colors):
    """Helper function to determine colors for human and LLM type comparison"""

    def is_any_or_none_type(type_str):
        """Check if type is Any/None (should be skipped in original logic)"""
        if not type_str or type_str.strip() == "":
            return True
        clean_type = type_str.strip().lower()
        return clean_type in ["any", "none", "dyn", "dynamic"]

    # Check if either type is Any/None (these should be skipped in original logic)
    human_is_any_none = is_any_or_none_type(human_type)
    llm_is_any_none = is_any_or_none_type(llm_type)

    # If either is Any/None, use gray (these were skipped in original comparison)
    if human_is_any_none and llm_is_any_none:
        return colors["any"], colors["any"]
    elif human_is_any_none or llm_is_any_none:
        return colors["any"], colors["any_mismatch"]
    elif not human_is_any_none and llm_is_any_none:
        return colors["any_mismatch"], colors["any"]

    # For non-Any/None types, use the match status from the data
    if match:
        return colors["match"], colors["match"]
    else:
        return colors["different"], colors["different"]


def main():
    """Main function to generate all visualizations"""
    llm_names = ["o1-mini"]  # Only o1-mini
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
        # create_error_pattern_analysis_combined(all_data)
        # create_same_pattern_analysis_combined(all_data)
        # create_llm_comparison_analysis(all_data)
        create_rectangle_comparison_visualization(all_data)
    else:
        print("No data files found!")


if __name__ == "__main__":
    main()
