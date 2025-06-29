import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import defaultdict


def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, "r") as f:
        return json.load(f)


def normalize_type(type_str: str) -> str:
    """Normalize type string"""
    if not type_str:
        return "dyn"
    return type_str.replace(" ", "").replace("'", "").replace('"', "").lower()


def is_any_or_none_type(type_str):
    """Check if type is Any/None"""
    if not type_str or type_str.strip() == "":
        return True
    clean_type = type_str.strip().lower()
    return clean_type in ["any", "none", "dyn", "dynamic"]


def is_primitive_type(type_str):
    """Check if type is primitive"""
    if not type_str:
        return False
    clean_type = type_str.strip().lower()
    primitive_types = {
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "bytes",
        "bytearray",
        "complex",
        "frozenset",
        "range",
        "slice",
        "memoryview",
        "object",
        "type",
        "ellipsis",
        "notimplemented",
    }
    return clean_type in primitive_types


def get_comparison_colors(human_type, llm_type, colors):
    """Get colors for human and LLM type comparison"""
    human_is_any_none = is_any_or_none_type(human_type)
    llm_is_any_none = is_any_or_none_type(llm_type)

    # Check if types match (after normalization)
    if not human_is_any_none and not llm_is_any_none:
        human_norm = normalize_type(human_type)
        llm_norm = normalize_type(llm_type)
        match = human_norm == llm_norm
    else:
        match = False

    # Color logic
    if human_is_any_none and llm_is_any_none:
        return colors["any"], colors["any"]
    elif human_is_any_none and not llm_is_any_none:
        return colors["any"], colors["any_mismatch"]
    elif not human_is_any_none and llm_is_any_none:
        return colors["any_mismatch"], colors["any"]
    elif match:
        # Check if both are primitive types
        if is_primitive_type(human_type) and is_primitive_type(llm_type):
            return colors["primitive_match"], colors["primitive_match"]
        else:
            return colors["non_primitive_match"], colors["non_primitive_match"]
    else:
        return colors["different"], colors["different"]


def extract_type_annotations(data):
    """Extract type annotations from the JSON structure"""
    annotations = {}

    for filename, file_data in data.items():
        if not isinstance(file_data, dict):
            continue

        file_annotations = []

        for func_sig, func_data in file_data.items():
            if not isinstance(func_data, list):
                continue

            for item in func_data:
                if not isinstance(item, dict):
                    continue

                # Extract type information
                type_list = item.get("type", [])
                if isinstance(type_list, list) and len(type_list) > 0:
                    type_str = type_list[0]
                    if isinstance(type_str, str):
                        file_annotations.append(
                            {
                                "category": item.get("category", "unknown"),
                                "name": item.get("name", "unknown"),
                                "type": type_str,
                            }
                        )

        if file_annotations:
            annotations[filename] = file_annotations

    return annotations


def count_color_buckets_for_file(human_annotations, llm_annotations, colors):
    """Count and print items in each color bucket for a specific file"""
    color_counts = {
        "both_any": 0,  # Both Any/Dyn: Gray + Gray
        "primitive_match": 0,  # Primitive type matches: Green + Green
        "non_primitive_match": 0,  # Non-primitive type matches: Orange + Orange
        "both_different": 0,  # Both different (not Any): Red + Red
        "any_mismatch": 0,  # One Any, one specific: Gray + Purple
    }

    # Match annotations by category and name
    human_dict = {
        (ann["category"], ann["name"]): ann["type"] for ann in human_annotations
    }
    llm_dict = {(ann["category"], ann["name"]): ann["type"] for ann in llm_annotations}

    # Find common annotations
    common_keys = set(human_dict.keys()) & set(llm_dict.keys())

    for key in common_keys:
        human_type = human_dict[key]
        llm_type = llm_dict[key]

        human_color, llm_color = get_comparison_colors(human_type, llm_type, colors)

        # Count based on the 5 scenarios
        if human_color == colors["any"] and llm_color == colors["any"]:
            color_counts["both_any"] += 1
        elif (
            human_color == colors["primitive_match"]
            and llm_color == colors["primitive_match"]
        ):
            color_counts["primitive_match"] += 1
        elif (
            human_color == colors["non_primitive_match"]
            and llm_color == colors["non_primitive_match"]
        ):
            color_counts["non_primitive_match"] += 1
        elif human_color == colors["different"] and llm_color == colors["different"]:
            color_counts["both_different"] += 1
        elif (human_color == colors["any"] and llm_color == colors["any_mismatch"]) or (
            human_color == colors["any_mismatch"] and llm_color == colors["any"]
        ):
            color_counts["any_mismatch"] += 1

    print(f"\nColor Bucket Counts for file:")
    print("=" * 50)
    print(f"{'Both Any/Dyn':20}: {color_counts['both_any']:5} items")
    print(f"{'Primitive matches':20}: {color_counts['primitive_match']:5} items")
    print(
        f"{'Non-primitive matches':20}: {color_counts['non_primitive_match']:5} items"
    )
    print(f"{'Both different (not Any)':20}: {color_counts['both_different']:5} items")
    print(f"{'One Any, one specific':20}: {color_counts['any_mismatch']:5} items")
    print("=" * 50)

    return color_counts


def create_rectangle_visualization(human_file, llm_file):
    """Create rectangle visualization comparing human vs LLM type annotations"""

    # Color scheme
    colors = {
        "any": "#808080",  # Gray for Any/Dyn
        "primitive_match": "#87CEEB",  # Light blue for primitive type matches
        "non_primitive_match": "#000080",  # Dark blue for non-primitive type matches
        "any_mismatch": "#9467bd",  # Purple for Any vs specific
        "different": "#d62728",  # Red for different specific types
    }

    # Create output directory
    output_dir = "simple_rectangle_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading human annotations...")
    human_data = load_json_data(human_file)
    print("Loading LLM annotations...")
    llm_data = load_json_data(llm_file)

    # Extract type annotations
    print("Extracting type annotations...")
    human_annotations = extract_type_annotations(human_data)
    llm_annotations = extract_type_annotations(llm_data)

    # Find common files
    common_files = set(human_annotations.keys()) & set(llm_annotations.keys())
    print(f"Found {len(common_files)} common files")

    # Calculate interest scores for all files
    file_interest_scores = {}
    number_of_files_with_all_5_buckets = 0

    for filename in common_files:
        human_file_anns = human_annotations[filename]
        llm_file_anns = llm_annotations[filename]

        # Calculate color counts
        color_counts = {
            "both_any": 0,
            "primitive_match": 0,
            "non_primitive_match": 0,
            "both_different": 0,
            "any_mismatch": 0,
        }

        # Match annotations by category and name
        human_dict = {
            (ann["category"], ann["name"]): ann["type"] for ann in human_file_anns
        }
        llm_dict = {
            (ann["category"], ann["name"]): ann["type"] for ann in llm_file_anns
        }
        common_keys = set(human_dict.keys()) & set(llm_dict.keys())

        for key in common_keys:
            human_type = human_dict[key]
            llm_type = llm_dict[key]

            human_color, llm_color = get_comparison_colors(human_type, llm_type, colors)

            # Count based on the 5 scenarios
            if human_color == colors["any"] and llm_color == colors["any"]:
                color_counts["both_any"] += 1
            elif (
                human_color == colors["primitive_match"]
                and llm_color == colors["primitive_match"]
            ):
                color_counts["primitive_match"] += 1
            elif (
                human_color == colors["non_primitive_match"]
                and llm_color == colors["non_primitive_match"]
            ):
                color_counts["non_primitive_match"] += 1
            elif (
                human_color == colors["different"] and llm_color == colors["different"]
            ):
                color_counts["both_different"] += 1
            elif (
                human_color == colors["any"] and llm_color == colors["any_mismatch"]
            ) or (human_color == colors["any_mismatch"] and llm_color == colors["any"]):
                color_counts["any_mismatch"] += 1

        # Calculate interest score
        buckets_with_data = sum(1 for count in color_counts.values() if count > 0)
        total_items = sum(color_counts.values())

        if buckets_with_data == 5:
            number_of_files_with_all_5_buckets += 1
            interest_score = 10000 + total_items
        else:
            interest_score = buckets_with_data * 1000 + total_items

        file_interest_scores[filename] = {
            "interest_score": interest_score,
            "buckets_with_data": buckets_with_data,
            "total_items": total_items,
            "color_counts": color_counts,
            "human_annotations": human_file_anns,
            "llm_annotations": llm_file_anns,
        }

    # Sort by interest score and take top-5
    top_files = sorted(
        file_interest_scores.items(), key=lambda x: x[1]["interest_score"], reverse=True
    )[:5]

    print(f"Number of files with all 5 buckets: {number_of_files_with_all_5_buckets}")
    print(f"Top-5 most interesting files:")
    for i, (filename, info) in enumerate(top_files, 1):
        print(
            f"{i}. {filename}: {info['total_items']} items, {info['buckets_with_data']}/5 buckets with data"
        )
        print(f"   Color counts: {info['color_counts']}")

    # Create visualizations
    figures_created = 0

    for filename, info in top_files:
        human_file_anns = info["human_annotations"]
        llm_file_anns = info["llm_annotations"]

        # Count color buckets for this specific file
        count_color_buckets_for_file(human_file_anns, llm_file_anns, colors)

        # Match annotations by category and name
        human_dict = {
            (ann["category"], ann["name"]): ann["type"] for ann in human_file_anns
        }
        llm_dict = {
            (ann["category"], ann["name"]): ann["type"] for ann in llm_file_anns
        }
        common_keys = set(human_dict.keys()) & set(llm_dict.keys())

        if not common_keys:
            continue

        # Create figure
        fig, (ax_human, ax_llm) = plt.subplots(
            2, 1, figsize=(10, 4), height_ratios=[1, 1]
        )

        # Draw rectangles
        rect_width = 0.8
        rect_height = 0.6
        max_items = len(common_keys)

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

        for j, key in enumerate(sorted(common_keys)):
            category, name = key
            human_type = human_dict[key]
            llm_type = llm_dict[key]

            # Determine colors based on comparison logic
            human_color, llm_color = get_comparison_colors(human_type, llm_type, colors)

            # Draw parameter sections
            x_pos = 0.1 + j * param_width

            # Human parameter section
            human_param_rect = patches.Rectangle(
                (x_pos, 0.2), param_width, rect_height, facecolor=human_color
            )
            ax_human.add_patch(human_param_rect)

            # LLM parameter section
            llm_param_rect = patches.Rectangle(
                (x_pos, 0.2), param_width, rect_height, facecolor=llm_color
            )
            ax_llm.add_patch(llm_param_rect)

        # Set titles and labels
        ax_human.set_title("Human Annotations", pad=5, y=0.85)
        ax_llm.set_title("LLM Annotations", pad=5, y=0.85)
        ax_human.set_xlim(0, 1)
        ax_human.set_ylim(0, 1)
        ax_llm.set_xlim(0, 1)
        ax_llm.set_ylim(0, 1)
        ax_human.axis("off")
        ax_llm.axis("off")

        # Add overall title
        fig.suptitle(f"Type Annotation Comparison: {filename}", fontsize=14, y=0.95)

        # Add legend
        legend_elements = [
            patches.Patch(color=colors["any"], label="Any/Dyn"),
            patches.Patch(
                color=colors["primitive_match"], label="Primitive Type Match"
            ),
            patches.Patch(
                color=colors["non_primitive_match"], label="Non-Primitive Type Match"
            ),
            patches.Patch(color=colors["any_mismatch"], label="Any vs Specific"),
            patches.Patch(color=colors["different"], label="Different Types"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, right=0.75, top=0.9)
        figures_created += 1

        # Save the figure to file
        output_dir = "rectangle_visualizations/gpt4o"
        os.makedirs(output_dir, exist_ok=True)
        safe_filename = filename.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = os.path.join(
            output_dir, f"rectangle_comparison_{safe_filename}.pdf"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

        # Show the figure
        # plt.show()

    return figures_created


if __name__ == "__main__":
    # File paths
    human_file = "Type_info_original_files.json"
    llm_file = "Type_info_gpt4o_benchmarks.json"

    print("Starting rectangle visualization generation...")
    figures_created = create_rectangle_visualization(human_file, llm_file)
    print(f"Created {figures_created} figures")
