"""
Compilation Consistency Visualization

Creates a scatter plot showing compilation consistency across LLM runs:
- X-axis: Number of parameters in each file
- Y-axis: Different LLM models (positioned at y=1,2,3,4,5,6)
- Colors: Based on compilation status across 1st and 2nd runs

Color coding:
1. File not present in 1st run (red)
2. File not present in 2nd run (orange)
3. File compiled successfully in both runs (green)
4. File compiled successfully only in 1st run (blue)
5. File compiled successfully only in 2nd run (purple)
6. File failed compilation in both runs (gray)
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_json_file(filename):
    """Load JSON file and return data"""
    with open(filename, "r") as f:
        return json.load(f)

def has_syntax_error(errors):
    non_type_related_errors = [
        "name-defined",
        "import",
        "syntax",
        "no-redef",
        "unused-ignore",
        "override-without-super",
        "redundant-cast",
        "literal-required",
        "typeddict-unknown-key",
        "typeddict-item",
        "truthy-function",
        "str-bytes-safe",
        "unused-coroutine",
        "explicit-override",
        "truthy-iterable",
        "redundant-self",
        "redundant-await",
        "unreachable",
    ]

    def extract_error_code(error):
        if "[" in error and "]" in error:
            return error[error.rindex("[") + 1 : error.rindex("]")]
        return ""

    if any(
        error_type in error.lower()
        for error in errors
        for error_type in ["syntax", "empty_body", "name_defined"]
    ):
        return True

    for error in errors:
        error_code = extract_error_code(error)
        if error_code in non_type_related_errors:
            return True
    return False

def get_compilation_status(file_key, untyped_data, first_run_data, second_run_data):
    """
    Determine compilation status for a file across runs

    Returns:
    - 'not_in_first': File not present in 1st run
    - 'not_in_second': File not present in 2nd run
    - 'both_success': Compiled successfully in both runs
    - 'first_only': Compiled successfully only in 1st run
    - 'second_only': Compiled successfully only in 2nd run
    - 'both_fail': Failed compilation in both runs
    """

    # Check if file exists in each run
    in_first = file_key in first_run_data
    in_second = file_key in second_run_data

    if not in_first and not in_second:
        return "not_in_both"
    elif not in_first:
        return "not_in_first"
    elif not in_second:
        return "not_in_second"
    elif in_first and has_syntax_error(first_run_data[file_key].get("errors", [])):
        return "not_in_first"
    elif in_second and has_syntax_error(second_run_data[file_key].get("errors", [])):
        return "not_in_second"

    # Both runs have the file, check compilation status
    first_compiled = first_run_data[file_key].get("isCompiled", False)
    second_compiled = second_run_data[file_key].get("isCompiled", False)

    if first_compiled and second_compiled:
        return "both_success"
    elif first_compiled and not second_compiled:
        return "first_only"
    elif not first_compiled and second_compiled:
        return "second_only"
    else:  # both not compiled
        return "both_fail"


def create_compilation_consistency_plot():
    """Create the compilation consistency scatter plot"""

    # Load untyped results (our baseline)
    untyped_data = load_json_file(
        "../mypy_outputs/mypy_results_untyped_with_errors.json"
    )

    # Define models and their file paths in the specified order
    models = [
        (
            "gpt35",
            "../mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
            "../mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
        ),
        (
            "gpt4o",
            "../mypy_outputs/mypy_results_gpt4o_with_errors.json",
            "../mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json",
        ),
        (
            "o1-mini",
            "../mypy_outputs/mypy_results_o1_mini_with_errors.json",
            "../mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json",
        ),
        (
            "o3_mini",
            "../mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
            "../mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json",
        ),
        (
            "deepseek",
            "../mypy_outputs/mypy_results_deepseek_with_errors.json",
            "../mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json",
        ),
        (
            "claude3_sonnet",
            "../mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
            "../mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json",
        ),
    ]

    # Color mapping for different statuses
    color_map = {
        "not_in_first": "#888888",  # Gray (unprocessed)
        "not_in_second": "#888888",  # Gray (unprocessed)
        "not_in_both": "#888888",  # Gray (unprocessed)
        "both_success": "#44AA44",  # Green
        "first_only": "#FF6B35",  # Orange-red (more contrasting)
        "second_only": "#1E90FF",  # Dodger blue (more contrasting)
        "both_fail": "#FF4444",  # Red
    }

    # Status labels for legend in the specified order
    status_labels = {
        "both_success": "Both Success",
        "first_only": "1st only success",
        "second_only": "2nd only success",
        "both_fail": "Both fail",
        "not_in_first": "Unprocessed",
        "not_in_second": "Unprocessed",
        "not_in_both": "Unprocessed",
    }

    # Create figure
    plt.figure(figsize=(14, 10))

    # Define the order for legend items (5 main categories)
    legend_order = [
        "both_success",
        "first_only",
        "second_only",
        "both_fail",
        "not_in_first",  # represents all unprocessed
    ]
    used_statuses = set()

    # Define sub-belt positions for each model (5 sub-belts)
    sub_belt_positions = {
        "both_success": 0.0,
        "first_only": 0.2,
        "second_only": 0.4,
        "both_fail": 0.6,
        "not_in_first": 0.8,  # represents all unprocessed
    }

    # Process each model
    for y_pos, (model_name, first_run_file, second_run_file) in enumerate(models, 1):
        print(f"Processing {model_name}...")

        # Load model results
        first_run_data = load_json_file(first_run_file)
        second_run_data = load_json_file(second_run_file)

        # Filter files that are compiled in untyped results
        compiled_files = {
            file_key: file_data
            for file_key, file_data in untyped_data.items()
            if file_data.get("isCompiled", False)
        }

        # Group files by compilation status
        status_groups = {}
        for file_key in compiled_files.keys():
            status = get_compilation_status(
                file_key, untyped_data, first_run_data, second_run_data
            )
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(file_key)

        # Combine all unprocessed categories
        unprocessed_files = []
        for unprocessed_status in ["not_in_first", "not_in_second", "not_in_both"]:
            if unprocessed_status in status_groups:
                unprocessed_files.extend(status_groups[unprocessed_status])
        status_groups["not_in_first"] = (
            unprocessed_files  # Use this as the combined unprocessed
        )

        # Plot each status group in the specified order
        for status in legend_order:
            if status not in status_groups or not status_groups[status]:
                continue

            files = status_groups[status]

            # Get parameter counts for these files
            param_counts = []
            for file_key in files:
                param_count = untyped_data[file_key]["stats"]["total_parameters"]
                param_counts.append(param_count)

            # Calculate y-position for sub-belt
            sub_belt_y = y_pos - 0.4 + sub_belt_positions[status]
            y_positions = np.random.normal(sub_belt_y, 0.05, len(param_counts))

            # Plot the points
            plt.scatter(
                param_counts,
                y_positions,
                c=color_map[status],
                alpha=0.7,
                s=30,
                label=status_labels[status] if status not in used_statuses else "",
            )

            if status not in used_statuses:
                used_statuses.add(status)

        print(f"  - Total compiled files: {len(compiled_files)}")

        # Print results in organized groups
        print("  - Both fail:", status_groups.get("both_fail", []).__len__())
        print("  - Both unprocessed:", status_groups.get("not_in_both", []).__len__())
        print(
            "    - 1st only unprocessed:",
            status_groups.get("not_in_first", []).__len__(),
        )
        print(
            "    - 2nd only unprocessed:",
            status_groups.get("not_in_second", []).__len__(),
        )
        print()
        print("  - Both Success:", status_groups.get("both_success", []).__len__())
        print("    - 1st only success:", status_groups.get("first_only", []).__len__())
        print("    - 2nd only success:", status_groups.get("second_only", []).__len__())

    # Customize plot
    plt.xlabel("Number of Parameters", fontsize=12)
    plt.ylabel("Language Models", fontsize=12)
    plt.title(
        "Compilation Consistency Across LLM Runs\n(Only files that compiled in untyped version)",
        fontsize=14,
        pad=20,
    )

    # Set y-axis labels
    model_names = [model[0] for model in models]
    plt.yticks(range(1, len(models) + 1), model_names)

    # Add horizontal divider lines between models
    for i in range(1, len(models)):
        plt.axhline(y=i + 0.5, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Show plot
    #plt.show()
    plt.savefig("compilation_consistency_plot.pdf", bbox_inches="tight")
    print("Plot saved as compilation_consistency_plot.pdf")


if __name__ == "__main__":
    create_compilation_consistency_plot()
