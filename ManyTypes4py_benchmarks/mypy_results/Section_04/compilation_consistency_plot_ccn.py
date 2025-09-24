"""
Compilation Consistency Visualization with CCN (Cyclomatic Complexity Number)

Creates a scatter plot showing compilation consistency across LLM runs:
- X-axis: Sum of top 3 functions CCN from complexity analysis
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
import os
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


def create_compilation_consistency_plot_ccn():
    """Create the compilation consistency scatter plot using CCN data"""

    # Load untyped results (our baseline)
    untyped_data = load_json_file(
        "../mypy_outputs/mypy_results_untyped_with_errors.json"
    )

    # Load complexity analysis data
    complexity_data = load_json_file(
        "../../complexity_of_source_codes/untyped_benchmarks_complexity_analysis.json"
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

    # We will create six individual figures (one per model)

    # Prepare output directory for PDFs
    output_dir = os.path.join(os.path.dirname(__file__), "ccn_bar_plots_pdf")
    os.makedirs(output_dir, exist_ok=True)

    # Categories to include (five total)
    legend_order = [
        "both_success",
        "first_only",
        "second_only",
        "both_fail",
        "not_in_first",  # represents all unprocessed combined
    ]

    # Prepare global CCN bins based on all compiled baseline files
    # Gather CCN sums for all baseline compiled files
    all_baseline_ccn = []
    for file_key, file_data in load_json_file(
        "../mypy_outputs/mypy_results_untyped_with_errors.json"
    ).items():
        if file_data.get("isCompiled", False):
            file_name = file_key.split("/")[-1] if "/" in file_key else file_key
            if file_name in complexity_data:
                top_3_ccn = complexity_data[file_name]["top_3_functions_CCN"]
                all_baseline_ccn.append(sum(top_3_ccn))
            else:
                all_baseline_ccn.append(0)

    if len(all_baseline_ccn) == 0:
        # Fallback to a simple range to avoid errors
        bin_edges = np.linspace(0, 1, 6)
    else:
        # Use 10 bins across the observed range
        min_val, max_val = float(min(all_baseline_ccn)), float(max(all_baseline_ccn))
        if min_val == max_val:
            # Avoid zero-width bins
            min_val, max_val = 0.0, max_val + 1.0
        bin_edges = np.linspace(min_val, max_val, 11)

    # Precompute global baseline totals per bin for consistent labeling and normalization
    baseline_counts_per_bin, _ = np.histogram(all_baseline_ccn, bins=bin_edges)

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

        # For each status, compute CCN sums for histogram
        status_to_ccn = {status: [] for status in legend_order}

        # Combine unprocessed categories into 'not_in_first'
        unprocessed_files = []
        for unprocessed_status in ["not_in_first", "not_in_second", "not_in_both"]:
            if unprocessed_status in status_groups:
                unprocessed_files.extend(status_groups[unprocessed_status])
        status_groups["not_in_first"] = unprocessed_files

        for status in legend_order:
            files = status_groups.get(status, [])
            for file_key in files:
                file_name = file_key.split("/")[-1] if "/" in file_key else file_key
                if file_name in complexity_data:
                    top_3_ccn = complexity_data[file_name]["top_3_functions_CCN"]
                    status_to_ccn[status].append(sum(top_3_ccn))
                else:
                    status_to_ccn[status].append(0)

        # Build grouped bars per CCN bin for this model in its own figure
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        # Use a fraction of bin width for grouped bars
        group_bar_width = 0.18

        # For each status, compute histogram counts then convert to percentages per bin
        counts_by_status = {}
        for status in legend_order:
            counts, _ = np.histogram(status_to_ccn[status], bins=bin_edges)
            counts_by_status[status] = counts

        # Use global baseline totals per bin for normalization to ensure consistency across figures
        totals_per_bin = baseline_counts_per_bin

        percentages_by_status = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for status in legend_order:
                percentages = np.where(
                    totals_per_bin > 0,
                    (counts_by_status[status] / totals_per_bin) * 100.0,
                    0.0,
                )
                percentages_by_status[status] = percentages

        # Plot grouped bars with slight offsets
        num_statuses = len(legend_order)
        for idx, status in enumerate(legend_order):
            offset = (idx - (num_statuses - 1) / 2.0) * (group_bar_width * bin_widths)
            ax.bar(
                bin_centers + offset,
                percentages_by_status[status],
                width=group_bar_width * bin_widths,
                color=color_map[status],
                alpha=0.85,
                label=status_labels[status],
                align="center",
            )

        ax.set_title(model_name)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Percentage of Files (%)")
        ax.set_xlabel("Sum of Top 3 Functions CCN")
        ax.set_ylim(0, 100)
        # Label x-axis bins with total file counts per bin (use global baseline totals)
        xtick_labels = []
        for center, total in zip(bin_centers, baseline_counts_per_bin):
            label_ccn = int(round(center))
            xtick_labels.append(f"CCN {label_ccn} ({int(total)} files)")
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save figure as PDF in the output directory
        safe_model_name = model_name.replace("/", "-")
        pdf_path = os.path.join(output_dir, f"compilation_consistency_{safe_model_name}.pdf")
        plt.savefig(pdf_path, bbox_inches="tight")

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

    # Show all figures (one per model)
    
    plt.show()

    # Show plot (do not save)
    plt.show()


if __name__ == "__main__":
    create_compilation_consistency_plot_ccn()
