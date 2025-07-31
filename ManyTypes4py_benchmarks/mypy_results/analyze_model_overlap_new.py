#!/usr/bin/env python3
"""
Script to analyze overlap between model files and split JSON files.
Analyzes how many files from a specific model are common in each of the split JSON files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Set, List


def load_json_file(file_path: str) -> Dict:
    """Load JSON file and return the data."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def get_file_names_from_json(data: Dict) -> Set[str]:
    """Extract file names from JSON data."""
    if "files" in data and "llm_only_failures" in data["files"]:
        llm_failures = data["files"]["llm_only_failures"]
        if isinstance(llm_failures, list):
            return set(llm_failures)
        elif isinstance(llm_failures, dict):
            return set(llm_failures.keys())
    return set(data.keys())


def analyze_overlap(
    model_files: Set[str],
    no_annotations_files: Set[str],
    with_annotations_files: Set[str],
    model_name: str,
) -> Dict[str, any]:
    """
    Analyze overlap between model files and the two split categories.

    Args:
        model_files: Set of file names from the model
        no_annotations_files: Set of file names with no parameter annotations
        with_annotations_files: Set of file names with parameter annotations
        model_name: Name of the model being analyzed

    Returns:
        Dictionary containing overlap analysis results
    """

    results = {
        "model_name": model_name,
        "model_total_files": len(model_files),
        "no_annotations_total": len(no_annotations_files),
        "with_annotations_total": len(with_annotations_files),
        "model_in_no_annotations": len(model_files & no_annotations_files),
        "model_in_with_annotations": len(model_files & with_annotations_files),
        "model_in_both": len(
            model_files & no_annotations_files & with_annotations_files
        ),
        "model_in_neither": len(
            model_files - (no_annotations_files | with_annotations_files)
        ),
        "files_model_in_no_annotations": list(model_files & no_annotations_files),
        "files_model_in_with_annotations": list(model_files & with_annotations_files),
        "files_model_in_both": list(
            model_files & no_annotations_files & with_annotations_files
        ),
        "files_model_in_neither": list(
            model_files - (no_annotations_files | with_annotations_files)
        ),
    }

    return results


def print_analysis_results(results: Dict[str, any]):
    """Print the analysis results in a formatted way."""

    model_name = results["model_name"]

    print(f"{model_name.upper()}:")
    print(f"  Total files: {results['model_total_files']:,}")
    print(f"  In NO annotations: {results['model_in_no_annotations']:,}")
    print(f"  In WITH annotations: {results['model_in_with_annotations']:,}")
    print(f"  In BOTH: {results['model_in_both']:,}")
    print(f"  In NEITHER: {results['model_in_neither']:,}")

    # Debug: Check if the numbers add up correctly
    total_accounted = (
        results["model_in_no_annotations"]
        + results["model_in_with_annotations"]
        - results["model_in_both"]
    )
    print(f"  Total accounted for: {total_accounted:,}")
    print(f"  Should be: {results['model_total_files']:,}")


def save_detailed_results(results: Dict[str, any], model_name: str):
    """Save detailed results to JSON file."""
    output_file = f"{model_name}_overlap_analysis.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main function to execute the analysis."""

    # File paths
    model_files = [
        "analysis_outputs/analysis_claude_3_5_sonnet_simplified.json",
        "analysis_outputs/analysis_deepseek_simplified.json",
        "analysis_outputs/analysis_o3_mini_1st_run_simplified.json",
        "analysis_outputs/analysis_o1-mini_simplified.json",
        "analysis_outputs/analysis_gpt4o_simplified.json",
    ]
    no_annotations_file = (
        "split_original_files/files_with_no_parameter_annotations.json"
    )
    with_annotations_file = "split_original_files/files_with_parameter_annotations.json"

    # Check if required files exist
    required_files = [no_annotations_file, with_annotations_file] + model_files
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return

    # Load annotation files once
    no_annotations_data = load_json_file(no_annotations_file)
    with_annotations_data = load_json_file(with_annotations_file)
    print(f"No annotations data: {len(no_annotations_data)}")
    print(f"With annotations data: {len(with_annotations_data)}")

    # Extract file names for annotations
    no_annotations_files = get_file_names_from_json(no_annotations_data)
    with_annotations_files = get_file_names_from_json(with_annotations_data)

    # Process each model
    for model_file in model_files:
        model_name = model_file.replace(
            "ManyTypes4py_benchmarks/mypy_results/analysis_outputs/analysis_", ""
        ).replace("_simplified.json", "")

        # Load model data
        model_data = load_json_file(model_file)
        model_files_set = get_file_names_from_json(model_data)
        count_no_annotations = 0
        count_with_annotations = 0
        count_neither = 0
        for file in model_files_set:
            if file in no_annotations_files:
                count_no_annotations += 1
            elif file in with_annotations_files:
                count_with_annotations += 1
            else:
                count_neither += 1
        total_files = count_no_annotations + count_with_annotations + count_neither
        no_annotations_pct = (
            (count_no_annotations / total_files * 100) if total_files > 0 else 0
        )
        with_annotations_pct = (
            (count_with_annotations / total_files * 100) if total_files > 0 else 0
        )
        neither_pct = (count_neither / total_files * 100) if total_files > 0 else 0
        no_annotations_pct_in_no_annotations = (
            (count_no_annotations / len(no_annotations_files) * 100)
            if len(no_annotations_files) > 0
            else 0
        )
        with_annotations_pct_in_with_annotations = (
            (count_with_annotations / len(with_annotations_files) * 100)
            if len(with_annotations_files) > 0
            else 0
        )

        print(f"Model: {model_name}")
        print(
            f"  Counts - No annotations: {count_no_annotations}, With annotations: {count_with_annotations}, Neither: {count_neither}"
        )
        print(
            f"  Percentages - No annotations: {no_annotations_pct:.1f}%, With annotations: {with_annotations_pct:.1f}%, Neither: {neither_pct:.1f}%"
        )
        print(
            f"  % of total no_annotations files: {no_annotations_pct_in_no_annotations:.1f}%"
        )
        print(
            f"  % of total with_annotations files: {with_annotations_pct_in_with_annotations:.1f}%"
        )


if __name__ == "__main__":
    main()
