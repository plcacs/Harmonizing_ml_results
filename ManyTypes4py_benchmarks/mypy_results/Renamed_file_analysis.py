import json
import os
from collections import defaultdict


def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None


def analyze_compilation_status():
    """Analyze compilation status of common files between the three JSON files."""

    # File paths
    file1_paths = [
        "mypy_outputs/renamed_output/mypy_results_o1_mini_renamed_output_2_with_errors.json",
        "mypy_outputs/renamed_output/mypy_results_deepseek_renamed_output_2_with_errors.json",
    ]
    file2_paths = [
        "mypy_outputs/mypy_results_o1_mini_with_errors.json",
        "mypy_outputs/mypy_results_deepseek_with_errors.json",
    ]
    file3_path = "mypy_outputs/mypy_results_untyped_with_errors.json"

    # Load file3 data for common files reference
    data3 = load_json_file(file3_path)
    if not data3:
        print("Failed to load file3.")
        return

    # Create truth tables for each pair
    for i, (file1_path, file2_path) in enumerate(zip(file1_paths, file2_paths)):
        # Load individual pair data
        data1 = load_json_file(file1_path)
        data2 = load_json_file(file2_path)

        if not data1 or not data2:
            print(f"Failed to load files for pair {i+1}")
            continue

        # Get file names from each dataset
        files1 = set(data1.keys())
        files2 = set(data2.keys())
        files3 = set(data3.keys())

        # Find common files (files that exist in all three datasets)
        common_files = files1.intersection(files2).intersection(files3)

        if not common_files:
            print(f"No common files found for pair {i+1}")
            continue

        # Determine model name from file path
        model_name = "o1-mini" if "o1_mini" in file1_path else "deepseek"

        # Truth table analysis between File1 and File2
        truth_table = {
            "file1_true_file2_true": 0,  # Both compiled
            "file1_true_file2_false": 0,  # File1 compiled, File2 not
            "file1_false_file2_true": 0,  # File1 not compiled, File2 compiled
            "file1_false_file2_false": 0,  # Both not compiled
        }

        for filename in sorted(common_files):
            is_compiled_1 = data1[filename].get("isCompiled", False)
            is_compiled_2 = data2[filename].get("isCompiled", False)

            if is_compiled_1 and is_compiled_2:
                truth_table["file1_true_file2_true"] += 1
            elif is_compiled_1 and not is_compiled_2:
                truth_table["file1_true_file2_false"] += 1
            elif not is_compiled_1 and is_compiled_2:
                truth_table["file1_false_file2_true"] += 1
            else:  # not is_compiled_1 and not is_compiled_2
                truth_table["file1_false_file2_false"] += 1

        # Print truth table for this model
        print(f"TRUTH TABLE - {model_name.upper()}:")
        print(
            f"{'File1 Compiled':<15} {'File2 Compiled':<15} {'Count':<10} {'Percentage':<12}"
        )
        print("-" * 55)
        print(
            f"{'True':<15} {'True':<15} {truth_table['file1_true_file2_true']:<10} {truth_table['file1_true_file2_true']/len(common_files)*100:.1f}%"
        )
        print(
            f"{'True':<15} {'False':<15} {truth_table['file1_true_file2_false']:<10} {truth_table['file1_true_file2_false']/len(common_files)*100:.1f}%"
        )
        print(
            f"{'False':<15} {'True':<15} {truth_table['file1_false_file2_true']:<10} {truth_table['file1_false_file2_true']/len(common_files)*100:.1f}%"
        )
        print(
            f"{'False':<15} {'False':<15} {truth_table['file1_false_file2_false']:<10} {truth_table['file1_false_file2_false']/len(common_files)*100:.1f}%"
        )
        print()


if __name__ == "__main__":
    analyze_compilation_status()
