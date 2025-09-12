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

    # File  paths
    file1_paths = [
        "../mypy_outputs/mypy_results_gpt35_renamed_output_with_errors.json",
        "../mypy_outputs/mypy_results_o1_mini_renamed_output_with_errors.json",
        "../mypy_outputs/mypy_results_deepseek_renamed_output_2_with_errors.json",
        "../mypy_outputs/mypy_results_claude_sonnet_renamed_output_with_errors.json"
    ]
    file2_paths = [
        "../mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
        "../mypy_outputs/mypy_results_o1_mini_with_errors.json",
        "../mypy_outputs/mypy_results_deepseek_with_errors.json",
        "../mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        
    ]
    file3_path = "../mypy_outputs/mypy_results_untyped_with_errors.json"

    # Load file3 data for common files reference
    data3 = load_json_file(file3_path)
    if not data3:
        print("Failed to load file3.")
        return

    # Collect all results first
    all_results = []

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
        if "gpt35" in file1_path:
            model_name = "gpt-3.5"
        elif "o1_mini" in file1_path:
            model_name = "o1-mini"
        elif "deepseek" in file1_path:
            model_name = "deepseek"
        elif "claude" in file1_path:
            model_name = "claude3-sonnet"
        else:
            model_name = "unknown"

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

        # Store results for consolidated table
        all_results.append({
            'model': model_name,
            'total_files': len(common_files),
            'both_compiled': truth_table['file1_true_file2_true'],
            'renamed_only': truth_table['file1_true_file2_false'],
            'original_only': truth_table['file1_false_file2_true'],
            'neither_compiled': truth_table['file1_false_file2_false']
        })

    # Print consolidated table
    print("CONSOLIDATED TRUTH TABLE - RENAMED vs ORIGINAL FILES")
    print("=" * 80)
    print(f"{'Model':<15} {'Total':<8} {'Both':<8} {'Renamed':<10} {'Original':<10} {'Neither':<8}")
    print(f"{'':<15} {'Files':<8} {'Compiled':<8} {'Only':<10} {'Only':<10} {'Compiled':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model']:<15} {result['total_files']:<8} {result['both_compiled']:<8} "
              f"{result['renamed_only']:<10} {result['original_only']:<10} {result['neither_compiled']:<8}")
    
    print("\nPERCENTAGES:")
    print("-" * 80)
    print(f"{'Model':<15} {'Both':<8} {'Renamed':<10} {'Original':<10} {'Neither':<8}")
    print(f"{'':<15} {'Compiled':<8} {'Only':<10} {'Only':<10} {'Compiled':<8}")
    print("-" * 80)
    
    for result in all_results:
        total = result['total_files']
        print(f"{result['model']:<15} {result['both_compiled']/total*100:<7.1f}% "
              f"{result['renamed_only']/total*100:<9.1f}% {result['original_only']/total*100:<9.1f}% "
              f"{result['neither_compiled']/total*100:<7.1f}%")


if __name__ == "__main__":
    analyze_compilation_status()
