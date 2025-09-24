"""
Run Consistency Analysis Script

This script analyzes the consistency of type annotation quality between 1st and 2nd runs
of different language models on baseline files (untyped files that successfully pass type checking).

What it produces:
- A table showing for each LLM model:
  * Number of unprocessed baseline files (not processed in either run)
  * Number of files successful in both runs
  * Number of files successful only in 1st run
  * Number of files successful only in 2nd run
  * Number of files that failed in both runs
  * Instability rate (percentage of processed files with inconsistent results between runs)

Key features:
- All calculations are based on baseline files (untyped files that successfully pass mypy type checking)
- Analyzes all baseline files, regardless of whether they appear in both runs
- Shows distribution of baseline files across different processing outcomes
- Excludes files with syntax errors from analysis for fair comparison
- Provides comprehensive view of how baseline files are handled across runs
"""

import json


def load_json_file(filename):
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


def analyze_run_consistency(model_name, first_run_file, second_run_file, baseline_file):
    first_run_files = load_json_file(first_run_file)
    second_run_files = load_json_file(second_run_file)
    baseline_files = load_json_file(baseline_file)

    # Get baseline files that successfully pass type checking (untyped successful files)
    baseline_successful = set()
    for file_key, file_data in baseline_files.items():
        if file_data.get("error_count", 0) == 0 and file_data.get("isCompiled", False):
            baseline_successful.add(file_key)

    # Find syntax error files for both runs
    first_run_syntax_errors = set()
    second_run_syntax_errors = set()
    
    for file_key, file_data in first_run_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            first_run_syntax_errors.add(file_key)
    
    for file_key, file_data in second_run_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            second_run_syntax_errors.add(file_key)

    # Get files that are in baseline and don't have syntax errors in either run
    first_run_valid = set(first_run_files.keys()) & baseline_successful - first_run_syntax_errors
    second_run_valid = set(second_run_files.keys()) & baseline_successful - second_run_syntax_errors

    # Categorize baseline files
    unprocessed = []
    both_success = []
    first_success_only = []
    second_success_only = []
    both_failure = []

    for file_key in baseline_successful:
        if file_key not in first_run_valid and file_key not in second_run_valid:
            # File not processed in either run
            unprocessed.append(file_key)
        elif file_key in first_run_valid and file_key in second_run_valid:
            # File processed in both runs
            first_error_count = first_run_files[file_key]["error_count"]
            second_error_count = second_run_files[file_key]["error_count"]

            if first_error_count == 0 and second_error_count == 0:
                both_success.append(file_key)
            elif first_error_count == 0 and second_error_count > 0:
                first_success_only.append(file_key)
            elif first_error_count > 0 and second_error_count == 0:
                second_success_only.append(file_key)
            else:  # both have errors
                both_failure.append(file_key)
        elif file_key in first_run_valid:
            # File only in first run
            first_error_count = first_run_files[file_key]["error_count"]
            if first_error_count == 0:
                first_success_only.append(file_key)
            else:
                both_failure.append(file_key)
        elif file_key in second_run_valid:
            # File only in second run
            second_error_count = second_run_files[file_key]["error_count"]
            if second_error_count == 0:
                second_success_only.append(file_key)
            else:
                both_failure.append(file_key)

    # Calculate instability rate (files that are successful in one run but fail in the other)
    inconsistent_files = len(first_success_only) + len(second_success_only)
    total_processed = len(both_success) + inconsistent_files + len(both_failure)
    instability_rate = (inconsistent_files / total_processed * 100) if total_processed > 0 else 0

    return {
        "model": model_name,
        "total_baseline_files": len(baseline_successful),
        "unprocessed": len(unprocessed),
        "both_success": len(both_success),
        "first_success_only": len(first_success_only),
        "second_success_only": len(second_success_only),
        "both_failure": len(both_failure),
        "instability_rate": instability_rate
    }


if __name__ == "__main__":
    # Baseline file (untyped successful files)
    baseline_file = "mypy_outputs/mypy_results_untyped_with_errors.json"
    
    # Models with both 1st and 2nd runs
    models = [
        ("gpt4o", 
         "mypy_outputs/mypy_results_gpt4o_with_errors.json",
         "mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json"),
        ("o1-mini", 
         "mypy_outputs/mypy_results_o1_mini_with_errors.json",
         "mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json"),
        ("deepseek", 
         "mypy_outputs/mypy_results_deepseek_with_errors.json",
         "mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json"),
        ("claude3_sonnet", 
         "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
         "mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json"),
        ("gpt35", 
         "mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
         "mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json"),
        ("o3_mini", 
         "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
         "mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json"),
    ]

    all_results = []
    for model_name, first_run_file, second_run_file in models:
        results = analyze_run_consistency(model_name, first_run_file, second_run_file, baseline_file)
        all_results.append(results)

    # Print summary table
    print("Model       Unprocessed    Both Success 1st Only   2nd Only   Both Fail  Instability Rate")
    print("-" * 85)
    
    for result in all_results:
        print(f"{result['model']:<12} {result['unprocessed']:<13} {result['both_success']:<12} {result['first_success_only']:<10} {result['second_success_only']:<10} {result['both_failure']:<10} {result['instability_rate']:.1f}%")
