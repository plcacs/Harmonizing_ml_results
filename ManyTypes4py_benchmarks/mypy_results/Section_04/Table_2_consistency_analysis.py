"""
Run Consistency Analysis Script

This script analyzes the consistency of type annotation quality between 1st and 2nd runs
of different language models on the same benchmark files.

What it produces:
- A table showing for each LLM model:
  * Number of files processed in both runs (excluding syntax errors)
  * Number of files successful in both runs
  * Number of files successful only in 1st run
  * Number of files successful only in 2nd run
  * Number of files that failed in both runs

How it differs from Table_1_analysis.py:
- Table_1_analysis.py compares LLM-generated code against untyped original code
- This script compares 1st run vs 2nd run results for the same LLM models
- Focuses on consistency/reproducibility rather than absolute performance
- Only analyzes models that have both 1st and 2nd run data available
- Excludes files with syntax errors from both runs for fair comparison
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


def analyze_run_consistency(model_name, first_run_file, second_run_file):
    first_run_files = load_json_file(first_run_file)
    second_run_files = load_json_file(second_run_file)

    # Find syntax error files for both runs
    first_run_syntax_errors = set()
    second_run_syntax_errors = set()
    
    for file_key, file_data in first_run_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            first_run_syntax_errors.add(file_key)
    
    for file_key, file_data in second_run_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            second_run_syntax_errors.add(file_key)

    # Get common files that are processed in both runs
    common_files = set(first_run_files.keys()) & set(second_run_files.keys())
    processed_in_both = common_files - first_run_syntax_errors - second_run_syntax_errors

    # Categorize files
    both_success = []
    first_success_only = []
    second_success_only = []
    both_failure = []

    for file_key in processed_in_both:
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

    return {
        "model": model_name,
        "total_common_files": len(common_files),
        "processed_in_both": len(processed_in_both),
        "both_success": len(both_success),
        "first_success_only": len(first_success_only),
        "second_success_only": len(second_success_only),
        "both_failure": len(both_failure),
        "consistency_rate": (len(both_success) / len(processed_in_both) * 100) if len(processed_in_both) > 0 else 0
    }


if __name__ == "__main__":
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
        results = analyze_run_consistency(model_name, first_run_file, second_run_file)
        all_results.append(results)

    # Print summary table
    print("Model       Processed in both runs     Both Success 1st Only   2nd Only   Both Fail")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model']:<12} {result['processed_in_both']:<25} {result['both_success']:<12} {result['first_success_only']:<10} {result['second_success_only']:<10} {result['both_failure']}")
