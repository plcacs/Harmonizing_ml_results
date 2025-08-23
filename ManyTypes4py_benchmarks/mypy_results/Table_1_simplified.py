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

    # Check for original syntax error patterns
    if any(
        error_type in error.lower()
        for error in errors
        for error_type in ["syntax", "empty_body", "name_defined"]
    ):
        return True

    # Check for non_type_related_errors
    for error in errors:
        error_code = extract_error_code(error)
        if error_code in non_type_related_errors:
            return True
    return False


def analyze_files(model_name, untyped_file, model_file):
    print(f"Analyzing {model_name} with {untyped_file} and {model_file}")

    # Load JSON files
    untyped_files = load_json_file(untyped_file)
    model_files = load_json_file(model_file)

    # Step 1: Calculate preprocessed error files
    Number_of_preprocessed_error_files = set(untyped_files.keys()) - set(
        model_files.keys()
    )
    print(
        f"Number of preprocessed error files: {len(Number_of_preprocessed_error_files)}"
    )

    # Step 2: Find syntax error files in model
    syntax_error_model_files = set()
    for file_key, file_data in model_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            syntax_error_model_files.add(file_key)
    print(f"Syntax error files in model: {len(syntax_error_model_files)}")

    # Step 3: Calculate total unprocessed files (no double entry)
    total_unprocessed_files = Number_of_preprocessed_error_files.union(
        syntax_error_model_files
    )
    print(f"Total unprocessed files: {len(total_unprocessed_files)}")

    # Step 4: Calculate processed files
    processed_files = set(untyped_files.keys()) - total_unprocessed_files
    print(f"Processed files: {len(processed_files)}")

    # Step 5: Categorize processed files
    both_error_files = []
    llm_only_failure_files = []
    both_success_files = []

    for file_key in processed_files:
        untyped_error_count = untyped_files[file_key]["error_count"]
        model_error_count = model_files[file_key]["error_count"]

        if untyped_error_count > 0:
            both_error_files.append(file_key)
        elif untyped_error_count == 0 and model_error_count > 0:
            llm_only_failure_files.append(file_key)
        elif untyped_error_count == 0 and model_error_count == 0:
            both_success_files.append(file_key)

    # Step 6: Verify total count
    total_count = (
        len(llm_only_failure_files)
        + len(both_error_files)
        + len(total_unprocessed_files)
        + len(both_success_files)
    )

    print(f"\nVerification:")
    print(f"LLM only failures: {len(llm_only_failure_files)}")
    print(f"Both error files: {len(both_error_files)}")
    print(f"Total unprocessed error files: {len(total_unprocessed_files)}")
    print(f"Both success files: {len(both_success_files)}")
    print(f"Total: {total_count}")
    print(f"Original untyped files: {len(untyped_files)}")
    print(f"Counts match: {total_count == len(untyped_files)}")

    # Calculate percentages
    llm_success_rate = (
        100
        * (
            1
            - (
                len(llm_only_failure_files)
                / (len(llm_only_failure_files) + len(both_success_files))
            )
        )
        if (len(llm_only_failure_files) + len(both_success_files)) > 0
        else 0
    )

    # Print results table
    print(f"\nResults for {model_name}:")
    print(
        "| Processed by LLM | Unprocessed | Both Errors | Both Success | LLM-Only Failures | LLM Success Rate |"
    )
    print("|---|---|---|---|---|---|")
    print(
        f"| {len(processed_files)} | {len(total_unprocessed_files)} | {len(both_error_files)} | {len(both_success_files)} | {len(llm_only_failure_files)} | {llm_success_rate:.2f}% |"
    )

    # Save results to JSON
    results = {
        "processed_by_llm": len(processed_files),
        "unprocessed": len(total_unprocessed_files),
        "both_errors": len(both_error_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failure_files),
        "llm_success_rate": llm_success_rate,
        "files": {
            "both_errors": list(both_error_files),
            "both_success": list(both_success_files),
            "llm_only_failures": list(llm_only_failure_files),
            "preprocessed_errors": list(Number_of_preprocessed_error_files),
            "syntax_errors": list(syntax_error_model_files),
        },
    }

    with open(f"analysis_{model_name}_simplified.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    """analyze_files(
        "original_files",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_original_files_with_errors.json",
    )
    print("="*100)
    analyze_files(
        "gpt4o",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_gpt4o_with_errors.json",
    )
    print("="*100)
    analyze_files(
        "o1-mini",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_o1_mini_with_errors.json",
    )
    print("="*100)
    analyze_files(
        "deepseek",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_deepseek_with_errors.json",
    )
    print("="*100)
    analyze_files(
        "o3_mini_1st_run",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_o3_mini_1st_run_with_errors.json",
    )"""
    print("=" * 100)
    analyze_files(
        "claude_3_7_sonnet",
        "mypy_outputs/mypy_results_untyped_with_errors.json",
        "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    )
