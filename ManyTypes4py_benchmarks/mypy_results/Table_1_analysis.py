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


def analyze_model(model_name, untyped_file, model_file):
    untyped_files = load_json_file(untyped_file)
    model_files = load_json_file(model_file)

    # Find syntax error files
    syntax_error_files = set()
    for file_key, file_data in model_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            syntax_error_files.add(file_key)

    # Categorize processed files
    both_error_files = []
    llm_only_failure_files = []
    both_success_files = []

    for file_key in model_files:
        if file_key in syntax_error_files:
            continue  # Skip syntax error files

        untyped_error_count = untyped_files[file_key]["error_count"]
        model_error_count = model_files[file_key]["error_count"]

        if untyped_error_count > 0:
            both_error_files.append(file_key)
        elif untyped_error_count == 0 and model_error_count > 0:
            llm_only_failure_files.append(file_key)
        elif untyped_error_count == 0 and model_error_count == 0:
            both_success_files.append(file_key)

    # Calculate success rate
    total_processed = len(llm_only_failure_files) + len(both_success_files)
    llm_success_rate = (
        100 * (1 - (len(llm_only_failure_files) / total_processed))
        if total_processed > 0
        else 0
    )

    return {
        "model": model_name,
        "processed_by_llm": len(model_files) - len(syntax_error_files),
        "unprocessed": len(syntax_error_files),
        "both_errors": len(both_error_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failure_files),
        "llm_success_rate": llm_success_rate,
    }


if __name__ == "__main__":
    # Models to analyze
    models = [
        # ("Human", "mypy_outputs/mypy_results_original_files_with_errors.json"),
        (
            "o3_mini_1st_run",
            "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        ),
        (
            "claude_3_7_sonnet",
            "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        ),
        ("gpt4o", "mypy_outputs/mypy_results_gpt4o_with_errors.json"),
        ("o1-mini", "mypy_outputs/mypy_results_o1_mini_with_errors.json"),
        ("deepseek", "mypy_outputs/mypy_results_deepseek_with_errors.json"),
        ("gpt35_1st_run", "mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json"),
        ("gpt4o_2nd_run", "mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json"),
        (
            "o1-mini_2nd_run",
            "mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json",
        ),
        (
            "deepseek_2nd_run",
            "mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json",
        ),
        (
            "claude3_sonnet_2nd_run",
            "mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json",
        ),
        ("gpt35_2nd_run", "mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json"),
        (
            "o3_mini_2nd_run",
            "mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json",
        ),
        ("gpt5_1st_run", "mypy_outputs/mypy_results_gpt5_1st_run_with_errors.json"),
    ]

    all_results = []
    for model_name, model_file in models:
        results = analyze_model(
            model_name,
            "mypy_outputs/mypy_results_untyped_with_errors.json",
            model_file,
        )
        all_results.append(results)

    # Print summary in the requested format
    for result in all_results:
        model = result["model"]
        print(model)
        print("unprocessed, both_fail, both_success, llm_only_fail, LLM_success_rate")
        print(
            f"{result['unprocessed']}, {result['both_errors']}, {result['both_success']}, {result['llm_only_failures']}, {result['llm_success_rate']:.2f}%"
        )
