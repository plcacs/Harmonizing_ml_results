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


def analyze_model_group(model_name, untyped_file, model_file, file_filter, group_name):
    untyped_files = load_json_file(untyped_file)
    model_files = load_json_file(model_file)

    # Filter to only model files in this group
    model_files_filtered = {k: v for k, v in model_files.items() if k in file_filter}
    untyped_files_filtered = {
        k: v for k, v in untyped_files.items() if k in file_filter
    }

    # Find syntax error files
    syntax_error_files = set()
    for file_key, file_data in model_files_filtered.items():
        if has_syntax_error(file_data.get("errors", [])):
            syntax_error_files.add(file_key)

    # Categorize processed files
    both_error_files = []
    llm_only_failure_files = []
    both_success_files = []

    for file_key in model_files_filtered:
        if file_key in syntax_error_files:
            continue  # Skip syntax error files

        untyped_error_count = untyped_files_filtered[file_key]["error_count"]
        model_error_count = model_files_filtered[file_key]["error_count"]

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
        "processed_by_llm": len(model_files_filtered) - len(syntax_error_files),
        "unprocessed": len(syntax_error_files),
        "both_errors": len(both_error_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failure_files),
        "llm_success_rate": llm_success_rate,
    }


def analyze_model(
    model_name, untyped_file, model_file, no_param_files, with_param_files
):
    # Calculate total preprocessed files
    model_files = load_json_file(model_file)
    total_preprocessed_error_files = (
        set(no_param_files) | set(with_param_files)
    ) - set(model_files)

    # Analyze each group
    results_no_param = analyze_model_group(
        model_name,
        untyped_file,
        model_file,
        set(no_param_files.keys()),
        "No Parameter Annotations",
    )

    results_with_param = analyze_model_group(
        model_name,
        untyped_file,
        model_file,
        set(with_param_files.keys()),
        "With Parameter Annotations",
    )

    total_preprocessed_error_files_count = len(total_preprocessed_error_files)

    total_preprocessed_error_files_count_no_param = 0
    for file in total_preprocessed_error_files:
        if file in no_param_files:
            total_preprocessed_error_files_count_no_param += 1
    total_preprocessed_error_files_count_with_param = 0
    for file in total_preprocessed_error_files:
        if file in with_param_files:
            total_preprocessed_error_files_count_with_param += 1

    return {
        "model": model_name,
        "total_preprocessed_files": total_preprocessed_error_files_count,
        "no_parameter_annotations": results_no_param,
        "with_parameter_annotations": results_with_param,
        "total_preprocessed_error_files_count_no_param": total_preprocessed_error_files_count_no_param,
        "total_preprocessed_error_files_count_with_param": total_preprocessed_error_files_count_with_param,
    }


if __name__ == "__main__":
    # Load split files
    no_param_files = load_json_file(
        "split_original_files/files_with_no_parameter_annotations.json"
    )
    with_param_files = load_json_file(
        "split_original_files/files_with_parameter_annotations.json"
    )

    # Models to analyze

    models = [
        ("Human", "mypy_outputs/mypy_results_original_files_with_errors.json"),
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
        ("gpt4o_2nd_run", "mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json"),
        (
            "o1-mini_2nd_run",
            "mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json",
        ),
        (
            "deepseek_2nd_run",
            "mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json",
        ),
        ("gpt35_1st_run", "mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json"),
        ("HiTyper_1st_run", "mypy_outputs/mypy_results_HiTyper_1st_run_with_errors.json"),
    ]

    all_results = []
    for model_name, model_file in models:
        results = analyze_model(
            model_name,
            "mypy_outputs/mypy_results_untyped_with_errors.json",
            model_file,
            no_param_files,
            with_param_files,
        )
        all_results.append(results)

    # Print summary in the requested format
    for result in all_results:
        model = result["model"]
        total_preprocessed = result["total_preprocessed_files"]

        no_param = result["no_parameter_annotations"]
        with_param = result["with_parameter_annotations"]
        total_preprocessed_error_files_count_no_param = result[
            "total_preprocessed_error_files_count_no_param"
        ]
        total_preprocessed_error_files_count_with_param = result[
            "total_preprocessed_error_files_count_with_param"
        ]
        print(model)
        # print("Total preprocessed error files:", total_preprocessed)
        print("unprocessed, both_fail, both_success, llm_only_fail, LLM_success_rate")
        print(
            f"{no_param['unprocessed']+total_preprocessed_error_files_count_no_param}, {no_param['both_errors']}, {no_param['both_success']}, {no_param['llm_only_failures']}, {no_param['llm_success_rate']:.2f}%"
        )
        print(
            f"{with_param['unprocessed']+total_preprocessed_error_files_count_with_param}, {with_param['both_errors']}, {with_param['both_success']}, {with_param['llm_only_failures']}, {with_param['llm_success_rate']:.2f}%"
        )
