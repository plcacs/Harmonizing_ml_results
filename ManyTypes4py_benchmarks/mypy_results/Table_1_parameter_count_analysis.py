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


def split_files_by_parameter_count(json_data):
    """Split files into two groups based on parameter count (threshold: 50)"""
    files_with_less_than_50_parameters = {}
    files_with_more_than_50_parameters = {}
    
    for file_key, file_data in json_data.items():
        total_parameters = file_data.get("stats", {}).get("total_parameters", 0)
        
        if total_parameters < 50:
            files_with_less_than_50_parameters[file_key] = file_data
        else:
            files_with_more_than_50_parameters[file_key] = file_data
    
    return files_with_less_than_50_parameters, files_with_more_than_50_parameters


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
    model_name, untyped_file, model_file, less_than_50_files, more_than_50_files
):
    # Calculate total preprocessed files
    model_files = load_json_file(model_file)
    total_preprocessed_error_files = (
        set(less_than_50_files) | set(more_than_50_files)
    ) - set(model_files)

    # Analyze each group
    results_less_than_50 = analyze_model_group(
        model_name,
        untyped_file,
        model_file,
        set(less_than_50_files.keys()),
        "Less Than 50 Parameters",
    )

    results_more_than_50 = analyze_model_group(
        model_name,
        untyped_file,
        model_file,
        set(more_than_50_files.keys()),
        "More Than 50 Parameters",
    )

    total_preprocessed_error_files_count = len(total_preprocessed_error_files)

    total_preprocessed_error_files_count_less_than_50 = 0
    for file in total_preprocessed_error_files:
        if file in less_than_50_files:
            total_preprocessed_error_files_count_less_than_50 += 1
    total_preprocessed_error_files_count_more_than_50 = 0
    for file in total_preprocessed_error_files:
        if file in more_than_50_files:
            total_preprocessed_error_files_count_more_than_50 += 1

    return {
        "model": model_name,
        "total_preprocessed_files": total_preprocessed_error_files_count,
        "less_than_50_parameters": results_less_than_50,
        "more_than_50_parameters": results_more_than_50,
        "total_preprocessed_error_files_count_less_than_50": total_preprocessed_error_files_count_less_than_50,
        "total_preprocessed_error_files_count_more_than_50": total_preprocessed_error_files_count_more_than_50,
    }


if __name__ == "__main__":
    # Load and split files by parameter count
    untyped_data = load_json_file("mypy_outputs/mypy_results_untyped_with_errors.json")
    files_with_less_than_50_parameters, files_with_more_than_50_parameters = split_files_by_parameter_count(untyped_data)

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
            files_with_less_than_50_parameters,
            files_with_more_than_50_parameters,
        )
        all_results.append(results)

    # Print summary in the requested format
    for result in all_results:
        model = result["model"]
        total_preprocessed = result["total_preprocessed_files"]

        less_than_50 = result["less_than_50_parameters"]
        more_than_50 = result["more_than_50_parameters"]
        total_preprocessed_error_files_count_less_than_50 = result[
            "total_preprocessed_error_files_count_less_than_50"
        ]
        total_preprocessed_error_files_count_more_than_50 = result[
            "total_preprocessed_error_files_count_more_than_50"
        ]
        print(model)
        # print("Total preprocessed error files:", total_preprocessed)
        print("unprocessed, both_fail, both_success, llm_only_fail, LLM_success_rate")
        print(
            f"{less_than_50['unprocessed']+total_preprocessed_error_files_count_less_than_50}, {less_than_50['both_errors']}, {less_than_50['both_success']}, {less_than_50['llm_only_failures']}, {less_than_50['llm_success_rate']:.2f}%"
        )
        print(
            f"{more_than_50['unprocessed']+total_preprocessed_error_files_count_more_than_50}, {more_than_50['both_errors']}, {more_than_50['both_success']}, {more_than_50['llm_only_failures']}, {more_than_50['llm_success_rate']:.2f}%"
        ) 