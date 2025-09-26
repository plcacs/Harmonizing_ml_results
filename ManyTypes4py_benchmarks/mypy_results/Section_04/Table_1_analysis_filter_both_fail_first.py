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


def analyze_model_filter_both_fail_first(model_name, untyped_file, model_file):
    untyped_files = load_json_file(untyped_file)
    model_files = load_json_file(model_file)

    # 1) Filter out both_failed: files with isCompiled == False in untyped version
    both_failed_files = {
        file_key
        for file_key, file_data in untyped_files.items()
        if not file_data.get("isCompiled", True)
    }

    # Remove both_failed files from model analysis
    remaining_model_files = {
        k: v for k, v in model_files.items() if k not in both_failed_files
    }

    # 2) Calculate unprocessed files: files with errors but not type-related errors
    unprocessed_files = []
    for file_key, file_data in remaining_model_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            unprocessed_files.append(file_key)

    # 3) Among remaining files, calculate both_success and LLM_only_fail
    both_success_files = []  # untyped == 0, model == 0
    llm_only_failure_files = []  # untyped == 0, model > 0

    for file_key in remaining_model_files:
        if file_key in unprocessed_files:
            continue  # Skip unprocessed files

        untyped_error_count = untyped_files[file_key]["error_count"]
        model_error_count = remaining_model_files[file_key]["error_count"]

        if untyped_error_count == 0 and model_error_count == 0:
            both_success_files.append(file_key)
        elif untyped_error_count == 0 and model_error_count > 0:
            llm_only_failure_files.append(file_key)

    # LLM success rate among evaluable set (untyped == 0)
    total_llm_evaluable = len(both_success_files) + len(llm_only_failure_files)
    llm_success_rate = (
        100 * (len(both_success_files) / total_llm_evaluable)
        if total_llm_evaluable > 0
        else 0
    )
    total_unprocessed = len(unprocessed_files) + (
        len(untyped_files)
        - len(both_success_files)
        - len(llm_only_failure_files)
        - len(both_failed_files)
        - len(unprocessed_files)
    )
    # Overall success ratio = 100 * (Both success / (Both success + LLM only fail + unprocessed))
    overall_success = (
        100
        * (
            len(both_success_files)
            / (
                len(both_success_files)
                + len(llm_only_failure_files)
                + total_unprocessed
            )
        )
        if (len(both_success_files) + len(llm_only_failure_files) + total_unprocessed)
        > 0
        else 0
    )

    return {
        "model": model_name,
        "both_failed": len(both_failed_files) - 2,
        "unprocessed": total_unprocessed,
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failure_files),
        "llm_success_rate": llm_success_rate,
        "overall_success": overall_success,
        "llm_only_failure_files": llm_only_failure_files,
    }


if __name__ == "__main__":
    # Ordered: GPT3.5 → GPT4O → o1-mini → o3-mini → Deepseek → Claude (then others)
    models = [
        (
            "gpt35_1st_run",
            "../mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
        ),
        (
            "gpt35_2nd_run",
            "../mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
        ),
        ("gpt4o", "../mypy_outputs/mypy_results_gpt4o_with_errors.json"),
        (
            "gpt4o_2nd_run",
            "../mypy_outputs/mypy_results_gpt4o_2nd_run_with_errors.json",
        ),
        ("o1-mini", "../mypy_outputs/mypy_results_o1_mini_with_errors.json"),
        (
            "o1-mini_2nd_run",
            "../mypy_outputs/mypy_results_o1_mini_2nd_run_with_errors.json",
        ),
        (
            "o3_mini_1st_run",
            "../mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        ),
        (
            "o3_mini_2nd_run",
            "../mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json",
        ),
        (
            "o3_mini_3rd_run",
            "../mypy_outputs/mypy_results_o3_mini_3rd_run_with_errors.json",
        ),
        ("deepseek", "../mypy_outputs/mypy_results_deepseek_with_errors.json"),
        (
            "deepseek_2nd_run",
            "../mypy_outputs/mypy_results_deepseek_2nd_run_with_errors.json",
        ),
        (
            "claude_3_7_sonnet",
            "../mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        ),
        (
            "claude3_sonnet_2nd_run",
            "../mypy_outputs/mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json",
        ),
        ("gpt5_1st_run", "../mypy_outputs/mypy_results_gpt5_1st_run_with_errors.json"),
        ("original", "../mypy_outputs/mypy_results_original_files_with_errors.json"),
    ]

    all_results = []
    all_llm_only_failures = {}

    for model_name, model_file in models:
        results = analyze_model_filter_both_fail_first(
            model_name,
            "../mypy_outputs/mypy_results_untyped_with_errors.json",
            model_file,
        )
        all_results.append(results)
        # Collect LLM-only failure files for each model
        all_llm_only_failures[model_name] = results["llm_only_failure_files"]

    # Save LLM-only failure files to JSON
    with open("llm_only_failure_files.json", "w") as f:
        json.dump(all_llm_only_failures, f, indent=2)

    for result in all_results:
        model = result["model"]
        print(model)
        print(
            "both failed, unprocessed, both_success, llm_only_fail, LLM_success_rate, overall_success"
        )
        print(
            f"{result['both_failed']}, {result['unprocessed']}, {result['both_success']}, {result['llm_only_failures']}, {result['llm_success_rate']:.2f}%, {result['overall_success']:.2f}%"
        )
