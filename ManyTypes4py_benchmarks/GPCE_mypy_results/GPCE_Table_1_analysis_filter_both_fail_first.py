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


def analyze_model_filter_both_fail_first(model_name, model_file):
    # Here we assume all files in model_file successfully type-check
    # in the untyped baseline, so we don't need that JSON.
    model_files = load_json_file(model_file)

    # 1) Calculate unprocessed files: files with errors but not type-related errors
    unprocessed_files = []
    for file_key, file_data in model_files.items():
        if has_syntax_error(file_data.get("errors", [])):
            unprocessed_files.append(file_key)

    # 2) Among remaining files, calculate both_success and LLM_only_fail
    #    (since untyped is assumed success for all, this is just
    #     model error_count == 0 vs > 0)
    both_success_files = []
    llm_only_failure_files = []

    for file_key, file_data in model_files.items():
        if file_key in unprocessed_files:
            continue  # Skip unprocessed files

        model_error_count = file_data["error_count"]

        if model_error_count == 0:
            both_success_files.append(file_key)
        elif model_error_count > 0:
            llm_only_failure_files.append(file_key)

    # 3) Metrics
    total_files_assumed = 500
    missing_files = max(0, total_files_assumed - len(model_files))

    # LLM success rate among evaluable set (all non-unprocessed files)
    total_llm_evaluable = len(both_success_files) + len(llm_only_failure_files)
    llm_success_rate = (
        100 * (len(both_success_files) / total_llm_evaluable)
        if total_llm_evaluable > 0
        else 0
    )
    # Treat missing entries as additional unprocessed files
    total_unprocessed = len(unprocessed_files) + missing_files
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
        "both_failed": 0,
        "unprocessed": total_unprocessed,
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failure_files),
        "llm_success_rate": llm_success_rate,
        "overall_success": overall_success,
        "llm_only_failure_files": llm_only_failure_files,
        "both_success_files": both_success_files,
        "unprocessed_files": unprocessed_files,
    }


if __name__ == "__main__":
    # GPCE comparison: focus on prompt variants/runs
    models = [
        (
            "gpt5_2_run",
            "mypy_results_gpt5_2_run_with_errors.json",
        ),
        (
            "gpt5_3_run",
            "mypy_results_gpt5_3_run_with_errors.json",
        ),
        (
            "claude3_sonnet_3_run",
            "mypy_results_claude3_sonnet_3_run_with_errors.json",
        ),
        (
            "claude3_sonnet_4_run",
            "mypy_results_claude3_sonnet_4_run_with_errors.json",
        ),
        ("deepseek_3_run", "mypy_results_deepseek_3_run_with_errors.json"),
        ("deepseek_4_run", "mypy_results_deepseek_4_run_with_errors.json"),
    ]

    all_results = []
    all_llm_only_failures = {}
    all_both_success_files = {}
    all_unprocessed_files = {}

    for model_name, model_file in models:
        results = analyze_model_filter_both_fail_first(
            model_name,
            model_file,
        )
        all_results.append(results)
        all_llm_only_failures[model_name] = results["llm_only_failure_files"]
        all_both_success_files[model_name] = results["both_success_files"]
        all_unprocessed_files[model_name] = results["unprocessed_files"]

    with open("GPCE_llm_only_failure_files.json", "w") as f:
        json.dump(all_llm_only_failures, f, indent=2)
    with open("GPCE_both_success_files.json", "w") as f:
        json.dump(all_both_success_files, f, indent=2)
    with open("GPCE_unprocessed_files.json", "w") as f:
        json.dump(all_unprocessed_files, f, indent=2)

    for result in all_results:
        model = result["model"]
        print(model)
        print(
            "both failed, unprocessed, both_success, llm_only_fail, LLM_success_rate, overall_success"
        )
        print(
            f"{result['both_failed']}, {result['unprocessed']}, {result['both_success']}, {result['llm_only_failures']}, {result['llm_success_rate']:.2f}%, {result['overall_success']:.2f}%"
        )

