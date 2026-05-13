import json
import os


def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_canonical_filenames(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".py"))


def has_non_type_error(errors):
    non_type_related_codes = [
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
        kw in error.lower()
        for error in errors
        for kw in ["syntax", "empty_body", "name_defined"]
    ):
        return True

    for error in errors:
        code = extract_error_code(error)
        if code in non_type_related_codes:
            return True
    return False


def analyze_model(model_data, canonical_files):
    unprocessed = []
    both_success = []
    llm_only_fail = []
    missing = []

    for fname in canonical_files:
        if fname not in model_data:
            missing.append(fname)
            continue

        file_data = model_data[fname]
        if has_non_type_error(file_data.get("errors", [])):
            unprocessed.append(fname)
            continue

        if file_data["error_count"] == 0:
            both_success.append(fname)
        else:
            llm_only_fail.append(fname)

    total_unprocessed = len(unprocessed) + len(missing)
    evaluable = len(both_success) + len(llm_only_fail)
    success_rate = 100 * len(both_success) / evaluable if evaluable > 0 else 0
    overall = (
        100 * len(both_success) / (evaluable + total_unprocessed)
        if (evaluable + total_unprocessed) > 0
        else 0
    )

    return {
        "unprocessed": total_unprocessed,
        "both_success": len(both_success),
        "llm_only_fail": len(llm_only_fail),
        "success_rate": success_rate,
        "overall_success": overall,
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    mypy_dir = os.path.join(base, "mypy_outputs")
    untyped_dir = os.path.join(base, "..", "500_untyped_files")

    canonical_files = get_canonical_filenames(untyped_dir)

    models = {
        "DeepSeek": {
            "strict": "mypy_results_deepseek_4_run_with_errors_strict.json",
            "unstrict": "mypy_results_deepseek_2nd_run_with_errors.json",
            "stub": "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
        },
        "GPT-5": {
            "strict": "mypy_results_gpt5_4_run_with_errors.json",
            "unstrict": "mypy_results_gpt5_1st_run_with_errors.json",
            "stub": "mypy_results_gpt5_1_infer_stub_run_with_errors.json",
        },
        "Claude": {
            "strict": "mypy_results_claude3_sonnet_4_run_with_errors_strict.json",
            "unstrict": "mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json",
        },
    }

    header = f"{'Model':<12} {'Setting':<10} {'Unprocessed':>12} {'Success':>9} {'LLM Fail':>10} {'Success%':>10} {'Overall%':>10}"
    sep = "-" * len(header)

    for llm_name, settings in models.items():
        print(f"\n{llm_name}")
        print(sep)
        print(header)
        print(sep)
        for setting_name, json_file in settings.items():
            data = load_json_file(os.path.join(mypy_dir, json_file))
            r = analyze_model(data, canonical_files)
            print(
                f"{llm_name:<12} {setting_name:<10} {r['unprocessed']:>12} "
                f"{r['both_success']:>9} {r['llm_only_fail']:>10} "
                f"{r['success_rate']:>9.2f}% {r['overall_success']:>9.2f}%"
            )
        print(sep)
