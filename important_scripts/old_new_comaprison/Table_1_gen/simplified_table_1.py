import json
import os

# File paths (edit as needed)
COMMON_FILE_INFO = [
    "signature_comparison_results_deepseek.json",
    "signature_comparison_results_o1-mini.json",
    "signature_comparison_results_gpt4o.json",
]
OLD_TYPED = [
    "mypy_results_deepseek_old_with_errors_with_types.json",
    "mypy_results_o1-mini_old_with_errors_with_types.json",
    "mypy_results_ALL_GPT40_old_with_errors_with_types.json",
]
OLD_UNTYPED = [
    "mypy_results_deepseek_old_with_errors_no_types.json",
    "mypy_results_o1-mini_old_with_errors_no_types.json",
    "mypy_results_ALL_GPT40_old_with_errors_no_types.json",
]
NEW_TYPED = [
    "mypy_results_deepseek_with_errors.json",
    "mypy_results_o1_mini_with_errors.json",
    "mypy_results_gpt4o_with_errors.json",
]
NEW_UNTYPED = [
    "mypy_results_untyped_with_errors.json",
    "mypy_results_untyped_with_errors.json",
    "mypy_results_untyped_with_errors.json",
]


# Helper functions
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def has_syntax_error(errors):
    return any(
        error_type in error.lower()
        for error in errors
        for error_type in ["syntax", "empty_body", "name_defined"]
    )


def get_common_files(common_file_info):
    data = load_json(common_file_info)
    common = set()
    for entry in data:
        for old_file, new_file in zip(entry["old_files"], entry["new_files"]):
            common.add((old_file, new_file))
    return common


def analyze_common_files(typed, untyped, common_files, use_old):
    typed_data = load_json(typed)
    untyped_data = load_json(untyped)
    idx = 0 if use_old else 1
    files = [f[idx] for f in common_files]
    results = {
        "Processed by LLM": 0,  # will be set to total
        "Not Processed": 0,
        "Both Failures": 0,
        "Both Success": 0,
        "LLM-Only Failures": 0,
        # 'LLM-Only Success' removed
    }
    total = len(files)
    for fname in files:
        typed_errors = typed_data.get(fname, {}).get("errors", [])
        untyped_errors = untyped_data.get(fname, {}).get("errors", [])
        if has_syntax_error(typed_errors):
            results["Not Processed"] += 1
        else:
            typed_ok = len(typed_errors) == 0
            untyped_ok = len(untyped_errors) == 0
            if not typed_ok and not untyped_ok:
                results["Both Failures"] += 1
            elif typed_ok and untyped_ok:
                results["Both Success"] += 1
            elif not typed_ok and untyped_ok:
                results["LLM-Only Failures"] += 1
            elif typed_ok and not untyped_ok:
                # Instead of tracking 'LLM-Only Success', add to 'Both Failures'
                results["Both Failures"] += 1
    results["Processed by LLM"] = total
    # % Success = 100 * (1 - (LLM-Only Failures) / (LLM-Only Failures + Both Success))
    denom = results["LLM-Only Failures"] + results["Both Success"]
    if denom > 0:
        results["% Success"] = 100 * (1 - (results["LLM-Only Failures"] / denom))
    else:
        results["% Success"] = 0
    # Check sum
    sum_check = (
        results["Not Processed"]
        + results["Both Failures"]
        + results["Both Success"]
        + results["LLM-Only Failures"]
        # 'LLM-Only Success' removed from sum
    )
    if sum_check != total:
        print(
            f"WARNING: Sum of categories ({sum_check}) does not match total ({total})!"
        )
    return results, total


def print_table(title, stats, total):
    print(f"\n{title} (Total common files: {total})")
    print(
        "| Processed by LLM | Not Processed | Both Failures | Both Success | LLM-Only Failures | % Success |"
    )
    print(
        "|------------------|--------------|---------------|--------------|-------------------|-----------|"
    )
    print(
        f"| {stats['Processed by LLM']} | {stats['Not Processed']} | {stats['Both Failures']} | {stats['Both Success']} | {stats['LLM-Only Failures']} | {stats['% Success']:.2f} |"
    )


if __name__ == "__main__":
    num_models = len(COMMON_FILE_INFO)
    for i in range(num_models):
        model_name = (
            os.path.splitext(os.path.basename(COMMON_FILE_INFO[i]))[0]
            .replace("signature_comparison_results_", "")
            .replace(".json", "")
            .upper()
        )
        print(f"\n===== {model_name} =====")
        common_files = get_common_files(COMMON_FILE_INFO[i])
        # Old run
        old_stats, old_total = analyze_common_files(
            OLD_TYPED[i], OLD_UNTYPED[i], common_files, use_old=True
        )
        print_table("Old Common Files", old_stats, old_total)
        # New run
        new_stats, new_total = analyze_common_files(
            NEW_TYPED[i], NEW_UNTYPED[i], common_files, use_old=False
        )
        print_table("New Common Files", new_stats, new_total)

        # Print number of files in NEW_TYPED not in common files
        new_typed_data = load_json(NEW_TYPED[i])
        total_new_typed = len(new_typed_data)
        not_in_common = total_new_typed - new_total
        print(f"Files in NEW_TYPED not in common files: {not_in_common} (Total NEW_TYPED: {total_new_typed}, Common: {new_total})")
