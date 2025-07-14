import json
from collections import defaultdict


def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)


def has_syntax_error(errors):
    return any(
        error_type in error.lower()
        for error in errors
        for error_type in ["syntax", "empty_body", "name_defined"]
    )


def extract_error_code(error):
    # Extract error code from the end of the message, e.g., "[arg-type]" from "... [arg-type]"
    if "[" in error and "]" in error:
        return error[error.rindex("[") + 1 : error.rindex("]")]
    return ""


explicit_type_error_keywords = [
    "arg-type",
    "assignment",
    "return-value",
    "index",
    "operator",
    "union-attr",
    "call-overload",
    "type-var",
    "valid-type",
    "has-type",
    "var-annotated",
    "override",
    "attr-defined",
    "misc",
    "type-alias",
    "type-arg",
    "comparison-overlap",
    "truthy-bool",
    "redundant-expr",
]
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


def is_type_error(error):
    error_code = extract_error_code(error)
    return error_code in explicit_type_error_keywords


def analyze_files(model_name, base_file, model_file, file_subset=None):
    print(f"Analyzing {model_name} with {base_file} and {model_file}")
    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)

    # Precompute sets
    if file_subset is not None:
        all_keys = set(file_subset)
    else:
        all_keys = set(no_type.keys())

    # Get files with syntax errors
    syntax_error_files = {
        k for k in model if has_syntax_error(model[k].get("errors", []))
    }

    # Number of files processed by LLM (present in model, excluding syntax errors)
    num_processed_by_llm = len(all_keys)
    # Number of files not processed by LLM (present in base but not in model, plus files with syntax errors)
    num_not_processed_by_llm = len(set(all_keys) - set(model.keys())) + len(
        syntax_error_files & all_keys
    )

    # Categorize files
    both_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] > 0
    ]
    both_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] == 0
    ]
    llm_only_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] > 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    llm_only_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] == 0
        and not has_syntax_error(model[k].get("errors", []))
    ]

    # Analyze type errors in llm_only_failures_files
    type_error_files = []
    other_error_files = []
    count = 0
    for file in llm_only_failures_files:
        errors = model[file].get("errors", [])
        error_codes = [extract_error_code(error) for error in errors if error]
        error_codes = [code for code in error_codes if code]  # Filter out empty strings
        # print(error_codes)
        is_type_error = True
        for error_code in error_codes:
            if error_code in non_type_related_errors:
                other_error_files.append(file)
                is_type_error = False
                break

        if len(error_codes) > 0 and is_type_error:
            type_error_files.append(file)

    # Save results to JSON
    results = {
        "num_processed_by_llm": num_processed_by_llm,
        "num_not_processed_by_llm": num_not_processed_by_llm,
        "both_failures": len(both_failures_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failures_files),
        "llm_only_success": len(llm_only_success_files),
        "type_error_files": len(type_error_files),
        "other_error_files": len(other_error_files),
        "files": {
            "both_failures": both_failures_files,
            "both_success": both_success_files,
            "llm_only_failures": llm_only_failures_files,
            "llm_only_success": llm_only_success_files,
            "type_error_files": type_error_files,
            "other_error_files": other_error_files,
        },
    }

    # with open(f"analysis_{model_name}.json", "w") as f:
    #    json.dump(results, f, indent=2)

    # Print table header
    print(f"\nResults for {model_name}:")
    print(
        "| Number of file processed by llm | Number of file could not processed by llm | both-failures | both success | llm-only failures | llm-only success | % LLM-Only Success | Type Errors | Other Errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    percent_success = (
        100
        * (
            1
            - (
                len(llm_only_failures_files)
                / (len(llm_only_failures_files) + len(both_success_files))
            )
        )
        if (len(llm_only_failures_files) + len(both_success_files)) > 0
        else 0
    )
    print(
        f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {len(both_failures_files)} | {len(both_success_files)} | {len(llm_only_failures_files)} | {len(llm_only_success_files)} | {percent_success:.2f} | {len(type_error_files)} | {len(other_error_files)} |\n"
    )
    print(f"Total LLM-only error files: {len(llm_only_failures_files)}\n")
    print(f"Type error files: {len(type_error_files)}")
    print(f"Other error files: {len(other_error_files)}")
    print(
        f"Type error files percentage: {100 * (len(type_error_files) / len(llm_only_failures_files)):.2f}%"
        if len(llm_only_failures_files) > 0
        else "Type error files percentage: 0%"
    )
    print(
        f"Other error files percentage: {100 * (len(other_error_files) / len(llm_only_failures_files)):.2f}%"
        if len(llm_only_failures_files) > 0
        else "Other error files percentage: 0%"
    )

    return results


def analyze_old_files(
    model_name,
    base_file,
    model_file,
    file_subset=None,
    base_count=1200,
    common_count=468,
):
    """Analyze files that are in the old_files group (files that existed in both old and new versions)"""
    print(f"Analyzing OLD files for {model_name}")

    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)

    # Filter to only old files
    if file_subset is not None:
        all_keys = set(file_subset)
    else:
        all_keys = set(no_type.keys())

    # Get files with syntax errors
    syntax_error_files = {
        k for k in model if has_syntax_error(model[k].get("errors", []))
    }

    # Number of files processed by LLM (present in model, excluding syntax errors)
    num_processed_by_llm = base_count - common_count
    # Number of files not processed by LLM (present in base but not in model, plus files with syntax errors)

    # Categorize files for old files group
    both_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] > 0
    ]
    both_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] == 0
    ]
    llm_only_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] > 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    llm_only_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] == 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    num_not_processed_by_llm = (
        num_processed_by_llm
        - len(llm_only_failures_files)
        - len(llm_only_success_files)
        - len(both_failures_files)
        - len(both_success_files)
    )

    # Analyze type errors in llm_only_failures_files
    type_error_files = []
    other_error_files = []
    for file in llm_only_failures_files:
        errors = model[file].get("errors", [])
        error_codes = [extract_error_code(error) for error in errors if error]
        error_codes = [code for code in error_codes if code]

        is_type_error = True
        for error_code in error_codes:
            if error_code in non_type_related_errors:
                other_error_files.append(file)
                is_type_error = False
                break

        if len(error_codes) > 0 and is_type_error:
            type_error_files.append(file)

    # Save results to JSON
    results = {
        "group": "old_files",
        "num_processed_by_llm": num_processed_by_llm,
        "num_not_processed_by_llm": num_not_processed_by_llm,
        "both_failures": len(both_failures_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failures_files),
        "llm_only_success": len(llm_only_success_files),
        "type_error_files": len(type_error_files),
        "other_error_files": len(other_error_files),
        "files": {
            "both_failures": both_failures_files,
            "both_success": both_success_files,
            "llm_only_failures": llm_only_failures_files,
            "llm_only_success": llm_only_success_files,
            "type_error_files": type_error_files,
            "other_error_files": other_error_files,
        },
    }

    # with open(f"analysis_{model_name}_old_files.json", "w") as f:
    #    json.dump(results, f, indent=2)
    """print("num_processed_by_llm: ", num_processed_by_llm)
    print("num_not_processed_by_llm: ", num_not_processed_by_llm)
    print("both_failures: ", len(both_failures_files))
    print("both_success: ", len(both_success_files))
    print("llm_only_failures: ", len(llm_only_failures_files))
    print("llm_only_success: ", len(llm_only_success_files))
    print("type_error_files: ", len(type_error_files))
    print("syntax_error_files: ", len(syntax_error_files))"""
    # Print results
    print(f"\nOLD FILES Results for {model_name}:")
    print(
        "| Processed by LLM | Not Processed | Both Failures | Both Success | LLM-Only Failures | LLM-Only Success | % Success | Type Errors | Other Errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    percent_success = (
        100
        * (
            1
            - (
                len(llm_only_failures_files)
                / (len(llm_only_failures_files) + len(both_success_files))
            )
        )
        if (len(llm_only_failures_files) + len(both_success_files)) > 0
        else 0
    )
    print(
        f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {len(both_failures_files)} | {len(both_success_files)} | {len(llm_only_failures_files)} | {len(llm_only_success_files)} | {percent_success:.2f}% | {len(type_error_files)} | {len(other_error_files)} |"
    )

    return results


def analyze_new_files(
    model_name,
    base_file,
    model_file,
    file_subset=None,
    base_count=1200,
    common_count=468,
):
    """Analyze files that are in the new_files group (files that only exist in new version)"""
    print(f"Analyzing NEW files for {model_name}")

    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)
    print("no_type: ", len(no_type))
    # Filter to only new files
    if file_subset is not None:
        all_keys = set(file_subset)
    else:
        all_keys = set(no_type.keys())

    # Get files with syntax errors
    syntax_error_files = {
        k for k in model if has_syntax_error(model[k].get("errors", []))
    }

    # Number of files processed by LLM (present in model, excluding syntax errors)
    num_processed_by_llm = base_count - common_count
    # Number of files not processed by LLM (present in base but not in model, plus files with syntax errors)

    # Categorize files for new files group
    both_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] > 0
    ]
    both_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] == 0
    ]
    llm_only_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] > 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    llm_only_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] == 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    num_not_processed_by_llm = (
        num_processed_by_llm
        - len(llm_only_failures_files)
        - len(llm_only_success_files)
        - len(both_failures_files)
        - len(both_success_files)
    )

    # Analyze type errors in llm_only_failures_files
    type_error_files = []
    other_error_files = []
    for file in llm_only_failures_files:
        errors = model[file].get("errors", [])
        error_codes = [extract_error_code(error) for error in errors if error]
        error_codes = [code for code in error_codes if code]

        is_type_error = True
        for error_code in error_codes:
            if error_code in non_type_related_errors:
                other_error_files.append(file)
                is_type_error = False
                break

        if len(error_codes) > 0 and is_type_error:
            type_error_files.append(file)

    # Save results to JSON
    results = {
        "group": "new_files",
        "num_processed_by_llm": num_processed_by_llm,
        "num_not_processed_by_llm": num_not_processed_by_llm,
        "both_failures": len(both_failures_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failures_files),
        "llm_only_success": len(llm_only_success_files),
        "type_error_files": len(type_error_files),
        "other_error_files": len(other_error_files),
        "files": {
            "both_failures": both_failures_files,
            "both_success": both_success_files,
            "llm_only_failures": llm_only_failures_files,
            "llm_only_success": llm_only_success_files,
            "type_error_files": type_error_files,
            "other_error_files": other_error_files,
        },
    }
    """print("num_processed_by_llm: ", num_processed_by_llm)
    print("num_not_processed_by_llm: ", num_not_processed_by_llm)
    print("both_failures: ", len(both_failures_files))
    print("both_success: ", len(both_success_files))
    print("llm_only_failures: ", len(llm_only_failures_files))
    print("llm_only_success: ", len(llm_only_success_files))
    print("type_error_files: ", len(type_error_files))
    print("syntax_error_files: ", len(syntax_error_files))"""
    # with open(f"analysis_{model_name}_new_files.json", "w") as f:
    #    json.dump(results, f, indent=2)

    # Print results
    print(f"\nNEW FILES Results for {model_name}:")
    print(
        "| Processed by LLM | Not Processed | Both Failures | Both Success | LLM-Only Failures | LLM-Only Success | % Success | Type Errors | Other Errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    percent_success = (
        100
        * (
            1
            - (
                len(llm_only_failures_files)
                / (len(llm_only_failures_files) + len(both_success_files))
            )
        )
        if (len(llm_only_failures_files) + len(llm_only_success_files)) > 0
        else 0
    )
    print(
        f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {len(both_failures_files)} | {len(both_success_files)} | {len(llm_only_failures_files)} | {len(llm_only_success_files)} | {percent_success:.2f}% | {len(type_error_files)} | {len(other_error_files)} |"
    )

    return results


def analyze_common_files(model_name, base_file, model_file, file_subset=None):
    """Analyze files that are in the common_files group (files that exist in mypy results but not in signature comparison)"""
    print(f"Analyzing COMMON files for {model_name}")

    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)

    # Filter to only common files
    if file_subset is not None:
        all_keys = set(file_subset)
    else:
        all_keys = set(no_type.keys())

    # Get files with syntax errors
    syntax_error_files = {
        k for k in model if has_syntax_error(model[k].get("errors", []))
    }

    # Number of files processed by LLM (present in model, excluding syntax errors)
    num_processed_by_llm = len(all_keys)
    # Number of files not processed by LLM (present in base but not in model, plus files with syntax errors)
    num_not_processed_by_llm = len(set(all_keys) - set(model.keys())) + len(
        syntax_error_files & all_keys
    )

    # Categorize files for common files group
    both_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] > 0
    ]
    both_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] == 0
    ]
    llm_only_failures_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] == 0
        and model[k]["error_count"] > 0
        and not has_syntax_error(model[k].get("errors", []))
    ]
    llm_only_success_files = [
        k
        for k in all_keys
        if k in no_type
        and k in model
        and no_type[k]["error_count"] > 0
        and model[k]["error_count"] == 0
        and not has_syntax_error(model[k].get("errors", []))
    ]

    # Analyze type errors in llm_only_failures_files
    type_error_files = []
    other_error_files = []
    for file in llm_only_failures_files:
        errors = model[file].get("errors", [])
        error_codes = [extract_error_code(error) for error in errors if error]
        error_codes = [code for code in error_codes if code]

        is_type_error = True
        for error_code in error_codes:
            if error_code in non_type_related_errors:
                other_error_files.append(file)
                is_type_error = False
                break

        if len(error_codes) > 0 and is_type_error:
            type_error_files.append(file)

    # Save results to JSON
    results = {
        "group": "common_files",
        "num_processed_by_llm": num_processed_by_llm,
        "num_not_processed_by_llm": num_not_processed_by_llm,
        "both_failures": len(both_failures_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failures_files),
        "llm_only_success": len(llm_only_success_files),
        "type_error_files": len(type_error_files),
        "other_error_files": len(other_error_files),
        "files": {
            "both_failures": both_failures_files,
            "both_success": both_success_files,
            "llm_only_failures": llm_only_failures_files,
            "llm_only_success": llm_only_success_files,
            "type_error_files": type_error_files,
            "other_error_files": other_error_files,
        },
    }

    # with open(f"analysis_{model_name}_common_files.json", "w") as f:
    #    json.dump(results, f, indent=2)

    # Print results
    print(f"\nCOMMON FILES Results for {model_name}:")
    print(
        "| Processed by LLM | Not Processed | Both Failures | Both Success | LLM-Only Failures | LLM-Only Success | % Success | Type Errors | Other Errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    percent_success = (
        100
        * (
            1
            - (
                len(llm_only_failures_files)
                / (len(llm_only_failures_files) + len(both_success_files))
            )
        )
        if (len(llm_only_failures_files) + len(both_success_files)) > 0
        else 0
    )
    print(
        f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {len(both_failures_files)} | {len(both_success_files)} | {len(llm_only_failures_files)} | {len(llm_only_success_files)} | {percent_success:.2f}% | {len(type_error_files)} | {len(other_error_files)} |"
    )

    return results


def get_groups(signature_file, mypy_results_file, mypy_results_file_old):
    with open(signature_file, "r") as f:
        sig_data = json.load(f)  # This is a list of dictionaries
    with open(mypy_results_file, "r") as f:
        new_data = json.load(f)
    with open(mypy_results_file_old, "r") as f:
        old_data = json.load(f)

    # Extract all old and new files from signature data
    old_files_from_sig = set()
    new_files_from_sig = set()
    for entry in sig_data:
        old_files_from_sig.update(entry.get("old_files", []))
        new_files_from_sig.update(entry.get("new_files", []))

    # Get all files from mypy results
    old_mypy_files = set(old_data.keys())
    new_mypy_files = set(new_data.keys())

    # Calculate the groups

    old_only_files = (
        old_mypy_files - old_files_from_sig
    )  # Files that only exist in old mypy results
    new_only_files = (
        new_mypy_files - new_files_from_sig
    )  # Files that only exist in new mypy results

    return (
        list(old_only_files),
        list(new_only_files),
        list(old_files_from_sig),
        list(new_files_from_sig),
    )


if __name__ == "__main__":
    """# GPT4o
    base_new_files = load_json_file("mypy_results_untyped_with_errors.json")
    base_count_new_files = len(base_new_files.keys())

    old_files, new_files, old_files_from_sig, new_files_from_sig = get_groups(
        "signature_comparison_results_gpt4o.json",
        "mypy_results_gpt4o_with_errors.json",
        "mypy_results_ALL_GPT40_old_with_errors_with_types.json",
    )
    print(
        "GPT4o: ",
        len(old_files),
        len(new_files),
        len(old_files_from_sig),
        len(new_files_from_sig),
    )

    print("\n--- GPT4o: Old unmatched Files Group ---")

    analyze_old_files(
        "gpt4o_old",
        "mypy_results_ALL_GPT40_old_with_errors_no_types.json",
        "mypy_results_ALL_GPT40_old_with_errors_with_types.json",
        file_subset=old_files,
        base_count=1200,
        common_count=len(old_files_from_sig),
    )
    print("\n--- GPT4o: old Common Files Group ---")

    analyze_common_files(
        "gpt4o_common",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_gpt4o_with_errors.json",
        file_subset=new_files_from_sig,
    )
    print("\n--- GPT4o: New unmatched Files Group ---")
    analyze_new_files(
        "gpt4o_new",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_gpt4o_with_errors.json",
        file_subset=new_files,
        base_count=base_count_new_files,
        common_count=len(new_files_from_sig),
    )
    print("\n--- GPT4o: New Common Files Group ---")

    analyze_common_files(
        "gpt4o_common",
        "mypy_results_ALL_GPT40_old_with_errors_no_types.json",
        "mypy_results_ALL_GPT40_old_with_errors_with_types.json",
        file_subset=old_files_from_sig,
    )
    """
    """
    # o1-mini
    base_new_files_o1 = load_json_file("mypy_results_untyped_with_errors.json")
    base_count_new_files_o1 = len(base_new_files_o1.keys())

    old_files_o1, new_files_o1, old_files_from_sig_o1, new_files_from_sig_o1 = (
        get_groups(
            "signature_comparison_results_o1-mini.json",
            "mypy_results_o1_mini_with_errors.json",
            "mypy_results_o1-mini_old_with_errors_with_types.json",
        )
    )
    print(
        "o1-mini: ",
        len(old_files_o1),
        len(new_files_o1),
        len(old_files_from_sig_o1),
        len(new_files_from_sig_o1),
    )

    print("\n--- o1-mini: Old unmatched Files Group ---")
    analyze_old_files(
        "o1-mini_old",
        "mypy_results_o1-mini_old_with_errors_no_types.json",
        "mypy_results_o1-mini_old_with_errors_with_types.json",
        file_subset=old_files_o1,
        base_count=1200,
        common_count=len(old_files_from_sig_o1),
    )
    print("\n--- o1-mini: old Common Files Group ---")
    analyze_common_files(
        "o1-mini_common_old",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_o1_mini_with_errors.json",
        file_subset=new_files_from_sig_o1,
    )
    print("\n--- o1-mini: New unmatched Files Group ---")
    analyze_new_files(
        "o1-mini_new",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_o1_mini_with_errors.json",
        file_subset=new_files_o1,
        base_count=base_count_new_files_o1,
        common_count=len(new_files_from_sig_o1),
    )
    print("\n--- o1-mini: New Common Files Group ---")
    analyze_common_files(
        "o1-mini_common_new",
        "mypy_results_o1-mini_old_with_errors_no_types.json",
        "mypy_results_o1-mini_old_with_errors_with_types.json",
        file_subset=old_files_from_sig_o1,
    )
    """

    # Deepseek
    base_new_files_deepseek = load_json_file("mypy_results_untyped_with_errors.json")
    base_count_new_files_deepseek = len(base_new_files_deepseek.keys())

    (
        old_files_deepseek,
        new_files_deepseek,
        old_files_from_sig_deepseek,
        new_files_from_sig_deepseek,
    ) = get_groups(
        "signature_comparison_results_deepseek.json",
        "mypy_results_deepseek_with_errors.json",
        "mypy_results_deepseek_old_with_errors_with_types.json",
    )
    print(
        "Deepseek: ",
        len(old_files_deepseek),
        len(new_files_deepseek),
        len(old_files_from_sig_deepseek),
        len(new_files_from_sig_deepseek),
    )

    print("\n--- Deepseek: Old unmatched Files Group ---")
    analyze_old_files(
        "deepseek_old",
        "mypy_results_deepseek_old_with_errors_no_types.json",
        "mypy_results_deepseek_old_with_errors_with_types.json",
        file_subset=old_files_deepseek,
        base_count=1200,
        common_count=len(old_files_from_sig_deepseek),
    )
    print("\n--- Deepseek: old Common Files Group ---")
    analyze_common_files(
        "deepseek_common_old",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_deepseek_with_errors.json",
        file_subset=new_files_from_sig_deepseek,
    )
    print("\n--- Deepseek: New unmatched Files Group ---")
    analyze_new_files(
        "deepseek_new",
        "mypy_results_untyped_with_errors.json",
        "mypy_results_deepseek_with_errors.json",
        file_subset=new_files_deepseek,
        base_count=base_count_new_files_deepseek,
        common_count=len(new_files_from_sig_deepseek),
    )
    print("\n--- Deepseek: New Common Files Group ---")
    analyze_common_files(
        "deepseek_common_new",
        "mypy_results_deepseek_old_with_errors_no_types.json",
        "mypy_results_deepseek_old_with_errors_with_types.json",
        file_subset=old_files_from_sig_deepseek,
    )
