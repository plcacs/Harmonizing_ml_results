import json
import os

type_error_keywords = [
    "incompatible type",
    "has type",
    "expected",
    "type error",
    "return value type",
    "argument",
    "type mismatch",
    "invalid type",
    "type hint",
    "typed",
    "assignment",
    "call-arg",
    "var-annotated",
]


def is_type_error(msg):
    return any(k in msg.lower() for k in type_error_keywords)


def extract_error_code(error):
    # Extract error code from the end of the message, e.g., "[arg-type]" from "... [arg-type]"
    if "[" in error and "]" in error:
        return error[error.rindex("[") + 1 : error.rindex("]")]
    return ""


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


def get_strict_type_error_files(json_file):
    with open(json_file) as f:
        data = json.load(f)

    strict_type_error_files = []
    for fname, details in data.items():
        errors = details.get("errors", [])
        error_codes = [extract_error_code(error) for error in errors if error]
        error_codes = [code for code in error_codes if code]
        is_type_error = True
        for error_code in error_codes:
            if error_code in non_type_related_errors:

                is_type_error = False
                break

        if is_type_error:
            strict_type_error_files.append(fname)
        """error_msgs = [msg for msg in details.get("errors", []) if "Found" not in msg]
        if (
            error_msgs
            and all(is_type_error(msg) for msg in error_msgs)
            and all(
                "syntax" not in msg.lower() and "eof" not in msg.lower()
                for msg in error_msgs
            )
        ):
            strict_type_error_files.append(fname)"""
    return strict_type_error_files


# List of JSON files to process
llm_mypy_results = [
    "mypy_results_deepseek_with_errors.json",
    "mypy_results_gpt4o_with_errors.json",
    "mypy_results_o1_mini_with_errors.json",
]
base_file = "mypy_results_untyped_with_errors.json"

# Create output directory if it doesn't exist
output_dir = "Filtered_type_errors"
os.makedirs(output_dir, exist_ok=True)

# Process each file and get strict type error files
results = {}
for json_file in llm_mypy_results:
    results[json_file] = get_strict_type_error_files(json_file)
    print(f"\nFiles with strict type errors in {json_file}: {len(results[json_file])}")

# Merge results with base file
merged_results = {}
for json_file in llm_mypy_results:
    with open(json_file) as f:
        llm_data = json.load(f)
    with open(base_file) as f:
        base_data = json.load(f)

    merged_data = {}
    for fname in results[json_file]:
        if fname in base_data:
            merged_data[fname] = {
                "base_stats": base_data[fname]["stats"],
                "llm_stats": llm_data[fname]["stats"],
                "base_error_count": base_data[fname]["error_count"],
                "llm_error_count": llm_data[fname]["error_count"],
                "base_errors": base_data[fname]["errors"],
                "llm_errors": llm_data[fname]["errors"],
            }

    output_file = os.path.join(
        output_dir, f"merged_{json_file.replace('_with_errors.json', '')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)
    print(f"Created merged file: {output_file} with {len(merged_data)} entries")
