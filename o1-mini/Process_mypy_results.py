import json

def summarize_mypy_results(json_file):
    """
    Summarizes mypy results from a JSON file with type and no-type variations.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return

    unique_files = set()
    both_compiled = 0
    only_no_types_compiled = 0
    only_types_compiled = 0

    for filename in data:
        base_filename = filename.replace("_no_types.py", ".py")
        unique_files.add(base_filename)

    unique_file_list = list(unique_files)

    for base_filename in unique_file_list:
        typed_filename = base_filename.replace(".py", ".py")
        no_types_filename = base_filename.replace(".py", "_no_types.py")

        if typed_filename in data and no_types_filename in data:
            typed_compiled = data[typed_filename]["isCompiled"]
            no_types_compiled = data[no_types_filename]["isCompiled"]

            if typed_compiled and no_types_compiled:
                both_compiled += 1
            elif not typed_compiled and no_types_compiled:
                only_no_types_compiled += 1
            elif typed_compiled and not no_types_compiled:
                only_types_compiled += 1

    print("Summary:")
    print(f"Total unique files: {len(unique_files)}")
    print(f"Files with both versions compiled: {both_compiled}")
    print(f"Files with only no-types version compiled: {only_no_types_compiled}")
    print(f"Files with only type annotated version compiled: {only_types_compiled}")

if __name__ == "__main__":
    json_file = "mypy_results.json"  # Replace with your JSON file name
    summarize_mypy_results(json_file)