import os
import ast
import json
from collections import defaultdict


def extract_signature_features(file_path):
    """Extracts function and class-based syntactic features from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "\x00" in content:
            return None  # Skip files with null bytes
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            if "\x00" in content:
                return None  # Skip files with null bytes
        except Exception:
            return None  # Skip files that can't be decoded

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return None  # Skip files with syntax errors

    features = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features.add(f"func:{node.name}")
            # features.add(f'args:{len(node.args.args)}')
        elif isinstance(node, ast.ClassDef):
            features.add(f"class:{node.name}")
    return sorted(features)


def process_directory(directory):
    result = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py") and not fname.endswith("_no_types.py"):
                fpath = os.path.join(root, fname)  # keep as relative path for reading
                features = extract_signature_features(fpath)
                if features is not None:
                    result[fname] = features  # only use the file name as key
    return result


# Example usage: update the path as needed
def update_syntactic_features_json(new_dir, json_output_path):
    # Load existing data
    if os.path.exists(json_output_path):
        with open(json_output_path, "r") as infile:
            all_results = json.load(infile)
    else:
        all_results = {}

    # Process new directory
    new_results = process_directory(new_dir)

    # Merge results
    all_results.update(new_results)

    # Save updated results
    with open(json_output_path, "w") as out:
        json.dump(all_results, out, indent=2)

    # print(f"Number of instances in output: {len(new_results)}")
    return all_results


# Example usage:
directories = [
    "ALL_GPT40",
    "ALL_O1_mini",
    "ALL_DEEP_SEEK",
]  # change this as needed for each directory
json_output_path = [
    "gpt40_code_similarity_old.json",
    "o1-mini_code_similarity_old.json",
    "deepseek_code_similarity_old.json",
]


for index, dir in enumerate(directories):
    print(f"Processing directory: {dir}")
    new_dir = dir
    update_syntactic_features_json(new_dir, json_output_path[index])

    # Print total count after processing all directories
    with open(json_output_path[index], "r") as f:
        data = json.load(f)
        print(f"\nTotal number of instances in {json_output_path[index]}: {len(data)}")
