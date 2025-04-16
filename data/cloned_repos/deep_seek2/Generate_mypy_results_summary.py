import json
from collections import defaultdict

# Load the JSON file
with open("mypy_results.json", "r") as f:
    data = json.load(f)

# Dictionary to store compilation results
file_pairs = defaultdict(lambda: {"original": None, "no_types": None})

# Process files
for filename, attributes in data.items():
    is_compiled = attributes.get("isCompiled", False)
    
    if filename.endswith("_no_types.py") and filename.count("_no_types") == 1:
        original_name = filename.replace("_no_types", "")
        file_pairs[original_name]["no_types"] = is_compiled
    elif "_no_types" not in filename:
        file_pairs[filename]["original"] = is_compiled

# Counters for different cases
both_true = 0
both_false = 0
original_false_no_types_true = 0
original_true_no_types_false = 0

# Compute the required counts
failed_original_compiled_no_types = []

# Compute the required counts
for filename, pair in file_pairs.items():
    if pair["original"] is not None and pair["no_types"] is not None:
        if pair["original"] and pair["no_types"]:
            both_true += 1
        elif not pair["original"] and not pair["no_types"]:
            both_false += 1
        elif not pair["original"] and pair["no_types"]:
            original_false_no_types_true += 1
            failed_original_compiled_no_types.append(filename)
        elif pair["original"] and not pair["no_types"]:
            original_true_no_types_false += 1

# Print results
print("Both compiled: ", both_true)
print("Both failed compilation: ", both_false)
print("Original failed, no_types compiled: ", original_false_no_types_true)
print("Original compiled, no_types failed: ", original_true_no_types_false)

# Save the filenames where original failed but no_types compiled
"""with open("failed_original_compiled_no_types.json", "w") as f:
    json.dump(failed_original_compiled_no_types, f, indent=4)"""
    
