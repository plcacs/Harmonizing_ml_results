# Re-import necessary libraries and reprocess after kernel reset
import json
import pandas as pd
from collections import defaultdict

# Load JSON files again
file1_path = "o1-mini_syntactic_features_code_similarity_old.json"
file2_path = "o1_mini_syntactic_features_code_similarity_new.json"

with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Convert each feature set into a frozenset and use it as a key
feature_map = defaultdict(lambda: {"old": [], "new": []})

for path, features in data1.items():
    key = frozenset(features)
    feature_map[key]["old"].append(path)

for path, features in data2.items():
    key = frozenset(features)
    feature_map[key]["new"].append(path)

# Only retain those that appear in both groups
common_feature_map = {
    tuple(sorted(key)): {
        "old": v["old"],
        "new": v["new"]
    }
    for key, v in feature_map.items()
    if v["old"] and v["new"]
}

# Reformat for example-style output
example_formatted = {}
for feature_key, paths in common_feature_map.items():
    feature_strs = sorted(list(feature_key))
    example_formatted[frozenset(feature_strs)] = {
        "old": paths["old"],
        "new": paths["new"],
    }

# Write to file
example_output_path = "feature_group_matches_o1_mini.json"
with open(example_output_path, "w") as f:
    json.dump({",".join(sorted(k)): v for k, v in example_formatted.items()}, f, indent=2)

  
