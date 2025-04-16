import json
import pandas as pd
import matplotlib.pyplot as plt

# Adjusted source code for barplot on mypy error count distribution

# Sample placeholder path for demonstration; replace with real ones if needed
file_paths = ["mypy_results_deepseek1.json","mypy_results_deepseek2.json"]

# Step 1: Load and merge data from all files
combined_data = {}
for path in file_paths:
    with open(path, "r") as f:
        partial_data = json.load(f)
        for k, v in partial_data.items():
            if not k.endswith("_no_types.py"):
                combined_data[k] = v

# Step 2: Categorize files into error count bins
error_bins = {
    "0": 0,
    "1–10": 0,
    "11–30": 0,
    "31–100": 0,
    ">100": 0
}

for entry in combined_data.values():
    error_count = entry.get("error_count", 0)
    if error_count == 0:
        error_bins["0"] += 1
    elif error_count <= 10:
        error_bins["1–10"] += 1
    elif error_count <= 30:
        error_bins["11–30"] += 1
    elif error_count <= 100:
        error_bins["31–100"] += 1
    else:
        error_bins[">100"] += 1

# Step 3: Barplot
labels = list(error_bins.keys())
values = list(error_bins.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, values, color='steelblue', edgecolor='black')
plt.title("Distribution of Files by Mypy Error Count", fontsize=14)
plt.xlabel("Mypy Error Count Range", fontsize=12)
plt.ylabel("Number of Files", fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.savefig("/mnt/data/error_count_distribution_barplot.pdf", format="pdf")
plt.show()
