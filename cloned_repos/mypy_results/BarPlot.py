import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjusted source code for barplot on mypy error count distribution

file_paths = [
   
    "mypy_results_gpt4o_with_errors.json",
    "mypy_results_o1_mini_with_errors.json",
    "mypy_results_deepseek_with_errors.json"
]

# Step 1: Load data for each model separately and filter based on no_types condition
model_data = {}
no_types_data = {}

# First load no_types data
with open("mypy_results_no_type.json", "r") as f:
    no_types_data = json.load(f)

# Then load and filter other model data
for path in file_paths:
    model_name = path.replace("mypy_results_", "").replace("_with_errors.json", "")
    with open(path, "r") as f:
        data = json.load(f)
        # Only keep files where no_types has 0 errors but current model has errors
        model_data[model_name] = {
            k: v for k, v in data.items() 
            if not k.endswith("_no_types.py") 
            and k in no_types_data 
            and no_types_data[k].get("error_count", 0) == 0
            and v.get("error_count", 0) > 0
        }

# Step 2: Define error bins
error_bins = {
    "1-2": 0,
    "3-5": 0,
    "6-10": 0,
    "11-20": 0,
    "21-30": 0,
    "31-50": 0,
    ">50": 0
}

# Step 3: Calculate error distributions for each model
model_distributions = {}
for model, data in model_data.items():
    model_distributions[model] = error_bins.copy()
    for entry in data.values():
        error_count = entry.get("error_count", 0)
        if error_count <= 2:
            model_distributions[model]["1-2"] += 1
        elif error_count <= 5:
            model_distributions[model]["3-5"] += 1
        elif error_count <= 10:
            model_distributions[model]["6-10"] += 1
        elif error_count <= 20:
            model_distributions[model]["11-20"] += 1
        elif error_count <= 30:
            model_distributions[model]["21-30"] += 1
        elif error_count <= 50:
            model_distributions[model]["31-50"] += 1
        else:
            model_distributions[model][">50"] += 1

# Step 4: Create grouped bar plot
labels = list(error_bins.keys())
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 6))
colors = ['steelblue', 'darkorange', 'forestgreen']
for i, (model, distribution) in enumerate(model_distributions.items()):
    values = list(distribution.values())
    ax.bar(x + i*width, values, width, label=model, color=colors[i], edgecolor='black')

ax.set_title("Distribution of Files by Mypy Error Count\n(Only files with no errors in no_types version)", fontsize=14)
ax.set_xlabel("Mypy Error Count Range", fontsize=12)
ax.set_ylabel("Number of Files", fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.savefig("/mnt/data/error_count_distribution_barplot.pdf", format="pdf")
plt.show()
