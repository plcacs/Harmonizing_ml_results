import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# File paths
paths = {
    "GPT-4o": "mypy_results_gpt4o_with_errors.json",
    "O1-mini": "mypy_results_o1_mini_with_errors.json",
    "DeepSeek": "mypy_results_deepseek_with_errors.json",
}

# Load base model data
with open("mypy_results_no_type.json", "r") as f:
    base_data = json.load(f)

# Custom type coverage bins and labels
custom_bins = [
    (0, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01),
]

custom_labels = [
    "0-5%",
    "5-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]


def has_syntax_error(errors):
    return any(error_type in error.lower() for error in errors for error_type in ["syntax", "empty_body", "name_defined"])


# Models to evaluate
models = ["GPT-4o", "O1-mini", "DeepSeek"]
bin_stats = {
    model: {label: {"error_count": 0, "no_error_count": 0} for label in custom_labels}
    for model in models
}

# Calculate statistics for each model and bin
for model in models:
    path = paths[model]
    with open(path, "r") as f:
        data = json.load(f)

    for file_path, file_data in data.items():
        base_error_count = base_data.get(file_path, {}).get("error_count", 0)
        if base_error_count > 0:
            continue

        stats = file_data.get("stats", {})
        total = stats.get("total_parameters", 0)
        annotated = stats.get("parameters_with_annotations", 0)
        error_count = (
            file_data.get("error_count", 0)
            if not has_syntax_error(file_data.get("errors", []))
            else 0
        )

        coverage = annotated / total if total > 0 else 0
        for i, (low, high) in enumerate(custom_bins):
            if low <= coverage < high:
                if error_count > 0:
                    bin_stats[model][custom_labels[i]]["error_count"] += 1
                else:
                    bin_stats[model][custom_labels[i]]["no_error_count"] += 1
                break

# Print statistics for each model
print("\nBin Statistics:")
for model in models:
    print(f"\n{model}:")
    for label in custom_labels:
        stats = bin_stats[model][label]
        total = stats['error_count'] + stats['no_error_count']
        ratio = (stats['error_count'] / total * 100) if total > 0 else 0
        print(f"{label}: Errors={stats['error_count']}, No Errors={stats['no_error_count']}, Ratio={ratio:.1f}%")

# Calculate error ratios and plot
plt.figure(figsize=(14, 6))
x = np.arange(len(custom_labels))
width = 0.25

for i, model in enumerate(models):
    ratios = []
    for label in custom_labels:
        stats = bin_stats[model][label]
        total = stats["error_count"] + stats["no_error_count"]
        ratio = (stats["error_count"] / total * 100) if total > 0 else 0
        ratios.append(ratio)

    plt.bar(x + i * width, ratios, width=width, label=model)

plt.xticks(x + width, custom_labels, rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Type Coverage Bins", fontsize=20)
plt.ylabel("Error Ratio (%)", fontsize=20)
plt.title("Type Coverage vs. Error Ratio (ManyTypes4Py)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig("TypeCoverage_vs_error_ratio_ManyTypes4Py.pdf", bbox_inches="tight")
#plt.show()
