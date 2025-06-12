import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# File paths
paths = {
    "GPT-4o": "mypy_results_gpt4o_with_errors.json",
    "O1-mini": "mypy_results_o1-mini_with_errors.json",
    "DeepSeek": "mypy_results_deep_seek_with_errors.json"
}

# Load base model data
with open("mypy_results_untyped_benchmarks_with_errors.json", "r") as f:
    base_data = json.load(f)

# Custom type coverage bins and labels
custom_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30),
               (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
               (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]

custom_labels = [
    "0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
]

# Models to evaluate
models = ["GPT-4o", "O1-mini", "DeepSeek"]
model_bin_counts = {model: defaultdict(int) for model in models}
bin_stats = {label: {"total_params": [], "annotated_params": []} for label in custom_labels}

# Store filenames for DeepSeek with 0-5% coverage
deepseek_low_coverage_files = []
def has_syntax_error(errors):
    return any("syntax" in error.lower() for error in errors)
# Load data and calculate counts
for model in models:
    path = paths[model]
    with open(path, "r") as f:
        data = json.load(f)

    error_zero_count = 0
    error_nonzero_count = 0
    for file_path, file_data in data.items():
        base_error_count = base_data.get(file_path, {}).get("error_count", 0)
        error_count = file_data.get("error_count", 0) if not has_syntax_error(file_data.get('errors', [])) else 0
        
        if base_error_count == 0 and error_count > 0:
            error_nonzero_count += 1
        else:
            error_zero_count += 1

    print(f"{model}: base_error=0 & model_error>0: {error_nonzero_count}, others: {error_zero_count}")

    for file_path, file_data in data.items():
        base_error_count = base_data.get(file_path, {}).get("error_count", 0)
        if base_error_count > 0:
            continue

        stats = file_data.get("stats", {})
        total = stats.get("total_parameters", 0)
        annotated = stats.get("parameters_with_annotations", 0)
        error_count = file_data.get("error_count", 0) if not has_syntax_error(file_data.get('errors', [])) else 0

        if error_count == 0:
            continue

        coverage = annotated / total if total > 0 else 0
        for i, (low, high) in enumerate(custom_bins):
            if low <= coverage < high:
                model_bin_counts[model][custom_labels[i]] += 1
                bin_stats[custom_labels[i]]["total_params"].append(total)
                bin_stats[custom_labels[i]]["annotated_params"].append(annotated)
                # Store DeepSeek filenames with 0-5% coverage
                if model == "DeepSeek" and i == 0:
                    deepseek_low_coverage_files.append(file_path)
                break

# Save DeepSeek filenames with 0-5% coverage
"""with open("deepseek_low_coverage_files.txt", "w") as f:
    for file_path in deepseek_low_coverage_files:  
        f.write(f"{file_path}\n")"""

# Print bin counts and totals for each model
print("\nBin counts for each model:")
print("Bin\t\tGPT-4o\tO1-mini\tDeepSeek")
print("-" * 50)
for label in custom_labels:
    counts = [model_bin_counts[model].get(label, 0) for model in models]
    print(f"{label}\t\t{counts[0]}\t{counts[1]}\t{counts[2]}")

print("\nTotal instances per model:")
for model in models:
    total = sum(model_bin_counts[model].values())
    print(f"{model}: {total}")

"""# Print average statistics for each bin
print("\nAverage Statistics per Coverage Bin:")
print("Bin\t\tAvg Total Params\tAvg Annotated Params")
print("-" * 50)
for label in custom_labels:
    total_avg = np.mean(bin_stats[label]["total_params"]) if bin_stats[label]["total_params"] else 0
    annotated_avg = np.mean(bin_stats[label]["annotated_params"]) if bin_stats[label]["annotated_params"] else 0
    print(f"{label}\t\t{total_avg:.1f}\t\t{annotated_avg:.1f}")"""

# Plot grouped bar chart
x = np.arange(len(custom_labels))
width = 0.25

plt.figure(figsize=(14, 6))
for i, model in enumerate(models):
    y_vals = [model_bin_counts[model].get(label, 0) for label in custom_labels]
    plt.bar(x + i * width, y_vals, width=width, label=model)

plt.xticks(x + width, custom_labels, rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Type Coverage Bins", fontsize=20)
plt.ylabel("Number of Files with error_count>0", fontsize=20)
plt.title("Type Coverage vs. Files with Errors (Pyperformance)",fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig("TypeCoverage_vs_mypy_error_Pyperformance.pdf", bbox_inches='tight')
#plt.show()