import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Hardcoded list of files
#file_paths = ["mypy_results_deepseek1.json", "mypy_results_deepseek2.json"]
#file_paths = ["mypy_results_o1_mini1.json", "mypy_results_o1_mini2.json","mypy_results_o1_mini3.json"]
file_paths = ["mypy_results_gpt4o1.json", "mypy_results_gpt4o2.json"]
# Step 2: Merge data from all files, excluding *_no_types.py
combined_data = {}
for path in file_paths:
    with open(path, "r") as f:
        partial_data = json.load(f)
        for k, v in partial_data.items():
            if not k.endswith("_no_types.py"):
                combined_data[k] = v

# Step 3: Extract metrics
records = []
for filename, entry in combined_data.items():
    total = entry["stats"]["total_parameters"]
    annotated = entry["stats"]["parameters_with_annotations"]
    error_count = entry["error_count"]
    if total > 0:
        coverage = annotated / total
        records.append({
            "file": filename,
            "coverage": coverage,
            "error_count": error_count
        })

df = pd.DataFrame(records)

# Step 4: Plotting
plt.style.use("default")
plt.figure(figsize=(8, 6))
plt.scatter(df["coverage"], df["error_count"], color='green', edgecolors='black', alpha=0.7)
plt.title("Type Coverage vs. Type Safety (mypy error count)", fontsize=14)
plt.xlabel("Type Coverage (annotated / total parameters)", fontsize=12)
plt.ylabel("Mypy Error Count", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("type_coverage_vs_safety_combined_gpt4o.pdf", format="pdf")
plt.show()

