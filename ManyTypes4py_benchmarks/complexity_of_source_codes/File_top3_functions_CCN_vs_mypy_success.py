import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
	"""Load a JSON file and return its contents, or None on error."""
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as exc:  # noqa: BLE001
		print(f"Error loading {file_path}: {exc}")
		return None



def extract_top3_ccn(complexity_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	"""Extract a scalar metric from top_3_functions_CCN for each file.

	If the value is a list/tuple, use the sum of the top-3 CCNs.
	If the value is a number, use it as-is.
	Missing or invalid entries default to 0.0.
	"""
	result: Dict[str, Dict[str, Any]] = {}
	for filename, data in complexity_data.items():
		if not isinstance(data, dict):
			continue
		raw_top3 = data.get("top_3_functions_CCN", [])
		metric_value: float
		if isinstance(raw_top3, (list, tuple)) and len(raw_top3) > 0:
			# Keep only numeric values, compute sum
			values: List[float] = [float(v) for v in raw_top3 if isinstance(v, (int, float))]
			metric_value = float(np.sum(values)) if values else 0.0
		elif isinstance(raw_top3, (int, float)):
			metric_value = float(raw_top3)
		else:
			metric_value = 0.0

		result[filename] = {
			"top3_sum_CCN": metric_value,
			"top3_raw": raw_top3,
		}
	return result


def extract_mypy_results(mypy_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	"""Extract compilation status and error counts from mypy results."""
	results: Dict[str, Dict[str, Any]] = {}
	for filename, data in mypy_data.items():
		if not isinstance(data, dict):
			continue
		results[filename] = {
			"isCompiled": bool(data.get("isCompiled", False)),
			"error_count": int(data.get("error_count", 0)),
		}
	return results


def plot_top3_ccn_distribution(df: pd.DataFrame, model_name: str, out_dir: str) -> None:
	"""Create grouped bar chart of compile success vs failure by top-3 sum CCN bins."""
	plt.figure(figsize=(12, 8))

	# Create bins over the metric
	bins = np.linspace(df["top3_sum_CCN"].min(), df["top3_sum_CCN"].max(), 21) if len(df) else np.linspace(0, 1, 21)
	bin_centers = (bins[:-1] + bins[1:]) / 2

	success_percentages: List[float] = []
	failure_percentages: List[float] = []
	total_files_per_bin: List[int] = []

	for i in range(len(bins) - 1):
		bin_mask = (df["top3_sum_CCN"] >= bins[i]) & (df["top3_sum_CCN"] < bins[i + 1])
		bin_data = df[bin_mask]
		if len(bin_data) > 0:
			success_count = int(bin_data["isCompiled"].sum())
			total_files = int(len(bin_data))
			failure_count = total_files - success_count
			success_percentages.append((success_count / total_files) * 100.0)
			failure_percentages.append((failure_count / total_files) * 100.0)
			total_files_per_bin.append(total_files)
		else:
			success_percentages.append(0.0)
			failure_percentages.append(0.0)
			total_files_per_bin.append(0)

	x = np.arange(len(bin_centers))
	width = 0.35

	success_bars = plt.bar(x - width / 2, success_percentages, width, label="Compiled Successfully", color="green", alpha=0.7)
	failure_bars = plt.bar(x + width / 2, failure_percentages, width, label="Compilation Failed", color="red", alpha=0.7)

	for i, (success_bar, failure_bar, total_files) in enumerate(zip(success_bars, failure_bars, total_files_per_bin)):
		if total_files > 0:
			if success_bar.get_height() > 0:
				plt.text(
					success_bar.get_x() + success_bar.get_width() / 2,
					success_bar.get_height() + 0.5,
					f"{success_bar.get_height():.1f}%",
					ha="center",
					va="bottom",
					fontsize=8,
				)
			if failure_bar.get_height() > 0:
				plt.text(
					failure_bar.get_x() + failure_bar.get_width() / 2,
					failure_bar.get_height() + 0.5,
					f"{failure_bar.get_height():.1f}%",
					ha="center",
					va="bottom",
					fontsize=8,
				)
			plt.text(i, max(success_bar.get_height(), failure_bar.get_height()) + 2, f"n={total_files}", ha="center", va="bottom", fontsize=8, weight="bold")

	plt.xlabel("Top-3 Functions Sum CCN", fontsize=12)
	plt.ylabel("Percentage of Files", fontsize=12)
	plt.title(f"Compilation Success vs Failure by Top-3 Sum CCN Bins\n{model_name}", fontsize=14)
	plt.xticks(x, [f"{bin_centers[i]:.1f}" for i in range(len(bin_centers))], rotation=45)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	# Save under a subdirectory inside the main results directory
	file_safe_model = model_name.replace(" ", "_").replace("-", "_")
	out_path = os.path.join(out_dir, f"{file_safe_model}_top3CCN_sum_distribution_percentage.pdf")
	plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
	plt.close()


def main() -> None:
	# Input mappings
	# We reuse the Human complexity data for top-3 CCN across all model mypy results
	complexity_files = {
		"Human": "ManyTypes4py_benchmarks/complexity_of_source_codes/original_files_complexity_analysis.json",
	}

	mypy_files = {
		"GPT4O": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
		"O1-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
		"O3-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
		"Deepseek": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
		"Human": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json",
		"gpt35_2nd_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
		"claude3_sonnet_1st_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
	}

	# Output directory: create subdirectory for these plots
	base_out_dir = "ManyTypes4py_benchmarks/complexity_of_source_codes/complexity_analysis_results"
	sub_dir = os.path.join(base_out_dir, "top3_functions_CCN")
	os.makedirs(sub_dir, exist_ok=True)

	# Load complexity data once (Human)
	complexity_data = load_json_file(complexity_files["Human"])
	if not complexity_data:
		print("Error: Could not load complexity data")
		return

	top3_info = extract_top3_ccn(complexity_data)

	# Process each model's mypy results
	for model_name, mypy_path in mypy_files.items():
		print(f"\nProcessing {model_name}...")
		mypy_data = load_json_file(mypy_path)
		if not mypy_data:
			print(f"Skipping {model_name} due to missing mypy data")
			continue

		mypy_results = extract_mypy_results(mypy_data)

		# Join by filename
		rows: List[Dict[str, Any]] = []
		for filename, mypy_entry in mypy_results.items():
			if filename not in top3_info:
				continue
			rows.append(
				{
					"filename": filename,
					"top3_sum_CCN": top3_info[filename]["top3_sum_CCN"],
					"isCompiled": mypy_entry["isCompiled"],
					"error_count": mypy_entry["error_count"],
				}
			)

		if not rows:
			print(f"No matching data found for {model_name}")
			continue

		df = pd.DataFrame(rows)
		plot_top3_ccn_distribution(df, model_name, sub_dir)
		print(f"Saved plot for {model_name}")


if __name__ == "__main__":
	main()


