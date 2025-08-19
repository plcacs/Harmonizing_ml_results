import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
 


def load_json(path: str) -> Optional[Dict[str, Any]]:
	"""Read a JSON file and return its contents or None on failure."""
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as exc:  # noqa: BLE001
		print(f"Failed to load {path}: {exc}")
		return None


def extract_top3_sum_ccn(complexity: Dict[str, Any]) -> Dict[str, float]:
	"""Map filename -> sum(top_3_functions_CCN). Missing/invalid -> 0.0"""
	result: Dict[str, float] = {}
	for filename, metrics in complexity.items():
		values = metrics.get("top_3_functions_CCN", [])
		if isinstance(values, (list, tuple)):
			numeric = [float(v) for v in values if isinstance(v, (int, float))]
			result[filename] = float(np.sum(numeric)) if numeric else 0.0
		elif isinstance(values, (int, float)):
			result[filename] = float(values)
		else:
			result[filename] = 0.0
	return result


def build_dataframe(
	complexity_json_path: str,
	mypy_results_json_path: str,
) -> pd.DataFrame:
	"""Join complexity and mypy stats on filename and assemble model-ready DataFrame."""
	complexity = load_json(complexity_json_path)
	if not complexity:
		raise RuntimeError("Could not load complexity data")

	mypy = load_json(mypy_results_json_path)
	if not mypy:
		raise RuntimeError("Could not load mypy results data")

	top3_sum_map = extract_top3_sum_ccn(complexity)
	rows: List[Dict[str, Any]] = []
	for filename, result in mypy.items():
		stats = result.get("stats", {}) if isinstance(result, dict) else {}
		if filename not in top3_sum_map:
			continue
		rows.append(
			{
				"filename": filename,
				"top3_sum_CCN": float(top3_sum_map[filename]),
				"total_parameters": int(stats.get("total_parameters", 0) or 0),
				"parameters_with_annotations": int(stats.get("parameters_with_annotations", 0) or 0),
				"isCompiled": bool(result.get("isCompiled", False)),
			}
		)

	df = pd.DataFrame(rows)
	if df.empty:
		raise RuntimeError("No overlapping filenames between complexity and mypy results")

	# Derived feature
	df["annotation_ratio"] = df.apply(
		lambda r: (r["parameters_with_annotations"] / r["total_parameters"]) if r["total_parameters"] > 0 else 0.0,
		axis=1,
	)
	return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute Pearson (point-biserial) and Spearman correlations vs isCompiled for key predictors."""
	metrics = [
		("top3_sum_CCN", "top3_sum_CCN"),
		("parameters_with_annotations", "parameters_with_annotations"),
		("total_parameters", "total_parameters"),
		("annotation_ratio", "annotation_ratio"),
	]

	compiled = df["isCompiled"].astype(int)
	rows: List[Dict[str, Any]] = []
	for metric_key, label in metrics:
		series = df[metric_key].astype(float)
		# Guard against constant or empty series
		pearson_val = float("nan")
		spearman_val = float("nan")
		if series.nunique(dropna=True) > 1 and compiled.nunique(dropna=True) > 1:
			pearson_val = float(series.corr(compiled, method="pearson"))
			spearman_val = float(series.corr(compiled, method="spearman"))
		rows.append({
			"metric": label,
			"pearson_pointbiserial": pearson_val,
			"spearman": spearman_val,
			"n": int(len(df)),
		})

	return pd.DataFrame(rows)


def main() -> None:
	# Inputs (relative paths)
	complexity_json_path = "ManyTypes4py_benchmarks/complexity_of_source_codes/original_files_complexity_analysis.json"
	# All mypy result files to process
	mypy_files = {
		"GPT4O": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
		"O1-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
		"O3-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
		"Deepseek": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
		"Human": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json",
		"gpt35_2nd_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
		"claude3_sonnet_1st_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
	}

	# Output directory (correlation summaries)
	base_out_dir = "ManyTypes4py_benchmarks/complexity_of_source_codes/complexity_analysis_results"
	correlation_dir = os.path.join(base_out_dir, "correlation_results")
	os.makedirs(correlation_dir, exist_ok=True)

	all_results: List[pd.DataFrame] = []
	for model_label, mypy_results_json_path in mypy_files.items():
		try:
			# Prepare data
			df = build_dataframe(complexity_json_path, mypy_results_json_path)
			# Correlations per model
			corr_df = compute_correlations(df)
			corr_df.insert(0, "model", model_label)
			# Save per-model CSV
			per_model_csv = os.path.join(correlation_dir, f"{model_label}_correlations.csv")
			corr_df.to_csv(per_model_csv, index=False)
			print(f"Saved correlations CSV to: {per_model_csv}")
			all_results.append(corr_df)
		except Exception as exc:  # noqa: BLE001
			print(f"Skipping {model_label} due to error: {exc}")

	# Combined CSV across models
	if all_results:
		combined = pd.concat(all_results, ignore_index=True)
		combined_csv = os.path.join(correlation_dir, "all_models_correlations.csv")
		combined.to_csv(combined_csv, index=False)
		print(f"Saved combined correlations CSV to: {combined_csv}")


if __name__ == "__main__":
	main()


