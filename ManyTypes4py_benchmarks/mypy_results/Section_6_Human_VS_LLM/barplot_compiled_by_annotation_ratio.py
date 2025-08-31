import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_compiled_files(untyped_path: str) -> Set[str]:
    data = load_json(untyped_path)
    baseline_files: Set[str] = set()
    for filename, info in data.items():
        if info.get("isCompiled") is True:
            baseline_files.add(filename)
    return baseline_files


def aggregate_counts_by_bin(
    llm_path: str,
    baseline_files: Set[str],
    bin_edges: List[Tuple[int, int]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    def label_for_percent(percent_value: float) -> str | None:
        for start, end in bin_edges:
            if (start <= percent_value < end) or (end == 100 and start <= percent_value <= end):
                return f"{start}-{end}%"
        return None

    data = load_json(llm_path)
    compiled_counts: Dict[str, int] = defaultdict(int)
    total_counts: Dict[str, int] = defaultdict(int)
    
    # Step 2: Only consider files that exist in baseline
    for filename in baseline_files:
        info = data.get(filename)
        if not info:
            continue
            
        # Step 3: Calculate annotation ratio from LLM file
        stats = info.get("stats", {})
        total_params = stats.get("total_parameters", 0)
        annotated_params = stats.get("parameters_with_annotations", 0)
        
        if total_params > 0:
            ratio = annotated_params / total_params
            percent_val = ratio * 100.0
            
            label = label_for_percent(percent_val)
            if label is not None:
                total_counts[label] += 1
                # Step 4: Count compiled files per bin
                if info.get("isCompiled") is True:
                    compiled_counts[label] += 1
    
    return compiled_counts, total_counts


def plot_grouped_bars(
    bin_labels: List[str],
    series: List[Tuple[str, List[int]]],
    total_series: List[Tuple[str, List[int]]],
) -> None:
    x = range(len(bin_labels))
    num_series = len(series)
    total_width = 0.82
    bar_width = total_width / max(1, num_series)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_series)]

    plt.figure(figsize=(14, 7))
    color_cycle = plt.get_cmap("tab10")
    bars_by_series = []
    for idx, ((label, counts), dx) in enumerate(zip(series, offsets)):
        positions = [i + dx for i in x]
        bars = plt.bar(positions, counts, width=bar_width, label=label, color=color_cycle(idx % 10))
        bars_by_series.append(bars)

    total_bars = len(bin_labels) * num_series
    if total_bars <= 150:
        for bars, (_, totals) in zip(bars_by_series, total_series):
            for rect, total in zip(bars, totals):
                height = rect.get_height()
                if height > 0:
                    plt.annotate(
                        f"{total}/{int(height)}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.xticks(list(x), bin_labels, rotation=30, ha="right")
    plt.xlabel("Annotation ratio", fontsize=16)
    plt.ylabel("Number of files typechecked successfully", fontsize=16)
    plt.title("Compiled successes by annotation ratio across LLMs (baseline filtered)", fontsize=20)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig("compiled_counts_by_annotation_ratio.pdf", bbox_inches="tight")
    plt.show()


def plot_grouped_bars_percent(
    bin_labels: List[str],
    series_percent: List[Tuple[str, List[float]]],
) -> None:
    x = range(len(bin_labels))
    num_series = len(series_percent)
    total_width = 0.82
    bar_width = total_width / max(1, num_series)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_series)]

    plt.figure(figsize=(14, 7))
    color_cycle = plt.get_cmap("tab10")
    bars_by_series = []
    for idx, ((label, percents), dx) in enumerate(zip(series_percent, offsets)):
        positions = [i + dx for i in x]
        bars = plt.bar(positions, percents, width=bar_width, label=label, color=color_cycle(idx % 10))
        bars_by_series.append(bars)

    total_bars = len(bin_labels) * num_series
    if total_bars <= 150:
        for bars in bars_by_series:
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    plt.annotate(
                        f"{height:.0f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.xticks(list(x), bin_labels, rotation=30, ha="right")
    plt.xlabel("Annotation ratio", fontsize=16)
    plt.ylabel("Percentage of files typechecked successfully", fontsize=16)
    plt.title("Compiled success rate by annotation ratio across LLMs (baseline filtered)", fontsize=20)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig("compiled_percent_by_annotation_ratio.pdf", bbox_inches="tight")
    plt.show()


def main() -> None:
    # Paths are expected to be relative to the working directory where the script is run
    untyped_path = "../mypy_outputs/mypy_results_untyped_with_errors.json"

    llm_paths = {
        "Human (original)": "../mypy_outputs/mypy_results_original_with_errors.json",
        "o3-mini": "../mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "deepseek": "../mypy_outputs/mypy_results_deepseek_with_errors.json",
        "claude3 sonnet": "../mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        "Union of 3 LLMs": "../mypy_outputs/mypy_results_union_three_llms_with_errors.json",
    }

    baseline_files = build_baseline_compiled_files(untyped_path)

    # Create bins: 0-10%, 11-20%, 21-30%, ..., 91-100%
    bin_edges: List[Tuple[int, int]] = []
    
    # Create 10% bins from 0 to 100
    for start in range(0, 100, 10):
        end = start + 10
        bin_edges.append((start, end))

    # Aggregate counts per bin for each LLM
    temp_series: List[Tuple[str, Dict[str, int]]] = []
    temp_total_series: List[Tuple[str, Dict[str, int]]] = []
    for label, path in llm_paths.items():
        compiled_counts, total_counts = aggregate_counts_by_bin(path, baseline_files, bin_edges)
        temp_series.append((label, compiled_counts))
        temp_total_series.append((label, total_counts))

    bin_labels = [f"{s}-{e}%" for s, e in bin_edges]

    # Baseline totals per bin for percentage denominator (using first LLM as reference)
    baseline_counts_map: Dict[str, int] = defaultdict(int)
    first_llm_path = list(llm_paths.values())[0]
    first_llm_data = load_json(first_llm_path)
    
    for filename in baseline_files:
        info = first_llm_data.get(filename)
        if not info:
            continue
            
        # Calculate annotation ratio from first LLM file
        stats = info.get("stats", {})
        total_params = stats.get("total_parameters", 0)
        annotated_params = stats.get("parameters_with_annotations", 0)
        
        if total_params > 0:
            ratio = annotated_params / total_params
            p = ratio * 100.0
            
            # Find which bin this file belongs to
            for s, e in bin_edges:
                if (s <= p < e) or (e == 100 and s <= p <= e):
                    baseline_counts_map[f"{s}-{e}%"] += 1
                    break

    # Filter out bins which are zero for all series to declutter
    non_empty_mask = []
    for lbl in bin_labels:
        total = sum(cm.get(lbl, 0) for _, cm in temp_series)
        non_empty_mask.append(total > 0)

    filtered_labels: List[str] = [lbl for lbl, keep in zip(bin_labels, non_empty_mask) if keep]
    if not filtered_labels:
        filtered_labels = bin_labels

    series: List[Tuple[str, List[int]]] = []
    total_series: List[Tuple[str, List[int]]] = []
    for label, counts_map in temp_series:
        series.append((label, [counts_map.get(lbl, 0) for lbl in filtered_labels]))
    for label, total_map in temp_total_series:
        total_series.append((label, [total_map.get(lbl, 0) for lbl in filtered_labels]))

    display_labels_with_n = [f"{lbl}" for lbl in filtered_labels]
    plot_grouped_bars(display_labels_with_n, series, total_series)

    # Percentage plot
    series_percent: List[Tuple[str, List[float]]] = []
    for label, counts_map in temp_series:
        percents: List[float] = []
        for lbl in filtered_labels:
            total = total_series[temp_series.index((label, counts_map))][1][filtered_labels.index(lbl)]
            num = counts_map.get(lbl, 0)
            percents.append((num * 100.0 / total) if total > 0 else 0.0)
        series_percent.append((label, percents))

    plot_grouped_bars_percent(filtered_labels, series_percent)


if __name__ == "__main__":
    main()
