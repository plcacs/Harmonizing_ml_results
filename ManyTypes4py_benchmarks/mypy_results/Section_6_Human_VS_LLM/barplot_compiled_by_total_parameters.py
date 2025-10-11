import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_compiled_and_params(
    untyped_path: str,
) -> Tuple[Set[str], Dict[str, int]]:
    data = load_json(untyped_path)
    baseline_files: Set[str] = set()
    filename_to_params: Dict[str, int] = {}
    for filename, info in data.items():
        if info.get("isCompiled") is True:
            baseline_files.add(filename)
            stats = info.get("stats", {})
            param_val = stats.get("total_parameters")
            if isinstance(param_val, int):
                filename_to_params[filename] = param_val
    return baseline_files, filename_to_params


def aggregate_counts_by_bin(
    llm_path: str,
    baseline_files: Set[str],
    filename_to_params: Dict[str, int],
    bin_edges: List[Tuple[int, int]],
) -> Dict[str, int]:
    def label_for(value: int) -> str | None:
        for start, end in bin_edges:
            if start <= value <= end:
                return f"{start}-{end}"
        return None

    data = load_json(llm_path)
    counts: Dict[str, int] = defaultdict(int)
    for filename in baseline_files:
        info = data.get(filename)
        if not info:
            continue
        if info.get("isCompiled") is True:
            param_val = filename_to_params.get(filename)
            if isinstance(param_val, int):
                label = label_for(param_val)
                if label is not None:
                    counts[label] += 1
    return counts


def aggregate_counts_by_bin_multi(
    llm_paths: List[str],
    baseline_files: Set[str],
    filename_to_params: Dict[str, int],
    bin_edges: List[Tuple[int, int]],
    mode: str,
) -> Dict[str, int]:
    """
    Aggregate counts per bin across multiple LLM result files using a set operation.

    mode: "union" counts if any of the provided LLMs compiled the file,
          "intersection" counts only if all provided LLMs compiled the file.
    """
    datasets = [load_json(p) for p in llm_paths]

    def in_bin_label(value: int) -> str | None:
        for start, end in bin_edges:
            if start <= value <= end:
                return f"{start}-{end}"
        return None

    counts: Dict[str, int] = defaultdict(int)
    for filename in baseline_files:
        compiled_flags: List[bool] = []
        for data in datasets:
            info = data.get(filename)
            compiled_flags.append(bool(info and info.get("isCompiled") is True))

        ok = False
        if mode == "union":
            ok = any(compiled_flags)
        elif mode == "intersection":
            ok = all(compiled_flags) and len(compiled_flags) > 0

        if not ok:
            continue

        param_val = filename_to_params.get(filename)
        if isinstance(param_val, int):
            label = in_bin_label(param_val)
            if label is not None:
                counts[label] += 1

    return counts


def plot_grouped_bars(
    bin_labels: List[str],
    series: List[Tuple[str, List[int]]],
) -> None:
    x = range(len(bin_labels))
    num_series = len(series)
    total_width = 0.82
    bar_width = total_width / max(1, num_series)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_series)]

    plt.figure(figsize=(14, 7))
    bars_by_series = []
    
    # Consistent colors across plots
    color_map = {
        # Core labels used in this script
        "Human (original)": "pink",
        "o3-mini": "red",
        "deepseek": "blue",
        "claude3 sonnet": "purple",
        "Union of 3 LLMs": "#8B5A96",  # keep distinct color for union
        # Also support alternative label spellings seen elsewhere
        "Human": "pink",
        "claude3-sonnet": "purple",
        "gpt-4o": "orange",
        "o1-mini": "skyblue",
        "gpt-3.5": "green",
    }
    
    for idx, ((label, counts), dx) in enumerate(zip(series, offsets)):
        positions = [i + dx for i in x]
        color = color_map.get(label, "#666666")  # Default gray if not found
        bars = plt.bar(positions, counts, width=bar_width, label=label, color=color)
        bars_by_series.append(bars)

    # Value labels for legibility (only annotate small numbers of bars)
    total_bars = len(bin_labels) * num_series
    if total_bars <= 150:
        for bars in bars_by_series:
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    plt.annotate(
                        f"{int(height)}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

    plt.xticks(list(x), bin_labels, rotation=30, ha="right")
    plt.xlabel("Parameter count (from untyped baseline)",fontsize=16)
    plt.ylabel("Number of files typechecked successfully",fontsize=16)
    # plt.title("Compiled successes by parameter count across LLMs (baseline filtered)",fontsize=20)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.savefig("Section_5_LLM_VS_LLM/compiled_counts_by_total_parameters.pdf", bbox_inches="tight")
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
    bars_by_series = []
    
    # Consistent colors across plots
    color_map = {
        # Core labels used in this script
        "Human (original)": "pink",
        "o3-mini": "red",
        "deepseek": "blue",
        "claude3 sonnet": "purple",
        "Union of 3 LLMs": "#8B5A96",  # keep distinct color for union
        # Also support alternative label spellings seen elsewhere
        "Human": "pink",
        "claude3-sonnet": "purple",
        "gpt-4o": "orange",
        "o1-mini": "skyblue",
        "gpt-3.5": "green",
    }
    
    for idx, ((label, percents), dx) in enumerate(zip(series_percent, offsets)):
        positions = [i + dx for i in x]
        color = color_map.get(label, "#666666")  # Default gray if not found
        bars = plt.bar(positions, percents, width=bar_width, label=label, color=color)
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
                        fontsize=12,
                    )

    plt.xticks(list(x), bin_labels, rotation=30, ha="right",fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlabel("Parameter count bins (from untyped baseline)",fontsize=16)
    plt.ylabel("Percentage of files typechecked successfully",fontsize=16)
    # plt.title("Compiled success rate by parameter count across LLMs (baseline filtered)",fontsize=20)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Section_6_Human_VS_LLM/compiled_percent_by_total_parameters.pdf", bbox_inches="tight")
    plt.show()


def main() -> None:
    # Paths are expected to be relative to the working directory where the script is run
    untyped_path = "mypy_outputs/mypy_results_untyped_with_errors.json"

    llm_paths = {
        "Human (original)": "mypy_outputs/mypy_results_original_files_with_errors.json",
        "o3-mini": "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "deepseek": "mypy_outputs/mypy_results_deepseek_with_errors.json",
        "claude3 sonnet": "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    }

    baseline_files, filename_to_params = build_baseline_compiled_and_params(untyped_path)

    # Build custom bins: 0-0 if present, 1–150 (step 10), 151–500 (step 50), 501+ (step 100)
    all_params = [p for f, p in filename_to_params.items() if f in baseline_files]
    if not all_params:
        print("No parameter data found in baseline compiled files.")
        return

    max_val = max(all_params)
    has_zero = any(p == 0 for p in all_params)

    def add_range_bins(edges: List[Tuple[int, int]], start: int, end: int, step: int) -> None:
        if end < start or step <= 0:
            return
        s = start
        while s <= end:
            e = min(s + step - 1, end)
            edges.append((s, e))
            s = e + 1

    bin_edges: List[Tuple[int, int]] = []
    #if has_zero:
    #    bin_edges.append((0, 0))

    if max_val >= 0:
        add_range_bins(bin_edges, 0, min(139, max_val), 20)
    if max_val >= 150:
        add_range_bins(bin_edges, 140, max_val,500)
    

    # Aggregate counts per bin for each LLM and compute baseline totals per bin
    temp_series: List[Tuple[str, Dict[str, int]]] = []

    series: List[Tuple[str, List[int]]] = []
    for label, path in llm_paths.items():
        counts_map = aggregate_counts_by_bin(path, baseline_files, filename_to_params, bin_edges)
        temp_series.append((label, counts_map))

    # Create union of the 3 LLMs (o3-mini, deepseek, claude3 sonnet)
    llm_union_paths = [
        llm_paths["o3-mini"],
        llm_paths["deepseek"], 
        llm_paths["claude3 sonnet"]
    ]
    union_counts = aggregate_counts_by_bin_multi(
        llm_union_paths,
        baseline_files,
        filename_to_params,
        bin_edges,
        mode="union"
    )
    temp_series.append(("Union of 3 LLMs", union_counts))

    # Ensure a fixed plotting order with Human first
    desired_order = [
        "Human (original)",
        "o3-mini",
        "deepseek",
        "claude3 sonnet",
        "Union of 3 LLMs",
    ]
    order_index = {name: i for i, name in enumerate(desired_order)}
    temp_series.sort(key=lambda item: order_index.get(item[0], len(desired_order)))

    bin_labels = [f"{s}-{e}" for s, e in bin_edges]

    # Baseline totals per bin for percentage denominator
    baseline_counts_map: Dict[str, int] = defaultdict(int)
    for filename in baseline_files:
        p = filename_to_params.get(filename)
        if not isinstance(p, int):
            continue
        # find label
        for s, e in bin_edges:
            if s <= p <= e:
                baseline_counts_map[f"{s}-{e}"] += 1
                break

    # Filter out bins which are zero for all series to declutter
    non_empty_mask = []
    for lbl in bin_labels:
        total = sum(cm.get(lbl, 0) for _, cm in temp_series)
        non_empty_mask.append(total > 0)

    filtered_labels: List[str] = [lbl for lbl, keep in zip(bin_labels, non_empty_mask) if keep]
    if not filtered_labels:
        # If everything is empty, keep original labels to show empty chart context
        filtered_labels = bin_labels

    for label, counts_map in temp_series:
        series.append((label, [counts_map.get(lbl, 0) for lbl in filtered_labels]))

    # Enhance first plot labels with baseline bin sizes
    display_labels_with_n = [f"{lbl} (n={baseline_counts_map.get(lbl, 0)})" for lbl in filtered_labels]
    #plot_grouped_bars(display_labels_with_n, series)

    # Percentage plot
    series_percent: List[Tuple[str, List[float]]] = []
    for label, counts_map in temp_series:
        percents: List[float] = []
        for lbl in filtered_labels:
            denom = baseline_counts_map.get(lbl, 0)
            num = counts_map.get(lbl, 0)
            percents.append((num * 100.0 / denom) if denom else 0.0)
        series_percent.append((label, percents))

    plot_grouped_bars_percent(display_labels_with_n, series_percent)


if __name__ == "__main__":
    main()


