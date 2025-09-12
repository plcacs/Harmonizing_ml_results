import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt

# Custom color map for consistent coloring across plots
color_map = {
    # Core labels used in this script
    "o3-mini": "red",
    "deepseek": "blue",
    "claude3 sonnet": "purple",
    "Union(o3-mini, deepseek, claude)": "#8B5A96",  # purple for first union
    "Union(o3-mini, o1-mini, gpt-4o, gpt-3.5)": "#FF8C00",  # dark orange for second union
    "Intersection(o3-mini, deepseek, claude)": "#FF6B6B",  # distinct color for intersection
    # Also support alternative label spellings seen elsewhere
    "Human": "pink",
    "claude3-sonnet": "purple",
    "gpt-4o": "orange",
    "o1-mini": "skyblue",
    "gpt-3.5": "green",  # lime green
}


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
    for idx, ((label, percents), dx) in enumerate(zip(series_percent, offsets)):
        positions = [i + dx for i in x]
        # Use custom color map, fallback to default if not found
        color = color_map.get(label, plt.get_cmap("tab10")(idx % 10))
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
                        fontsize=8,
                    )

    plt.xticks(list(x), bin_labels, rotation=30, ha="right", fontsize=14)
    plt.xlabel("Parameter count bins (from untyped baseline)", fontsize=18)
    plt.ylabel("Percentage of files typechecked successfully", fontsize=18)

    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
   
    plt.tight_layout()
    plt.savefig(
        "Section_05/compiled_percent_by_total_parameters_pyperformance.pdf",
        bbox_inches="tight",
    )
    plt.show()


def main() -> None:
    # Paths are expected to be relative to the working directory where the script is run
    untyped_path = "mypy_outputs/mypy_results_untyped_with_errors.json"

    llm_paths = {
        "gpt-3.5": "mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
        "gpt-4o": "mypy_outputs/mypy_results_gpt4o_1st_run_with_errors.json",
        "o1-mini": "mypy_outputs/mypy_results_o1_mini_1st_run_with_errors.json",
        "o3-mini": "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "deepseek": "mypy_outputs/mypy_results_deepseek_1st_run_with_errors.json",
        "claude3 sonnet": "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    }

    baseline_files, filename_to_params = build_baseline_compiled_and_params(
        untyped_path
    )

    # Build custom bins: 0-0 if present, 1–150 (step 10), 151–500 (step 50), 501+ (step 100)
    all_params = [p for f, p in filename_to_params.items() if f in baseline_files]
    if not all_params:
        print("No parameter data found in baseline compiled files.")
        return

    max_val = max(all_params)
    has_zero = any(p == 0 for p in all_params)

    def add_range_bins(
        edges: List[Tuple[int, int]], start: int, end: int, step: int
    ) -> None:
        if end < start or step <= 0:
            return
        s = start
        while s <= end:
            e = min(s + step - 1, end)
            edges.append((s, e))
            s = e + 1

    bin_edges: List[Tuple[int, int]] = []
    # Granular until 10, then a single bin for >10
    candidate_bins: List[Tuple[int, int]] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 10),
    ]
    for start, end in candidate_bins:
        if start <= max_val:
            bin_edges.append((start, min(end, max_val)))
    if max_val > 10:
        bin_edges.append((11, max_val))

    # Aggregate counts per bin for each LLM and compute baseline totals per bin
    temp_series: List[Tuple[str, Dict[str, int]]] = []

    series: List[Tuple[str, List[int]]] = []
    for label, path in llm_paths.items():
        counts_map = aggregate_counts_by_bin(
            path, baseline_files, filename_to_params, bin_edges
        )
        temp_series.append((label, counts_map))

    # Add requested aggregate series
    trio_for_union_intersection = [
        llm_paths["o3-mini"],
        llm_paths["deepseek"],
        llm_paths["claude3 sonnet"],
    ]
    union_trio = aggregate_counts_by_bin_multi(
        trio_for_union_intersection,
        baseline_files,
        filename_to_params,
        bin_edges,
        mode="union",
    )
    intersection_trio = aggregate_counts_by_bin_multi(
        trio_for_union_intersection,
        baseline_files,
        filename_to_params,
        bin_edges,
        mode="intersection",
    )
    trio_for_second_union = [
        llm_paths["o3-mini"],
        llm_paths["o1-mini"],
        llm_paths["gpt-4o"],
    ]
    union_trio_2 = aggregate_counts_by_bin_multi(
        trio_for_second_union,
        baseline_files,
        filename_to_params,
        bin_edges,
        mode="union",
    )
    temp_series.append(("Union(o3-mini, deepseek, claude)", union_trio))
    temp_series.append(("Intersection(o3-mini, deepseek, claude)", intersection_trio))
    temp_series.append(("Union(o3-mini, o1-mini, gpt-4o, gpt-3.5)", union_trio_2))

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

    filtered_labels: List[str] = [
        lbl for lbl, keep in zip(bin_labels, non_empty_mask) if keep
    ]
    if not filtered_labels:
        # If everything is empty, keep original labels to show empty chart context
        filtered_labels = bin_labels

    # Print counts per bin for each LLM/aggregate in readable terminal format
    print("Counts per bin (compiled files):")
    for lbl in bin_labels:
        print(f"- Bin {lbl}:")
        for label, counts_map in temp_series:
            print(f"  {label}: {counts_map.get(lbl, 0)}")

    for label, counts_map in temp_series:
        series.append((label, [counts_map.get(lbl, 0) for lbl in filtered_labels]))

    # Enhance first plot labels with baseline bin sizes
    display_labels_with_n = [
        f"{lbl} (n={baseline_counts_map.get(lbl, 0)})" for lbl in filtered_labels
    ]
    # plot_grouped_bars(display_labels_with_n, series)

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
