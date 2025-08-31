import csv
from collections import defaultdict
from typing import Dict, List, Tuple
from statistics import mean

import matplotlib.pyplot as plt

COUNTS_CSV = "./winner_group_counts.csv"
COUNTS_NORM_CSV = "./winner_group_counts_normalized.csv"

RAW_PDF = "./winner_group_counts.pdf"
NORM_PDF = "./winner_group_counts_normalized.pdf"

# New inputs/outputs for precision-based plot
PRECISION_PER_FILE_CSV = "./precision_points_per_file.csv"
PRECISION_PDF = "./winner_group_avg_precision.pdf"


def read_counts(path: str) -> Dict[int, Dict[str, float]]:
    data: Dict[int, Dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["winner_group_size"])  # 1..5
            llm = row["llm"]
            val_str = row.get("count") or row.get("percent_of_group")
            if val_str is None:
                continue
            # strip % if present
            if isinstance(val_str, str) and val_str.endswith("%"):
                val = float(val_str[:-1])
            else:
                val = float(val_str)
            data[k][llm] = val
    return data


def grouped_bar(
    data: Dict[int, Dict[str, float]],
    ylabel: str,
    title: str,
    outfile: str,
    xlabel: str = "Precision tie group size",
) -> None:
    groups = sorted(data.keys())
    # Maintain specific LLM order
    llm_order = [
        "gpt-3.5",
        "gpt-4o",
        "o1-mini",
        "o3-mini",
        "claude3-sonnet",
        "deepseek",
    ]
    llms = [llm for llm in llm_order if any(llm in data[g] for g in groups)]

    x = range(len(groups))
    num_llms = len(llms)
    total_width = 0.82
    bar_width = total_width / max(1, num_llms)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_llms)]

    plt.figure(figsize=(14, 7))
    color_cycle = plt.get_cmap("tab10")
    bars_by_series = []
    all_heights: List[float] = []

    for idx, llm in enumerate(llms):
        heights = [data[g].get(llm, 0.0) for g in groups]
        all_heights.extend(heights)
        positions = [i + offsets[idx] for i in x]
        bars = plt.bar(
            positions, heights, width=bar_width, label=llm, color=color_cycle(idx % 10)
        )
        bars_by_series.append(bars)

    # Add value labels
    total_bars = len(groups) * num_llms
    if total_bars <= 150:
        for bars in bars_by_series:
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    # Use float labels for precision-scale charts (<= 1.5), else integer counts
                    label = f"{height:.2f}" if height <= 1.5 else f"{int(height)}"
                    plt.annotate(
                        label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    plt.xticks(list(x), [str(g) for g in groups], rotation=30, ha="right")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def read_precision_points(path: str) -> List[Dict[str, object]]:
    """Read per-file precision rows produced by Section_5_precision.py.
    Each row format: filename, winners (semi-colon separated), points_each, then columns like "llm:score".
    Returns list of dicts with keys: filename, winners (List[str]), group_size (int), scores (Dict[str, float]).
    """
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(
            reader, None
        )  # ["filename", "winners", "points_each", "scores..."]
        for r in reader:
            if len(r) < 3:
                continue
            filename = r[0]
            winners_raw = r[1].strip()
            winners = [w for w in winners_raw.split(";") if w]
            group_size = max(1, len(winners))
            score_pairs = r[3:]
            scores: Dict[str, float] = {}
            for sp in score_pairs:
                if not sp:
                    continue
                if ":" not in sp:
                    continue
                name, val = sp.split(":", 1)
                try:
                    scores[name] = float(val)
                except ValueError:
                    continue
            rows.append(
                {
                    "filename": filename,
                    "winners": winners,
                    "group_size": group_size,
                    "scores": scores,
                }
            )
    return rows


def build_avg_precision_by_group(
    rows: List[Dict[str, object]],
) -> Dict[int, Dict[str, float]]:
    """Group files by tie group size and compute average precision per LLM within each group.
    Uses all available per-file precision scores for an LLM in that group (not only winners).
    Returns mapping: group_size -> { llm -> mean_precision }.
    """
    grouped_scores: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        k = int(row["group_size"])  # tie size
        scores: Dict[str, float] = row["scores"]  # type: ignore
        for llm, val in scores.items():
            grouped_scores[k][llm].append(val)

    result: Dict[int, Dict[str, float]] = {}
    for k, llm_dict in grouped_scores.items():
        result[k] = {llm: mean(vals) for llm, vals in llm_dict.items() if len(vals) > 0}
    return result


def main() -> None:
    raw = read_counts(COUNTS_CSV)
    norm = read_counts(COUNTS_NORM_CSV)

    grouped_bar(
        raw,
        ylabel="Number of files with highest precision",
        title="LLM appearances by precision-group size (precision ratio)",
        outfile=RAW_PDF,
        xlabel="Precision tie group size",
    )

    # New: precision-based grouped bar (Y-axis is average precision, 0..1)
    try:
        precision_rows = read_precision_points(PRECISION_PER_FILE_CSV)
        avg_prec_by_group = build_avg_precision_by_group(precision_rows)
        grouped_bar(
            avg_prec_by_group,
            ylabel="Average precision (non-Any ratio)",
            title="Average precision by precision-group size (non-Any ratio)",
            outfile=PRECISION_PDF,
            xlabel="Winner group size (number of top scorers)",
        )
    except FileNotFoundError:
        # If the precision CSV does not exist yet, skip this plot silently
        pass

    """grouped_bar(
        norm,
        ylabel="Percent of group (%)",
        title="LLM share per winner-group size (non-Any ratio)",
        outfile=NORM_PDF,
    )"""


if __name__ == "__main__":
    main()
