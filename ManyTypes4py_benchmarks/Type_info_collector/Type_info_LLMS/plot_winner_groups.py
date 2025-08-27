import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

COUNTS_CSV = "./winner_group_counts.csv"
COUNTS_NORM_CSV = "./winner_group_counts_normalized.csv"

RAW_PDF = "./winner_group_counts.pdf"
NORM_PDF = "./winner_group_counts_normalized.pdf"


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


def grouped_bar(data: Dict[int, Dict[str, float]], ylabel: str, title: str, outfile: str) -> None:
    groups = sorted(data.keys())
    # Collect union of LLMs across groups
    llms: List[str] = sorted({llm for g in groups for llm in data[g].keys()})

    x = range(len(groups))
    num_llms = len(llms)
    total_width = 0.8
    bar_width = total_width / max(1, num_llms)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_llms)]

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    for idx, llm in enumerate(llms):
        heights = [data[g].get(llm, 0.0) for g in groups]
        positions = [i + offsets[idx] for i in x]
        plt.bar(positions, heights, width=bar_width, label=llm, color=cmap(idx % 10))

    plt.xticks(list(x), [str(g) for g in groups])
    plt.xlabel("Winner group size (number of tied winners)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def main() -> None:
    raw = read_counts(COUNTS_CSV)
    norm = read_counts(COUNTS_NORM_CSV)

    grouped_bar(
        raw,
        ylabel="Count of appearances as winner",
        title="LLM winner appearances by winner-group size (non-Any ratio)",
        outfile=RAW_PDF,
    )

    grouped_bar(
        norm,
        ylabel="Percent of group (%)",
        title="LLM share per winner-group size (non-Any ratio)",
        outfile=NORM_PDF,
    )


if __name__ == "__main__":
    main()
