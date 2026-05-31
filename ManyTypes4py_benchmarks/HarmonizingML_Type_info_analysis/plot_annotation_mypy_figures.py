import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from annotation_vs_mypy_joined import SETTINGS, build_joined, load_selected_stems


def short_name(label: str) -> str:
    mapping = {
        "GPT5 setting1 (inline)": "GPT5-inline",
        "GPT5 setting2 (stub)": "GPT5-stub",
        "DeepSeek setting1 (inline)": "DS-inline",
        "DeepSeek setting2 (stub)": "DS-stub",
    }
    return mapping.get(label, label)


def bucket_any(any_pct: float) -> str:
    if any_pct == 0:
        return "0%"
    if any_pct < 10:
        return "0-10%"
    if any_pct < 20:
        return "10-20%"
    return "20%+"


def clean_rate(rows):
    if not rows:
        return 0.0
    clean = sum(1 for r in rows if r["is_clean"])
    return clean / len(rows) * 100


def load_rows_by_setting():
    allowed = load_selected_stems()
    rows_by_setting = {}
    for s in SETTINGS:
        rows_by_setting[s["label"]] = build_joined(s, allowed)
    return rows_by_setting


def plot_coverage_vs_errors_scatter(rows_by_setting):
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        "GPT5 setting1 (inline)": "#1f77b4",
        "GPT5 setting2 (stub)": "#ff7f0e",
        "DeepSeek setting1 (inline)": "#2ca02c",
        "DeepSeek setting2 (stub)": "#d62728",
    }

    for setting, rows in rows_by_setting.items():
        xs = [r["coverage_pct"] for r in rows]
        ys = [r["error_count"] for r in rows]
        ax.scatter(
            xs,
            ys,
            s=22,
            alpha=0.55,
            label=f"{short_name(setting)} (n={len(rows)})",
            color=color_map.get(setting),
            edgecolors="none",
        )

    ax.set_title("Figure 1: Coverage vs Mypy Errors")
    ax.set_xlabel("Annotation coverage (%)")
    ax.set_ylabel("Mypy error_count per file")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_any_bucket_clean_rate(rows_by_setting):
    bucket_labels = ["0%", "0-10%", "10-20%", "20%+"]
    x = np.arange(len(bucket_labels))

    fig, ax = plt.subplots(figsize=(10, 5))

    for setting, rows in rows_by_setting.items():
        grouped = defaultdict(list)
        for r in rows:
            grouped[bucket_any(r["any_pct"])].append(r)

        rates = [clean_rate(grouped[b]) for b in bucket_labels]
        ax.plot(x, rates, marker="o", linewidth=2, label=short_name(setting))

    ax.set_title("Figure 2: Any% vs Mypy Clean Rate")
    ax.set_xlabel("Any% bucket (per file)")
    ax.set_ylabel("Mypy clean rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2 combined annotation-vs-mypy figures (no file saving)."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Build figures without displaying windows."
    )
    args = parser.parse_args()

    rows_by_setting = load_rows_by_setting()

    plot_coverage_vs_errors_scatter(rows_by_setting)
    plot_any_bucket_clean_rate(rows_by_setting)

    if args.no_show:
        print("Generated 2 figures (not shown, not saved).")
    else:
        plt.show()


if __name__ == "__main__":
    main()
