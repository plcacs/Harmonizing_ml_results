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


def plot_any_bucket_clean_rate(rows_by_setting):
    bucket_labels = ["0%", "0-10%", "10-20%", "20%+"]
    settings = list(rows_by_setting.keys())

    x = np.arange(len(bucket_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, setting in enumerate(settings):
        rows = rows_by_setting[setting]
        grouped = defaultdict(list)
        for r in rows:
            grouped[bucket_any(r["any_pct"])].append(r)

        rates = [clean_rate(grouped[b]) for b in bucket_labels]
        offset = (i - (len(settings) - 1) / 2) * width
        ax.bar(x + offset, rates, width=width, label=short_name(setting))

    ax.set_title("Figure 1: Mypy Clean Rate by Any% Bucket")
    ax.set_xlabel("Any% bucket (per file)")
    ax.set_ylabel("Mypy clean rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def paired_rows(rows_a, rows_b):
    map_a = {r["file"]: r for r in rows_a}
    map_b = {r["file"]: r for r in rows_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    return map_a, map_b, common


def collect_divergent_deltas(rows1, rows2):
    map1, map2, common = paired_rows(rows1, rows2)
    cov_s1_clean_s2_fail = []
    cov_s2_clean_s1_fail = []
    any_s1_clean_s2_fail = []
    any_s2_clean_s1_fail = []

    for f in common:
        r1, r2 = map1[f], map2[f]
        if r1["is_clean"] == r2["is_clean"]:
            continue

        cov_delta = r2["coverage_pct"] - r1["coverage_pct"]
        any_delta = r2["any_pct"] - r1["any_pct"]

        if r1["is_clean"] and not r2["is_clean"]:
            cov_s1_clean_s2_fail.append(cov_delta)
            any_s1_clean_s2_fail.append(any_delta)
        elif r2["is_clean"] and not r1["is_clean"]:
            cov_s2_clean_s1_fail.append(cov_delta)
            any_s2_clean_s1_fail.append(any_delta)

    return {
        "cov": [cov_s1_clean_s2_fail, cov_s2_clean_s1_fail],
        "any": [any_s1_clean_s2_fail, any_s2_clean_s1_fail],
    }


def _boxplot_or_text(ax, data, labels, title):
    if sum(len(d) for d in data) == 0:
        ax.text(0.5, 0.5, "No divergent files", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        return

    safe_data = [d if d else [0] for d in data]
    bp = ax.boxplot(safe_data, tick_labels=labels, showfliers=False)
    for box in bp["boxes"]:
        box.set_alpha(0.6)

    for idx, arr in enumerate(data, start=1):
        if not arr:
            continue
        jitter_x = np.random.normal(idx, 0.03, size=len(arr))
        ax.scatter(jitter_x, arr, s=12, alpha=0.35)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_divergent_deltas(rows_by_setting):
    pairs = [
        ("GPT5 setting1 (inline)", "GPT5 setting2 (stub)", "GPT5"),
        ("DeepSeek setting1 (inline)", "DeepSeek setting2 (stub)", "DeepSeek"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="col")

    for row_idx, (s1, s2, model_name) in enumerate(pairs):
        deltas = collect_divergent_deltas(rows_by_setting[s1], rows_by_setting[s2])
        labels = ["s1 clean -> s2 fail", "s2 clean -> s1 fail"]

        _boxplot_or_text(
            axes[row_idx, 0],
            deltas["cov"],
            labels,
            f"{model_name}: Coverage delta (s2 - s1) on divergent files",
        )
        _boxplot_or_text(
            axes[row_idx, 1],
            deltas["any"],
            labels,
            f"{model_name}: Any% delta (s2 - s1) on divergent files",
        )

    fig.suptitle("Figure 2: Divergent-file Deltas Between Inline and Stub", y=1.02)
    fig.tight_layout()
    return fig


def plot_composition_and_clean_rate(rows_by_setting):
    settings = list(rows_by_setting.keys())
    labels = [short_name(s) for s in settings]

    concrete_p, any_p, blank_p, clean_p = [], [], [], []

    for s in settings:
        rows = rows_by_setting[s]
        total_slots = sum(r["total_slots"] for r in rows)
        total_concrete = sum((r["concrete_pct"] / 100.0) * r["total_slots"] for r in rows)
        total_any = sum(r["any_count"] for r in rows)
        total_blank = sum(r["total_slots"] - r["annotated"] for r in rows)

        if total_slots == 0:
            concrete_p.append(0)
            any_p.append(0)
            blank_p.append(0)
        else:
            concrete_p.append(total_concrete / total_slots * 100)
            any_p.append(total_any / total_slots * 100)
            blank_p.append(total_blank / total_slots * 100)

        clean_p.append(clean_rate(rows))

    x = np.arange(len(settings))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x, concrete_p, label="Concrete", color="#4c78a8")
    ax1.bar(x, any_p, bottom=concrete_p, label="Any", color="#f58518")
    ax1.bar(x, blank_p, bottom=np.array(concrete_p) + np.array(any_p), label="Blank", color="#e45756")

    ax1.set_ylabel("Annotation composition (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.set_title("Figure 3: Annotation Composition with Mypy Clean Rate")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, clean_p, color="black", marker="o", linewidth=2, label="Mypy clean rate")
    ax2.set_ylabel("Mypy clean rate (%)")
    ax2.set_ylim(0, 100)

    for i, v in enumerate(clean_p):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate 3 combined annotation-vs-mypy figures (no file saving).")
    parser.add_argument("--no-show", action="store_true", help="Build figures without displaying windows.")
    args = parser.parse_args()

    rows_by_setting = load_rows_by_setting()

    plot_any_bucket_clean_rate(rows_by_setting)
    plot_divergent_deltas(rows_by_setting)
    plot_composition_and_clean_rate(rows_by_setting)

    if args.no_show:
        print("Generated 3 figures (not shown, not saved).")
    else:
        plt.show()


if __name__ == "__main__":
    main()
