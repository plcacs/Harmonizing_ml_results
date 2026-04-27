"""
plot_annotation_mypy_figures_v2.py

Extended visualization suite: 7 figures.

  Fig 1  - Error code breakdown stacked bar (setting1 vs setting2)
  Fig 2  - Per-file error delta scatter     (setting1 error vs setting2 error)
  Fig 3  - Coverage gain vs error change    (paired coverage vs errors)
  Fig 4  - Error code heatmap by coverage bucket
  Fig 5  - Clean-rate 2-D heatmap           (coverage bucket x Any bucket)
  Fig 6  - File outcome transition matrix   (clean/fail per setting pair)
  Fig 7  - Error count distribution violin  (failing files only)

Usage:
  python plot_annotation_mypy_figures_v2.py            # show all figures
  python plot_annotation_mypy_figures_v2.py --no-show  # build without displaying
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from annotation_vs_mypy_joined import SETTINGS, build_joined, load_selected_stems

MYPY_DIR = Path(__file__).resolve().parent.parent / "GPCE_mypy_results"

MYPY_JSONS = {
    "GPT5 setting1 (inline)":     MYPY_DIR / "mypy_results_gpt5_2_run_with_errors.json",
    "GPT5 setting2 (stub)":       MYPY_DIR / "mypy_results_gpt5_1_infer_stub_run_with_errors.json",
    "DeepSeek setting1 (inline)": MYPY_DIR / "mypy_results_deepseek_3_run_with_errors.json",
    "DeepSeek setting2 (stub)":   MYPY_DIR / "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
}

SETTING_PAIRS = [
    ("GPT5 setting1 (inline)",     "GPT5 setting2 (stub)"),
    ("DeepSeek setting1 (inline)", "DeepSeek setting2 (stub)"),
]

COLOR_MAP = {
    "GPT5 setting1 (inline)":     "#1f77b4",
    "GPT5 setting2 (stub)":       "#ff7f0e",
    "DeepSeek setting1 (inline)": "#2ca02c",
    "DeepSeek setting2 (stub)":   "#d62728",
}

COV_BUCKETS = [(0, 50), (50, 80), (80, 95), (95, 101)]
COV_LABELS  = ["0-50%", "50-80%", "80-95%", "95-100%"]
ANY_BUCKETS = [(0, 0.01), (0.01, 25), (25, 50), (50, 101)]
ANY_LABELS  = ["0%", "0-25%", "25-50%", "50%+"]


def short(label):
    return {
        "GPT5 setting1 (inline)":     "GPT5-inline",
        "GPT5 setting2 (stub)":       "GPT5-stub",
        "DeepSeek setting1 (inline)": "DS-inline",
        "DeepSeek setting2 (stub)":   "DS-stub",
    }.get(label, label)


# -- Data loaders -------------------------------------------------------------

def load_rows_by_setting():
    allowed = load_selected_stems()
    return {s["label"]: build_joined(s, allowed) for s in SETTINGS}


def load_error_codes_by_setting(allowed_stems):
    result = {}
    for label, path in MYPY_JSONS.items():
        with open(path) as f:
            data = json.load(f)
        codes = []
        for filename, entry in data.items():
            stem = Path(filename).stem
            if stem not in allowed_stems:
                continue
            for err_line in entry.get("errors", []):
                m = re.search(r"\[([a-z][a-z0-9\-]*)\]", err_line)
                if m:
                    codes.append(m.group(1))
        result[label] = codes
    return result


# -- Fig 1: Error code breakdown stacked bar ----------------------------------

def plot_error_code_breakdown(error_codes_by_setting, top_n=12):
    all_codes = []
    for codes in error_codes_by_setting.values():
        all_codes.extend(codes)
    top_codes = [c for c, _ in Counter(all_codes).most_common(top_n)]

    settings = list(error_codes_by_setting.keys())
    counts = np.zeros((len(settings), len(top_codes)), dtype=int)
    for i, label in enumerate(settings):
        c = Counter(error_codes_by_setting[label])
        for j, code in enumerate(top_codes):
            counts[i, j] = c.get(code, 0)

    cmap = plt.colormaps["tab20"]
    colors = [cmap(k / len(top_codes)) for k in range(len(top_codes))]

    fig, ax = plt.subplots(figsize=(12, 5))
    lefts = np.zeros(len(settings))
    for j, code in enumerate(top_codes):
        ax.barh(
            [short(s) for s in settings],
            counts[:, j],
            left=lefts,
            color=colors[j],
            label=code,
            edgecolor="white",
            linewidth=0.4,
        )
        lefts += counts[:, j]

    ax.set_title(f"Figure 1: Top-{top_n} Mypy Error Codes by Setting")
    ax.set_xlabel("Error count (all evaluable files)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Error code")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# -- Fig 2: Per-file error delta scatter ---------------------------------------

def plot_per_file_error_delta(rows_by_setting):
    n_pairs = len(SETTING_PAIRS)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)

    for ax, (l1, l2) in zip(axes[0], SETTING_PAIRS):
        map1 = {r["file"]: r for r in rows_by_setting[l1]}
        map2 = {r["file"]: r for r in rows_by_setting[l2]}
        common = sorted(set(map1) & set(map2))

        xs = [map1[f]["error_count"] for f in common]
        ys = [map2[f]["error_count"] for f in common]
        cov_deltas = [map2[f]["coverage_pct"] - map1[f]["coverage_pct"] for f in common]
        point_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in cov_deltas]

        ax.scatter(xs, ys, c=point_colors, s=18, alpha=0.55, edgecolors="none")
        lim = max(max(xs, default=0), max(ys, default=0)) + 2
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.set_xlabel(f"{short(l1)} errors")
        ax.set_ylabel(f"{short(l2)} errors")
        ax.set_title(f"{short(l1)} vs {short(l2)}\n"
                     f"(green=cov up in s2, red=cov down)")
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 2: Per-file Error Count -- Setting1 vs Setting2", fontweight="bold")
    fig.tight_layout()
    return fig


# -- Fig 3: Coverage gain vs error change (paired) ----------------------------

def plot_coverage_gain_vs_error_change(rows_by_setting):
    n_pairs = len(SETTING_PAIRS)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)

    for ax, (l1, l2) in zip(axes[0], SETTING_PAIRS):
        map1 = {r["file"]: r for r in rows_by_setting[l1]}
        map2 = {r["file"]: r for r in rows_by_setting[l2]}
        common = sorted(set(map1) & set(map2))

        dx = [map2[f]["coverage_pct"] - map1[f]["coverage_pct"] for f in common]
        dy = [map2[f]["error_count"] - map1[f]["error_count"] for f in common]

        ax.scatter(dx, dy, s=18, alpha=0.55, color=COLOR_MAP[l1], edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coverage delta % (s2 - s1)")
        ax.set_ylabel("Error count delta (s2 - s1)")
        ax.set_title(f"{short(l1)} -> {short(l2)}\n(n={len(common)} paired files)")
        ax.grid(alpha=0.2)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.text(xlim[1], ylim[1], "more cov\nmore err", ha="right", va="top",
                fontsize=7, color="red", alpha=0.7)
        ax.text(xlim[1], ylim[0], "more cov\nfewer err", ha="right", va="bottom",
                fontsize=7, color="green", alpha=0.7)
        ax.text(xlim[0], ylim[1], "less cov\nmore err", ha="left", va="top",
                fontsize=7, color="orange", alpha=0.7)
        ax.text(xlim[0], ylim[0], "less cov\nfewer err", ha="left", va="bottom",
                fontsize=7, color="steelblue", alpha=0.7)

    fig.suptitle("Figure 3: Coverage Delta vs Error Delta (Paired)", fontweight="bold")
    fig.tight_layout()
    return fig


# -- Fig 4: Error code heatmap by coverage bucket -----------------------------

def plot_error_code_heatmap_by_coverage(rows_by_setting, error_codes_by_setting, top_n=10):
    all_codes = []
    for codes in error_codes_by_setting.values():
        all_codes.extend(codes)
    top_codes = [c for c, _ in Counter(all_codes).most_common(top_n)]

    n_settings = len(SETTINGS)
    fig, axes = plt.subplots(1, n_settings, figsize=(5 * n_settings, 5), squeeze=False)

    for ax, s in zip(axes[0], SETTINGS):
        label = s["label"]
        rows = rows_by_setting[label]
        row_map = {r["file"]: r for r in rows}

        with open(MYPY_JSONS[label]) as f:
            mypy_data = json.load(f)

        matrix = np.zeros((len(top_codes), len(COV_BUCKETS)), dtype=int)

        for filename, entry in mypy_data.items():
            stem = Path(filename).stem
            if stem not in row_map:
                continue
            cov = row_map[stem]["coverage_pct"]
            b_idx = None
            for bi, (lo, hi) in enumerate(COV_BUCKETS):
                if lo <= cov < hi:
                    b_idx = bi
                    break
            if b_idx is None:
                continue
            for err_line in entry.get("errors", []):
                m = re.search(r"\[([a-z][a-z0-9\-]*)\]", err_line)
                if m and m.group(1) in top_codes:
                    ci = top_codes.index(m.group(1))
                    matrix[ci, b_idx] += 1

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(COV_LABELS)))
        ax.set_xticklabels(COV_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(top_codes)))
        ax.set_yticklabels(top_codes, fontsize=8)
        ax.set_title(short(label), fontsize=9)
        plt.colorbar(im, ax=ax, label="count")

        for ci in range(len(top_codes)):
            for bi in range(len(COV_BUCKETS)):
                val = matrix[ci, bi]
                if val > 0:
                    ax.text(bi, ci, str(val), ha="center", va="center",
                            fontsize=6, color="black" if val < matrix.max() * 0.7 else "white")

    fig.suptitle(f"Figure 4: Error Code Heatmap by Coverage Bucket (top {top_n})",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# -- Fig 5: Clean-rate 2-D heatmap (coverage x Any) ---------------------------

def plot_2d_clean_rate_heatmap(rows_by_setting):
    n_settings = len(SETTINGS)
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), squeeze=False)

    for ax, s in zip(axes[0], SETTINGS):
        label = s["label"]
        rows = rows_by_setting[label]

        matrix = np.full((len(COV_BUCKETS), len(ANY_BUCKETS)), np.nan)
        for ci, (clo, chi) in enumerate(COV_BUCKETS):
            for ai, (alo, ahi) in enumerate(ANY_BUCKETS):
                bucket = [r for r in rows
                          if clo <= r["coverage_pct"] < chi
                          and alo <= r["any_pct"] < ahi]
                if bucket:
                    matrix[ci, ai] = sum(r["is_clean"] for r in bucket) / len(bucket) * 100

        im = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(ANY_LABELS)))
        ax.set_xticklabels(ANY_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(COV_LABELS)))
        ax.set_yticklabels(COV_LABELS, fontsize=8)
        ax.set_title(short(label), fontsize=9)
        plt.colorbar(im, ax=ax, label="clean rate %")

        for ci in range(len(COV_BUCKETS)):
            for ai in range(len(ANY_BUCKETS)):
                val = matrix[ci, ai]
                if not np.isnan(val):
                    ax.text(ai, ci, f"{val:.0f}%", ha="center", va="center",
                            fontsize=7,
                            color="black" if 20 < val < 80 else "white")

    fig.suptitle("Figure 5: Clean Rate % by Coverage Bucket x Any% Bucket",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# -- Fig 6: File outcome transition matrix -------------------------------------

def plot_transition_matrix(rows_by_setting):
    n_pairs = len(SETTING_PAIRS)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4), squeeze=False)
    labels = ["Clean", "Fail"]

    for ax, (l1, l2) in zip(axes[0], SETTING_PAIRS):
        map1 = {r["file"]: r for r in rows_by_setting[l1]}
        map2 = {r["file"]: r for r in rows_by_setting[l2]}
        common = sorted(set(map1) & set(map2))

        matrix = np.zeros((2, 2), dtype=int)
        for f in common:
            i = 0 if map1[f]["is_clean"] else 1
            j = 0 if map2[f]["is_clean"] else 1
            matrix[i, j] += 1

        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"{short(l2)}\n{lb}" for lb in labels])
        ax.set_yticklabels([f"{short(l1)} {lb}" for lb in labels])
        ax.set_title(f"{short(l1)} -> {short(l2)}\n(n={len(common)} files)")

        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = f"\n({val / total * 100:.1f}%)" if total else ""
                ax.text(j, i, f"{val}{pct}", ha="center", va="center",
                        fontsize=11,
                        color="white" if val > matrix.max() * 0.6 else "black")

        plt.colorbar(im, ax=ax)

    fig.suptitle("Figure 6: File Outcome Transition Matrix (Clean / Fail)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# -- Fig 7: Error count distribution violin ------------------------------------

def plot_error_count_violin(rows_by_setting):
    setting_labels = [s["label"] for s in SETTINGS]
    data = []
    for label in setting_labels:
        failing = [r["error_count"] for r in rows_by_setting[label] if not r["is_clean"]]
        data.append(failing if failing else [0])

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=range(len(setting_labels)),
                          showmedians=False, showextrema=False)

    for i, (pc, label) in enumerate(zip(parts["bodies"], setting_labels)):
        pc.set_facecolor(COLOR_MAP[label])
        pc.set_alpha(0.7)

    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", linewidth=4, zorder=3)
        ax.scatter([i], [med], color="white", s=30, zorder=4)

    ax.set_xticks(range(len(setting_labels)))
    ax.set_xticklabels([short(l) for l in setting_labels])
    ax.set_ylabel("Mypy error count")
    ax.set_title("Figure 7: Error Count Distribution (Failing Files Only)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 7 annotation-vs-mypy figures."
    )
    parser.add_argument("--no-show", action="store_true",
                        help="Build all figures without opening display windows.")
    args = parser.parse_args()

    print("Loading rows...")
    rows_by_setting = load_rows_by_setting()

    print("Loading error codes...")
    allowed = load_selected_stems()
    error_codes_by_setting = load_error_codes_by_setting(allowed)

    print("Plotting Figure 1: Error code breakdown...")
    plot_error_code_breakdown(error_codes_by_setting)

    print("Plotting Figure 2: Per-file error delta scatter...")
    plot_per_file_error_delta(rows_by_setting)

    print("Plotting Figure 3: Coverage gain vs error change...")
    plot_coverage_gain_vs_error_change(rows_by_setting)

    print("Plotting Figure 4: Error code heatmap by coverage bucket...")
    plot_error_code_heatmap_by_coverage(rows_by_setting, error_codes_by_setting)

    print("Plotting Figure 5: 2D clean-rate heatmap...")
    plot_2d_clean_rate_heatmap(rows_by_setting)

    print("Plotting Figure 6: Transition matrix...")
    plot_transition_matrix(rows_by_setting)

    print("Plotting Figure 7: Error count violin...")
    plot_error_count_violin(rows_by_setting)

    if args.no_show:
        print("\nAll 7 figures generated (not shown, not saved).")
    else:
        plt.show()


if __name__ == "__main__":
    main()
