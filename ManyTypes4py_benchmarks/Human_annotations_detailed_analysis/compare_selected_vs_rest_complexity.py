import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLEXITY_JSON = os.path.join(SCRIPT_DIR, "original_files_complexity_analysis.json")
SELECTED_DIR = os.path.join(SCRIPT_DIR, "selected_files")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "selected_vs_rest_analysis")


def load_data():
    with open(COMPLEXITY_JSON, "r", encoding="utf-8") as f:
        complexity = json.load(f)

    selected_filenames = {
        fname for fname in os.listdir(SELECTED_DIR)
        if fname.endswith(".py")
    }

    selected, rest = {}, {}
    for fname, metrics in complexity.items():
        if fname in selected_filenames:
            selected[fname] = metrics
        else:
            rest[fname] = metrics

    return selected, rest


def metrics_to_df(data: dict, group_label: str) -> pd.DataFrame:
    rows = []
    for fname, m in data.items():
        top3 = m.get("top_3_functions_CCN", [])
        rows.append({
            "filename": fname,
            "group": group_label,
            "average_CCN": m.get("average_CCN", 0),
            "max_CCN": max(top3) if top3 else 0,
            "total_line_count": m.get("total_line_count", 0),
            "function_count": m.get("function_count", 0),
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    metrics = ["average_CCN", "max_CCN", "total_line_count", "function_count"]
    summary = df.groupby("group")[metrics].agg(["count", "mean", "median", "std", "min", "max"])
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    for m in metrics:
        print(f"\n--- {m} ---")
        print(summary[m].to_string())


def run_statistical_tests(df: pd.DataFrame):
    sel = df[df["group"] == "Selected"]
    rst = df[df["group"] == "Rest"]
    metrics = ["average_CCN", "max_CCN", "total_line_count", "function_count"]

    print("\n" + "=" * 90)
    print("STATISTICAL TESTS  (Mann-Whitney U, two-sided)")
    print("=" * 90)

    results = []
    for m in metrics:
        a, b = sel[m].dropna(), rst[m].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        # rank-biserial effect size: r = 1 - 2U/(n1*n2)
        r_effect = 1 - (2 * stat) / (n1 * n2)
        results.append({
            "metric": m,
            "U_statistic": stat,
            "p_value": p,
            "effect_size_r": round(r_effect, 4),
            "selected_median": a.median(),
            "rest_median": b.median(),
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n{m}:  U={stat:.0f}  p={p:.4e}  r={r_effect:.4f}  [{sig}]")
        print(f"  Selected median={a.median():.3f}   Rest median={b.median():.3f}")

    return pd.DataFrame(results)


def plot_boxplots(df: pd.DataFrame, out_dir: str):
    metrics = ["average_CCN", "max_CCN", "total_line_count", "function_count"]
    labels = ["Average CCN", "Max CCN (top-3)", "Total Line Count", "Function Count"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, m, label in zip(axes, metrics, labels):
        data_sel = df[df["group"] == "Selected"][m]
        data_rst = df[df["group"] == "Rest"][m]
        bp = ax.boxplot(
            [data_sel, data_rst],
            tick_labels=["Selected", "Rest"],
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label)

    fig.suptitle("Selected Files vs Rest — Complexity Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "boxplots_selected_vs_rest.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    print(f"\nSaved boxplots to {path}")
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out_dir: str):
    metrics = ["average_CCN", "max_CCN", "total_line_count", "function_count"]
    labels = ["Average CCN", "Max CCN (top-3)", "Total Line Count", "Function Count"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, m, label in zip(axes, metrics, labels):
        sel = df[df["group"] == "Selected"][m]
        rst = df[df["group"] == "Rest"][m]
        bins = np.histogram_bin_edges(np.concatenate([sel, rst]), bins="auto")
        ax.hist(rst, bins=bins, alpha=0.55, label="Rest", color="#DD8452", density=True)
        ax.hist(sel, bins=bins, alpha=0.55, label="Selected", color="#4C72B0", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Distribution Overlap — Selected vs Rest", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "histograms_selected_vs_rest.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    print(f"Saved histograms to {path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected, rest = load_data()
    print(f"Selected files: {len(selected)}   |   Rest files: {len(rest)}")

    df_sel = metrics_to_df(selected, "Selected")
    df_rst = metrics_to_df(rest, "Rest")
    df = pd.concat([df_sel, df_rst], ignore_index=True)

    print_summary(df)
    test_results = run_statistical_tests(df)

    test_results.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)
    print(f"\nSaved test results CSV to {OUTPUT_DIR}/statistical_tests.csv")

    plot_boxplots(df, OUTPUT_DIR)
    plot_histograms(df, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
