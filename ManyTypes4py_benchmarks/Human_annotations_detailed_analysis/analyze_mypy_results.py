import json
import os
import matplotlib.pyplot as plt
import numpy as np


PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PERCENT_NAMES = [
    "zero", "ten", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "fully",
]

LLM_CONFIGS = {
    "deepseek": {
        "base_dir": "deepseek_outputs",
        "prefix": "mypy_results_deepseek_",
        "suffix": "_percent_typed_output.json",
    },
    "o3_mini": {
        "base_dir": "o3_mini_outputs",
        "prefix": "mypy_results_o3_mini_",
        "suffix": "_percent_typed_output.json",
    },
}


def load_results(base_dir, prefix, suffix):
    """Load mypy results for all percentages. Returns dict: {percent: data}."""
    all_data = {}
    for pct, name in zip(PERCENTAGES, PERCENT_NAMES):
        path = os.path.join(base_dir, f"{prefix}{name}{suffix}")
        with open(path, "r") as f:
            all_data[pct] = json.load(f)
    return all_data


def compute_metrics(all_data):
    """Compute compilation rate and avg errors per percentage."""
    metrics = {"percent": [], "compilation_rate": [], "avg_errors": [], "total_files": []}
    for pct in PERCENTAGES:
        data = all_data[pct]
        total = len(data)
        compiled = sum(1 for v in data.values() if v["isCompiled"])
        errors = [v["error_count"] for v in data.values()]
        metrics["percent"].append(pct)
        metrics["compilation_rate"].append(compiled / total * 100 if total else 0)
        metrics["avg_errors"].append(sum(errors) / total if total else 0)
        metrics["total_files"].append(total)
    return metrics


def print_table(llm_name, metrics):
    print(f"\n{'=' * 60}")
    print(f"  {llm_name.upper()} — Mypy Results by Annotation Percentage")
    print(f"{'=' * 60}")
    print(f"{'%':>5}  {'Files':>6}  {'Compiled%':>10}  {'Avg Errors':>11}")
    print(f"{'-' * 5}  {'-' * 6}  {'-' * 10}  {'-' * 11}")
    for i, pct in enumerate(metrics["percent"]):
        print(
            f"{pct:>5}  {metrics['total_files'][i]:>6}  "
            f"{metrics['compilation_rate'][i]:>9.1f}%  "
            f"{metrics['avg_errors'][i]:>11.2f}"
        )


def plot_comparison(all_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for llm_name, metrics in all_metrics.items():
        axes[0].plot(metrics["percent"], metrics["compilation_rate"], marker="o", label=llm_name)
    axes[0].set_xlabel("Human Annotation %")
    axes[0].set_ylabel("Compilation Success Rate (%)")
    axes[0].set_title("Compilation Success vs. Annotation %")
    axes[0].set_xticks(PERCENTAGES)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for llm_name, metrics in all_metrics.items():
        axes[1].plot(metrics["percent"], metrics["avg_errors"], marker="o", label=llm_name)
    axes[1].set_xlabel("Human Annotation %")
    axes[1].set_ylabel("Average Mypy Errors per File")
    axes[1].set_title("Avg Errors vs. Annotation %")
    axes[1].set_xticks(PERCENTAGES)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis_compilation_vs_annotation.png", dpi=150)
    print("\nPlot saved: analysis_compilation_vs_annotation.png")
    plt.show()


def find_sweet_spot(metrics):
    """Find the percentage where marginal improvement drops below 1%."""
    rates = metrics["compilation_rate"]
    pcts = metrics["percent"]
    print("\n  Marginal gains (compilation rate increase per 10% more annotations):")
    for i in range(1, len(rates)):
        delta = rates[i] - rates[i - 1]
        marker = " <-- diminishing" if abs(delta) < 1.0 else ""
        print(f"    {pcts[i-1]}% -> {pcts[i]}%:  {delta:+.2f}%{marker}")


if __name__ == "__main__":
    all_metrics = {}

    for llm_name, cfg in LLM_CONFIGS.items():
        all_data = load_results(cfg["base_dir"], cfg["prefix"], cfg["suffix"])
        metrics = compute_metrics(all_data)
        all_metrics[llm_name] = metrics
        print_table(llm_name, metrics)
        find_sweet_spot(metrics)

    plot_comparison(all_metrics)
