"""
Scatter plot: Any% (x-axis) vs Mypy error count (y-axis).

One figure per LLM (GPT-5, DeepSeek), each with both settings overlaid.
Files with 0 errors (clean / "compiled") sit on the x-axis.
"""

import matplotlib.pyplot as plt
from annotation_vs_mypy_joined import SETTINGS, build_joined, load_selected_stems

# Group the 4 settings into 2 LLMs, each with 2 settings
LLM_GROUPS = {
    "GPT-5": [
        ("Inline", "GPT5 setting1 (inline)"),
        ("Stub",   "GPT5 setting2 (stub)"),
    ],
    "DeepSeek": [
        ("Inline", "DeepSeek setting1 (inline)"),
        ("Stub",   "DeepSeek setting2 (stub)"),
    ],
}

COLORS = {"Inline": "#1f77b4", "Stub": "#ff7f0e"}


def main():
    # Load data
    allowed = load_selected_stems()
    rows_by_label = {}
    for s in SETTINGS:
        rows_by_label[s["label"]] = build_joined(s, allowed)

    # One figure per LLM
    for llm_name, setting_pairs in LLM_GROUPS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for short_name, full_label in setting_pairs:
            rows = rows_by_label[full_label]
            x = [r["any_pct"] for r in rows]
            y = [r["error_count"] for r in rows]

            ax.scatter(
                x, y,
                label=f"{short_name} (n={len(rows)})",
                color=COLORS[short_name],
                alpha=0.5,
                s=25,
                edgecolors="none",
            )

        ax.set_xlabel("Any %  (proportion of Any/blank annotations)")
        ax.set_ylabel("Mypy error count  (0 = clean / compiled)")
        ax.set_title(f"{llm_name}: Any% vs Mypy Errors  (Inline vs Stub)")
        ax.set_xticks(range(0, 101, 10))
        ax.set_xlim(-2, 102)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
