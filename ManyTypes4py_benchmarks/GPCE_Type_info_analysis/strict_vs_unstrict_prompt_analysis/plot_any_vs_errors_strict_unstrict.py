"""
Scatter plot: Any% (x-axis) vs Mypy error count (y-axis).

One figure per LLM (GPT-5, DeepSeek), each with strict vs unstrict overlaid.
Files with 0 errors (clean / "compiled") sit on the x-axis.

Usage:
  python plot_any_vs_errors_strict_unstrict.py            # show figures
  python plot_any_vs_errors_strict_unstrict.py --save      # save to figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from prompt_settings import SETTINGS, load_rows_by_setting

LLM_GROUPS = {
    "GPT-5": [
        ("Unstrict", "GPT5 unstrict"),
        ("Strict",   "GPT5 strict"),
    ],
    "DeepSeek": [
        ("Unstrict", "DeepSeek unstrict"),
        ("Strict",   "DeepSeek strict"),
    ],
}

COLORS = {"Unstrict": "#1f77b4", "Strict": "#ff7f0e"}

OUT_DIR = Path(__file__).resolve().parent / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save figures to figures/")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("Loading rows...")
    rows_by_label = load_rows_by_setting()

    if args.save:
        OUT_DIR.mkdir(exist_ok=True)

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
        ax.set_title(f"{llm_name}: Any% vs Mypy Errors  (Unstrict vs Strict Prompt)")
        ax.set_xticks(range(0, 101, 10))
        ax.set_xlim(-2, 102)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if args.save:
            safe = llm_name.replace("-", "").lower()
            fname = OUT_DIR / f"any_vs_errors_{safe}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")

    if not args.no_show and not args.save:
        plt.show()


if __name__ == "__main__":
    main()
