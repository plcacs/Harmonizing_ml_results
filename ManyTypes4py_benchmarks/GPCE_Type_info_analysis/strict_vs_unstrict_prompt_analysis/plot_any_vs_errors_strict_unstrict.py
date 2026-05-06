"""Scatter plot of Any% vs Mypy errors for strict/unstrict prompts per LLM."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from prompt_settings import SETTINGS, load_rows_by_setting

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "figures"

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

STRUCTURAL_CSVS = {
    "GPT5 unstrict":     SCRIPT_DIR / "files_with_changes_gpt5_2.csv",
    "GPT5 strict":       SCRIPT_DIR / "files_with_changes_gpt5_4.csv",
    "DeepSeek unstrict": SCRIPT_DIR / "files_with_changes_deepseek_2.csv",
    "DeepSeek strict":   SCRIPT_DIR / "files_with_changesdeepseek4.csv",
}


def load_structural_stems(csv_path):
    """Return set of file stems that have structural changes."""
    stems = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            stems.add(fname.removesuffix(".py"))
    return stems


def plot_2d(rows_by_label, args):
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


def plot_3d(rows_by_label, args):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    structural_stems = {
        label: load_structural_stems(csv_path)
        for label, csv_path in STRUCTURAL_CSVS.items()
    }

    for llm_name, setting_pairs in LLM_GROUPS.items():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        for short_name, full_label in setting_pairs:
            rows = rows_by_label[full_label]
            changed = structural_stems[full_label]

            x = [r["any_pct"] for r in rows]
            y = [r["error_count"] for r in rows]
            z = [1 if r["file"] in changed else 0 for r in rows]

            ax.scatter(
                x, y, z,
                label=f"{short_name} (n={len(rows)})",
                color=COLORS[short_name],
                alpha=0.5,
                s=25,
                edgecolors="none",
            )

        ax.set_xlabel("Any %")
        ax.set_ylabel("Mypy error count")
        ax.set_zlabel("Structural change (0=no, 1=yes)")
        ax.set_title(f"{llm_name}: Any% vs Mypy Errors vs Structural Changes")
        ax.set_xticks(range(0, 101, 10))
        ax.set_zticks([0, 1])
        ax.legend()
        fig.tight_layout()

        if args.save:
            safe = llm_name.replace("-", "").lower()
            fname = OUT_DIR / f"any_vs_errors_3d_{safe}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save figures to figures/")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="3D plot with structural-change z-axis")
    args = parser.parse_args()

    print("Loading rows...")
    rows_by_label = load_rows_by_setting()

    if args.save:
        OUT_DIR.mkdir(exist_ok=True)

    if args.three_d:
        plot_3d(rows_by_label, args)
    else:
        plot_2d(rows_by_label, args)

    if not args.no_show and not args.save:
        plt.show()


if __name__ == "__main__":
    main()
