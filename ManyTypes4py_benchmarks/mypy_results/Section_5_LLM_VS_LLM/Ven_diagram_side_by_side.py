import json
from typing import Set
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def load_compiled_success_files(untyped_path: str) -> Set[str]:
    with open(untyped_path) as f_untyped:
        untyped_results = json.load(f_untyped)
    return {
        filename
        for filename, info in untyped_results.items()
        if info.get("isCompiled") is True
    }


def load_success_set(path: str, allowed_filenames: Set[str]) -> Set[str]:
    with open(path) as f:
        data = json.load(f)
    return {
        filename
        for filename, info in data.items()
        if info.get("isCompiled") is True and filename in allowed_filenames
    }


def fmt_label(name: str, s: Set[str], total_base: int) -> str:
    count = len(s)
    pct = (count * 100.0 / total_base) if total_base else 0.0
    return f"{name} ({count}/{total_base}, {pct:.1f}%)"


def draw_equal_circle_venn(ax, sets, labels):
    try:
        # Newer matplotlib-venn supports explicit equal-circle layout
        v = venn3(sets, set_labels=labels, ax=ax, layout_algorithm="circular")
    except TypeError:
        # Fallback for older versions without layout_algorithm
        v = venn3(sets, set_labels=labels, ax=ax)
    return v


def draw_manytypes4py(ax):
    compiled_success_files = load_compiled_success_files(
        "mypy_outputs/mypy_results_untyped_with_errors.json"
    )
    total_base = len(compiled_success_files)

    claude_path = "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json"
    o3mini_path = "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json"
    deepseek_path = "mypy_outputs/mypy_results_deepseek_with_errors.json"

    claude = load_success_set(claude_path, compiled_success_files)
    o3mini = load_success_set(o3mini_path, compiled_success_files)
    deepseek = load_success_set(deepseek_path, compiled_success_files)

    v = draw_equal_circle_venn(
        ax,
        [claude, o3mini, deepseek],
        (
            fmt_label("Claude 3 Sonnet", claude, total_base),
            fmt_label("o3-mini", o3mini, total_base),
            fmt_label("Deepseek", deepseek, total_base),
        ),
    )
    for text in v.set_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight("bold")
    for text in v.subset_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight("bold")
    ax.set_axis_off()


def draw_pyperformance(ax):
    compiled_success_files = load_compiled_success_files(
        "../../pyperformance_benchmarks/mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
        
    )
    total_base = len(compiled_success_files)

    claude_path = "../../pyperformance_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json"
    o3mini_path = "../../pyperformance_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json"
    deepseek_path = "../../pyperformance_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_1st_run_with_errors.json"

    claude = load_success_set(claude_path, compiled_success_files)
    o3mini = load_success_set(o3mini_path, compiled_success_files)
    deepseek = load_success_set(deepseek_path, compiled_success_files)

    v = draw_equal_circle_venn(
        ax,
        [claude, o3mini, deepseek],
        (
            fmt_label("Claude 3 Sonnet", claude, total_base),
            fmt_label("o3-mini", o3mini, total_base),
            fmt_label("Deepseek", deepseek, total_base),
        ),
    )
    for text in v.set_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight("bold")
    for text in v.subset_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight("bold")
    ax.set_axis_off()


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: ManyTypes4py; Right: PyPerformance
    draw_manytypes4py(ax1)
    draw_pyperformance(ax2)

    plt.rcParams.update({"font.size": 14})
    plt.tight_layout()
    # Save PDF next to this script
    out_path =  "Section_5_LLM_VS_LLM/venn_diagrams_side_by_side.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()


