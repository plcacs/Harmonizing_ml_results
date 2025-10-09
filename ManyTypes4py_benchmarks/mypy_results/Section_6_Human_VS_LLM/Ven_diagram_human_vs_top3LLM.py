import json
import os
from typing import Set
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Global style tweaks
plt.style.use('default')
plt.rcParams.update({
    'font.size': 16,
})
plt.rcParams['savefig.pad_inches'] = 0

# Color map as requested
COLOR_MAP = {
    "Human (original)": "pink",
    "o3-mini": "red",
    "deepseek": "blue",
    "claude3 sonnet": "purple",
    "Union of 3 LLMs": "#8B5A96",  
    "Human": "pink",
    "claude3-sonnet": "purple",
    "gpt-4o": "orange",
    "o1-mini": "skyblue",
    "gpt-3.5": "green",
}

# Build the set of compiled-success files from untyped results
with open("mypy_outputs/mypy_results_untyped_with_errors.json") as f_untyped:
    untyped_results = json.load(f_untyped)

compiled_success_files = {
    filename for filename, info in untyped_results.items() if info.get("isCompiled") is True
}
print(f"Number of compiled-success files: {len(compiled_success_files)}")


def load_success_set(path: str, allowed_filenames: Set[str]) -> Set[str]:
    with open(path) as f:
        data = json.load(f)
    return {
        filename
        for filename, info in data.items()
        if info.get("isCompiled") is True and filename in allowed_filenames
    }


# Paths relative to this script
claude_path = "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json"
o3mini_path = "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json"
deepseek_path = "mypy_outputs/mypy_results_deepseek_with_errors.json"
human_path = "mypy_outputs/mypy_results_original_files_with_errors.json"

claude_zero = load_success_set(claude_path, compiled_success_files)
o3mini_zero = load_success_set(o3mini_path, compiled_success_files)
deepseek_zero = load_success_set(deepseek_path, compiled_success_files)
human_zero = load_success_set(human_path, compiled_success_files)

# Create union of all LLM results
llm_union = claude_zero | o3mini_zero | deepseek_zero

# Create single figure with 4 horizontal subplots
total_base = len(compiled_success_files)

def fmt_label(name: str, s: Set[str]) -> str:
    count = len(s)
    pct = (count * 100.0 / total_base) if total_base else 0.0
    return f"{name}\n({count}/{total_base}, {pct:.1f}%)"

def create_venn_subplot(ax, human_set, llm_set, llm_name, colors):
    """Create a single Venn diagram subplot"""
    vd = venn2(
        [human_set, llm_set],
        set_labels=("", ""),  # Empty labels, we'll position them manually
        set_colors=colors, alpha=0.7,
        ax=ax
    )
    
    # Apply consistent fonts
    if vd.set_labels:
        for lbl in vd.set_labels:
            if lbl is not None:
                lbl.set_fontsize(14)
                lbl.set_fontweight('bold')
                lbl.set_clip_on(False)
    if vd.subset_labels:
        for lbl in vd.subset_labels:
            if lbl is not None:
                lbl.set_fontsize(14)
                lbl.set_fontweight('bold')
                lbl.set_clip_on(False)
    
    # Position labels manually: Human at top, LLM at bottom (closer to the figure)
    ax.text(0, 0.8, fmt_label("Human", human_set), ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -0.8, fmt_label(llm_name, llm_set), ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')

# Create single figure with 1x4 horizontal subplot layout
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Define the comparisons with their colors
comparisons = [
    (axes[0], human_zero, o3mini_zero, "o3-mini", (COLOR_MAP["Human"], COLOR_MAP["o3-mini"])),
    (axes[1], human_zero, claude_zero, "claude3 sonnet", (COLOR_MAP["Human"], COLOR_MAP["claude3 sonnet"])),
    (axes[2], human_zero, deepseek_zero, "deepseek", (COLOR_MAP["Human"], COLOR_MAP["deepseek"])),
    (axes[3], human_zero, llm_union, "Union of LLMs", (COLOR_MAP["Human"], COLOR_MAP["Union of 3 LLMs"])),
]

# Create each subplot
for ax, human_set, llm_set, llm_name, colors in comparisons:
    create_venn_subplot(ax, human_set, llm_set, llm_name, colors)

# Remove all spacing and padding
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# Save the single figure with no padding
plt.savefig("Section_6_Human_VS_LLM/venn_human_vs_all_llms_combined.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()

# Print statistics for all comparisons
print("\n" + "="*60)
print("COMPARISON STATISTICS")
print("="*60)

comparisons = [
    ("Human vs o3-mini", human_zero, o3mini_zero),
    ("Human vs Claude", human_zero, claude_zero),
    ("Human vs Deepseek", human_zero, deepseek_zero),
    ("Human vs Union", human_zero, llm_union)
]

for name, human_set, llm_set in comparisons:
    print(f"\n{name}:")
    print(f"  Human successful: {len(human_set)}")
    print(f"  LLM successful: {len(llm_set)}")
    print(f"  Only Human: {len(human_set - llm_set)}")
    print(f"  Only LLM: {len(llm_set - human_set)}")
    print(f"  Both: {len(human_set & llm_set)}")
    print(f"  Neither: {len(compiled_success_files - (human_set | llm_set))}")

