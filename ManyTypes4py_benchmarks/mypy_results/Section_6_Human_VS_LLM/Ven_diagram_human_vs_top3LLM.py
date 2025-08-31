import json
from typing import Set
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

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

# Create 4 individual Venn diagrams
plt.style.use('default')
total_base = len(compiled_success_files)

def fmt_label(name: str, s: Set[str]) -> str:
    count = len(s)
    pct = (count * 100.0 / total_base) if total_base else 0.0
    return f"{name}\n({count}/{total_base}, {pct:.1f}%)"

# 1. Human vs o3-mini
plt.figure(figsize=(12, 10))
venn2([human_zero, o3mini_zero], 
      set_labels=(fmt_label("Human", human_zero), fmt_label("o3-mini", o3mini_zero)),
      set_colors=('#2E86AB', '#A23B72'), alpha=0.7)
plt.title("Human vs o3-mini: Compiled Successes Among Baseline Files", 
          fontsize=18, fontweight='bold', pad=20)
# Increase font size for numbers and percentages
for text in plt.gca().texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
plt.tight_layout()
plt.savefig("Section_6_Human_VS_LLM/venn_human_vs_o3mini.pdf", bbox_inches='tight')
plt.show()

# 2. Human vs Claude
plt.figure(figsize=(12, 10))
venn2([human_zero, claude_zero], 
      set_labels=(fmt_label("Human", human_zero), fmt_label("Claude 3 Sonnet", claude_zero)),
      set_colors=('#2E86AB', '#F18F01'), alpha=0.7)
plt.title("Human vs Claude 3 Sonnet: Compiled Successes Among Baseline Files", 
          fontsize=18, fontweight='bold', pad=20)
# Increase font size for numbers and percentages
for text in plt.gca().texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
plt.tight_layout()
plt.savefig("Section_6_Human_VS_LLM/venn_human_vs_claude.pdf", bbox_inches='tight')
plt.show()

# 3. Human vs Deepseek
plt.figure(figsize=(12, 10))
venn2([human_zero, deepseek_zero], 
      set_labels=(fmt_label("Human", human_zero), fmt_label("Deepseek", deepseek_zero)),
      set_colors=('#2E86AB', '#C73E1D'), alpha=0.7)
plt.title("Human vs Deepseek: Compiled Successes Among Baseline Files", 
          fontsize=18, fontweight='bold', pad=20)
# Increase font size for numbers and percentages
for text in plt.gca().texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
plt.tight_layout()
plt.savefig("Section_6_Human_VS_LLM/venn_human_vs_deepseek.pdf", bbox_inches='tight')
plt.show()

# 4. Human vs Union of all 3 LLMs
plt.figure(figsize=(12, 10))
venn2([human_zero, llm_union], 
      set_labels=(fmt_label("Human", human_zero), fmt_label("Union of LLMs", llm_union)),
      set_colors=('#2E86AB', '#8B5A96'), alpha=0.7)
plt.title("Human vs Union of All 3 LLMs: Compiled Successes Among Baseline Files", 
          fontsize=18, fontweight='bold', pad=20)
# Increase font size for numbers and percentages
for text in plt.gca().texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
plt.tight_layout()
plt.savefig("Section_6_Human_VS_LLM/venn_human_vs_union.pdf", bbox_inches='tight')
plt.show()

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

