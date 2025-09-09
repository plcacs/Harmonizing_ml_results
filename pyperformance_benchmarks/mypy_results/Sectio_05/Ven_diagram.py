import json
from typing import Set
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

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
deepseek_path = "mypy_outputs/mypy_results_deepseek_1st_run_with_errors.json"

claude_zero = load_success_set(claude_path, compiled_success_files)
o3mini_zero = load_success_set(o3mini_path, compiled_success_files)
deepseek_zero = load_success_set(deepseek_path, compiled_success_files)

# Plot Venn diagram using only compiled-success files
plt.figure(figsize=(8, 8))
total_base = len(compiled_success_files)

def fmt_label(name: str, s: Set[str]) -> str:
    count = len(s)
    pct = (count * 100.0 / total_base) if total_base else 0.0
    return f"{name} ({count}/{total_base}, {pct:.1f}%)"

v = venn3(
    [claude_zero, o3mini_zero, deepseek_zero],
    set_labels=(
        fmt_label("Claude 3 Sonnet", claude_zero),
        fmt_label("o3-mini", o3mini_zero),
        fmt_label("Deepseek", deepseek_zero),
    ),
)

# Make numbers in circles larger and bold
for text in v.set_labels:
    if text:
        text.set_fontsize(14)
        text.set_fontweight('bold')
for text in v.subset_labels:
    if text:
        text.set_fontsize(14)
        text.set_fontweight('bold')

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig("venn_diagram_compiled_successes_3_models_pyperformance.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()
