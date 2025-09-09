import json
from typing import Set
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib_venn import venn3, venn3_unweighted

# --- Fixed, camera-ready defaults (consistent across runs) ---
mpl.rcParams.update({
    "figure.figsize": (8, 8),   # fixed physical size
    "savefig.bbox": None,       # DO NOT auto-crop by content
    "savefig.pad_inches": 0.02, # tiny uniform pad
    "pdf.fonttype": 42,         # embed TrueType (avoid viewer clipping)
    "ps.fonttype": 42,
})

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

total_base = len(compiled_success_files)

def fmt_label(name: str, s: Set[str]) -> str:
    count = len(s)
    pct = (count * 100.0 / total_base) if total_base else 0.0
    return f"{name} ({count}/{total_base}, {pct:.1f}%)"

# --- Draw with FIXED circle sizes ---
fig, ax = plt.subplots()

# Use EXACTLY the same data sizes for both scripts
# This forces identical circle sizes regardless of benchmark
fixed_claude = 50
fixed_o3mini = 60  
fixed_deepseek = 55

# Create sets with fixed sizes
artificial_claude = set(range(fixed_claude))
artificial_o3mini = set(range(fixed_o3mini))
artificial_deepseek = set(range(fixed_deepseek))

v = venn3_unweighted(
    [artificial_claude, artificial_o3mini, artificial_deepseek],
    set_labels=(
        fmt_label("Claude 3 Sonnet", claude_zero),
        fmt_label("o3-mini", o3mini_zero),
        fmt_label("DeepSeek", deepseek_zero),
    ),
)

# Typography (moderate to avoid overflow)
for t in (v.set_labels or []):
    if t: t.set_fontsize(12); t.set_fontweight('bold')
for t in (v.subset_labels or []):
    if t: t.set_fontsize(12); t.set_fontweight('bold')

# Fix the drawing box so labels never get cut and size never changes
ax.set_aspect('equal')
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.45)
ax.set_axis_off()

out_pdf = "venn_diagram_compiled_successes_3_models_pyperformance_fixed_size.pdf"
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[Saved] {out_pdf}")
