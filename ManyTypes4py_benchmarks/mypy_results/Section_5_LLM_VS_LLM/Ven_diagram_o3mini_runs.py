import json
from typing import Set
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted

# Build the set of compiled-success files from untyped results
with open("mypy_outputs/mypy_results_untyped_with_errors.json") as f_untyped:
    untyped_results = json.load(f_untyped)

compiled_success_files = {
    filename for filename, info in untyped_results.items() if info.get("isCompiled") is True
}
print(f"Number of compiled-success files (untyped): {len(compiled_success_files)}")


def load_success_set(path: str, allowed_filenames: Set[str]) -> Set[str]:
    with open(path) as f:
        data = json.load(f)
    return {
        filename
        for filename, info in data.items()
        if info.get("isCompiled") is True and filename in allowed_filenames
    }


# Paths relative to this script (o3-mini only)
o3mini_path1 = "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json"
o3mini_path2 = "mypy_outputs/mypy_results_o3_mini_2nd_run_with_errors.json"
o3mini_path3 = "mypy_outputs/mypy_results_o3_mini_3rd_run_with_errors.json"

o3mini_run1 = load_success_set(o3mini_path1, compiled_success_files)
o3mini_run2 = load_success_set(o3mini_path2, compiled_success_files)
o3mini_run3 = load_success_set(o3mini_path3, compiled_success_files)

# Plot Venn diagram using only files that compile in untyped and each run
plt.figure(figsize=(8, 8))
total_base = len(compiled_success_files)


v = venn3_unweighted(
    [o3mini_run1, o3mini_run2, o3mini_run3],
    set_labels=(
        "1st run",
        "2nd run",
        "3rd run",
    ),
)

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
# Per user preference, do not save figure; only display it
#plt.show()
plt.savefig("Section_5_LLM_VS_LLM/o3mini_venn_diagram.pdf", bbox_inches="tight")


