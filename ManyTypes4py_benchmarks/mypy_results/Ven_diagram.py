import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Load JSON files
with open("mypy_results_deepseek_with_errors.json") as f1, \
     open("mypy_results_gpt4o_with_errors.json") as f2, \
     open("mypy_results_o1_mini_with_errors.json") as f3:
    deepseek = json.load(f1)
    gpt4o = json.load(f2)
    o1_mini = json.load(f3)

# Get sets where error_count == 0
deepseek_zero = {fname for fname, v in deepseek.items() if v.get("error_count", 0) == 0}
gpt4o_zero = {fname for fname, v in gpt4o.items() if v.get("error_count", 0) == 0}
o1_mini_zero = {fname for fname, v in o1_mini.items() if v.get("error_count", 0) == 0}

# Plot 3-way Venn diagram
plt.figure(figsize=(8, 8))
venn3(
    [gpt4o_zero,o1_mini_zero,deepseek_zero ],
    set_labels=('GPT-4o','O1-mini','Deepseek')
)
plt.title("Venn Diagram of Files with error_count == 0 (ManyTypes4Py)")
plt.tight_layout()
plt.savefig("venn_diagram_llms_ManyTypes4Py.pdf", bbox_inches='tight')
#plt.show()

