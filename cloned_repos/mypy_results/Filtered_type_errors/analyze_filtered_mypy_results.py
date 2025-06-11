
import json
from collections import Counter

def analyze_mypy_json(filepath):
    with open(filepath) as f:
        data = json.load(f)

    categories = {
        "base=0 & llm>0": [],
        "base>0 & llm>0": [],
        "base=0 & llm=0": [],
        "base>0 & llm=0": []
    }

    error_keywords = ["assignment", "arg-type", "return", "call-arg", "var-annotated", "attr-defined", "name-defined", "type-arg", "misc"]
    llm_error_type_counts = Counter()

    for fname, entry in data.items():
        b_err = entry["base_error_count"]
        l_err = entry["llm_error_count"]

        if b_err == 0 and l_err > 0:
            categories["base=0 & llm>0"].append(fname)
            for err in entry["llm_errors"]:
                for key in error_keywords:
                    if f"[{key}]" in err:
                        llm_error_type_counts[key] += 1
                        break
        elif b_err > 0 and l_err > 0:
            categories["base>0 & llm>0"].append(fname)
        elif b_err == 0 and l_err == 0:
            categories["base=0 & llm=0"].append(fname)
        elif b_err > 0 and l_err == 0:
            categories["base>0 & llm=0"].append(fname)

    print(f"base=0 & llm>0: {len(categories['base=0 & llm>0'])} files")
    print(f"base>0 & llm>0: {len(categories['base>0 & llm>0'])} files")
    print(f"base=0 & llm=0: {len(categories['base=0 & llm=0'])} files")
    print(f"base>0 & llm=0: {len(categories['base>0 & llm=0'])} files\n")

    print("LLM Error Types in 'base=0 & llm>0':")
    for key, count in llm_error_type_counts.items():
        print(f"  {key}: {count}")

# Example usage:
json_files = ["merged_mypy_results_gpt4o.json", "merged_mypy_results_o1_mini.json", "merged_mypy_results_deepseek.json"]
for json_file in json_files:
    print(f"Analyzing {json_file}")
    analyze_mypy_json(json_file)
    print("\n")
