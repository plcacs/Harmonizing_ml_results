import json
from typing import Dict, List, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(__file__))
from semantic_type_comparison import types_equivalent_semantic

def load_annotations(file_path: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    with open(file_path, "r") as f:
        return json.load(f)

def normalize_type(type_str: str) -> str:
    if not type_str:
        return "dyn"
    return type_str.replace(" ", "").replace("'", "").replace('"', "").lower()

def compare_llm_annotations(file1: str, file2: str, llm1_name: str, llm2_name: str, output_dir: str = "."):
    ann1 = load_annotations(file1)
    ann2 = load_annotations(file2)
    exact = semantic = total = 0
    details = []
    # Find common files
    common_files = set(ann1.keys()) & set(ann2.keys())
    for filename in common_files:
        funcs1 = ann1[filename]
        funcs2 = ann2[filename]
        if not isinstance(funcs1, dict) or not isinstance(funcs2, dict):
            continue
        common_funcs = set(funcs1.keys()) & set(funcs2.keys())
        for func_sig in common_funcs:
            params1 = funcs1[func_sig]
            params2 = funcs2[func_sig]
            if not isinstance(params1, list) or not isinstance(params2, list):
                continue
            for item1 in params1:
                if not isinstance(item1, dict):
                    continue
                for item2 in params2:
                    if not isinstance(item2, dict):
                        continue
                    if item1.get("category") == item2.get("category") and item1.get("name") == item2.get("name"):
                        t1 = item1.get("type", [])
                        t2 = item2.get("type", [])
                        if (
                            isinstance(t1, list) and isinstance(t2, list)
                            and len(t1) > 0 and len(t2) > 0
                            and isinstance(t1[0], str) and isinstance(t2[0], str)
                        ):
                            type1 = t1[0]
                            type2 = t2[0]
                            if (
                                not type1.strip() or not type2.strip()
                                or type1.strip().lower() == "none" or type2.strip().lower() == "none"
                                or type1.strip().lower() == "any" or type2.strip().lower() == "any"
                            ):
                                continue
                            total += 1
                            is_exact = normalize_type(type1) == normalize_type(type2)
                            is_semantic = types_equivalent_semantic(type1, type2)
                            if is_exact:
                                exact += 1
                            if is_semantic:
                                semantic += 1
                            details.append({
                                "filename": filename,
                                "function": func_sig,
                                "category": item1.get("category"),
                                "name": item1.get("name"),
                                "Run_1": type1,
                                "Run_2": type2,
                                "exact_match": is_exact,
                                "semantic_match": is_semantic
                            })
                        break
    print(f"Exact matches: {exact}/{total} ({(exact/total*100 if total else 0):.2f}%)")
    print(f"Semantic matches: {semantic}/{total} ({(semantic/total*100 if total else 0):.2f}%)")
    # Optionally save details
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"llm_vs_llm_comparison_{llm1_name}_vs_{llm2_name}.json")
    with open(output_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    file1s = ["Type_info_deep_seek_2nd_run_benchmarks.json", "Type_info_gpt4o_2nd_run_benchmarks.json", "Type_info_o1_mini_2nd_run_benchmarks.json"]
    file2s = ["Type_info_deep_seek_benchmarks.json", "Type_info_gpt4o_benchmarks.json", "Type_info_o1_mini_benchmarks.json"]
    llm1_names = ["DeepSeek_2nd_run", "GPT-4o_2nd_run", "o1_mini_2nd_run"]
    llm2_names = ["DeepSeek", "GPT-4o", "o1_mini"]
    for file1, file2, llm1_name, llm2_name in zip(file1s, file2s, llm1_names, llm2_names):
        compare_llm_annotations(file1, file2, llm1_name, llm2_name, output_dir="llm_vs_llm_comparison") 
    