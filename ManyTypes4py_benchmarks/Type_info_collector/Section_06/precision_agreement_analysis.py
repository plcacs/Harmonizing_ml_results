#!/usr/bin/env python3
"""
precision_agreement_analysis.py

Compares type annotations from multiple LLMs against human-written types,
reporting parameter- and file-level agreement. Also evaluates an oracle-style
ensemble (“Union of Top 3”) that treats a parameter as covered if any of the
top-3 models annotated it and prefers an agreeing annotation when one exists.

Author: Mohammad Khan (rewritten for clarity and true-union logic)
"""

import json
import csv
import re
import os
from typing import Dict, List, Tuple, Set
import numpy as np

# ============================================================
# Configuration
# ============================================================

LLM_CONFIGS = {
    "gpt-3.5": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
    },
    "gpt-4o": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
    },
    "o1-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o1_mini_benchmarks.json",
    },
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
    },
    "Human": {
        "type_info_path": "../Type_info_LLMS/Type_info_original_files.json",
    },
}

# Optional: restrict to files that type-check in the untyped baseline corpus
FILTER_TO_COMPILED: bool = True
COMPILED_RESULTS_PATH: str = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"

# ============================================================
# IO helpers
# ============================================================

def _normalize_path(p: str) -> str:
    return p.replace("\\", "/") if isinstance(p, str) else p

def load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def load_compiled_true_files(path: str) -> Tuple[Set[str], Set[str]]:
    """
    Load filenames where isCompiled == true from a mypy results JSON.
    Returns (full_paths, basenames) to allow either match.
    """
    data = load_json(path)
    full_paths: Set[str] = set()
    basenames: Set[str] = set()

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("isCompiled") is True:
                name = entry.get("filename") or entry.get("file") or entry.get("path")
                if isinstance(name, str) and name:
                    norm = _normalize_path(name)
                    full_paths.add(norm)
                    basenames.add(os.path.basename(norm))
    elif isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and entry.get("isCompiled") is True and isinstance(name, str):
                norm = _normalize_path(name)
                full_paths.add(norm)
                basenames.add(os.path.basename(norm))

    return full_paths, basenames

def filter_data_by_compiled(data: Dict, compiled_full: Set[str], compiled_base: Set[str]) -> Dict:
    """
    Keep file entries whose normalized full path or basename appears in the compiled sets.
    """
    if not compiled_full and not compiled_base:
        return data
    filtered: Dict = {}
    for fn, v in data.items():
        norm = _normalize_path(fn)
        base = os.path.basename(norm)
        if norm in compiled_full or base in compiled_base:
            filtered[fn] = v
    return filtered

# ============================================================
# Type normalization & matching
# ============================================================

def normalize_type(type_str: str) -> str:
    if not isinstance(type_str, str):
        return ""
    t = type_str.strip().lower()
    t = re.sub(r'^typing\.', '', t)
    t = re.sub(r'^tp\.', '', t)
    t = re.sub(r'^t\.', '', t)
    return t

def normalize_union_types(type_str: str) -> str:
    """
    Canonicalize unions so that variations like Optional[int], int|None, Union[int, None]
    compare equal.
    """
    normalized = type_str.strip().lower()

    # Optional[X] -> Union[X, None]
    if normalized.startswith('optional[') and normalized.endswith(']'):
        inner = normalized[9:-1]
        return f"union[{inner}, none]"

    # A | B | C -> union[a,b,c]
    if '|' in normalized:
        parts = [p.strip() for p in normalized.split('|')]
        parts.sort()
        return f"union[{','.join(parts)}]"

    # Union[A, B, ...] -> union[a,b,...] (sorted)
    if normalized.startswith('union[') and normalized.endswith(']'):
        inner = normalized[6:-1]
        parts = [p.strip() for p in inner.split(',')]
        parts.sort()
        return f"union[{','.join(parts)}]"

    return normalized

def categorize_type(type_str: str) -> str:
    if not type_str:
        return "empty"

    normalized = normalize_type(type_str)

    if normalized in ['str', 'int', 'float', 'bool', 'bytes', 'complex']:
        return "primitive"

    if any(container in normalized for container in ['list', 'dict', 'set', 'tuple']):
        return "container"

    # Generic single-letter (T, KT, VT, etc.)
    if re.search(r'\b[A-Z][A-Z0-9_]*\b', normalized) and len(normalized) <= 3:
        return "generic"

    if 'union' in normalized or '|' in normalized:
        return "union"

    # Likely user-defined class (PascalCase without module)
    if re.match(r'^[A-Z][a-zA-Z0-9_]*$', type_str):
        return "custom"

    return "other"

def are_types_semantically_similar(human_type: str, llm_type: str) -> bool:
    """
    A pragmatic, precision-oriented notion of semantic similarity:
    - exact match after normalization
    - unions equal under canonicalization
    - Any ~ object
    - primitive ~ primitive (even if different primitives)
    - container base equality (list vs list, dict vs dict, etc.)
    """
    if not human_type or not llm_type:
        return False

    h = normalize_type(human_type)
    l = normalize_type(llm_type)

    h_u = normalize_union_types(h)
    l_u = normalize_union_types(l)

    if h == l or h_u == l_u:
        return True

    if h in ['any', 'object'] and l in ['any', 'object']:
        return True

    hcat = categorize_type(human_type)
    lcat = categorize_type(llm_type)

    if hcat == lcat == "primitive":
        return True

    if hcat == lcat == "container":
        hb = re.search(r'(list|dict|set|tuple)', h)
        lb = re.search(r'(list|dict|set|tuple)', l)
        if hb and lb and hb.group(1) == lb.group(1):
            return True

    return False

# ============================================================
# Analyses
# ============================================================

def analyze_parameter_agreement(human_data: Dict, llm_data: Dict, llm_name: str) -> Dict:
    """
    Compare types per parameter (excluding 'self') for files/functions common to both datasets.
    """
    total_common_params = 0
    agreement_count = 0
    disagreement_count = 0

    category_agreements = {
        "primitive": {"total": 0, "agreed": 0},
        "container": {"total": 0, "agreed": 0},
        "generic": {"total": 0, "agreed": 0},
        "union": {"total": 0, "agreed": 0},
        "custom": {"total": 0, "agreed": 0},
        "other": {"total": 0, "agreed": 0},
    }

    common_files = set(human_data.keys()) & set(llm_data.keys())

    for filename in common_files:
        hf = human_data.get(filename, {})
        lf = llm_data.get(filename, {})
        common_funcs = set(hf.keys()) & set(lf.keys())

        for func_name in common_funcs:
            hparams = hf.get(func_name, [])
            lparams = lf.get(func_name, [])
            if not isinstance(hparams, list) or not isinstance(lparams, list):
                continue

            hdict = {}
            for p in hparams:
                if not isinstance(p, dict):
                    continue
                name = p.get("name", "")
                cat = p.get("category", "")
                if cat == "arg" and isinstance(name, str) and name.strip().lower() == "self":
                    continue
                hdict[(cat, name)] = p

            ldict = {}
            for p in lparams:
                if not isinstance(p, dict):
                    continue
                name = p.get("name", "")
                cat = p.get("category", "")
                if cat == "arg" and isinstance(name, str) and name.strip().lower() == "self":
                    continue
                ldict[(cat, name)] = p

            for key in set(hdict.keys()) & set(ldict.keys()):
                hp = hdict[key]
                lp = ldict[key]
                ht = hp.get("type", [""])[0] if hp.get("type") else ""
                lt = lp.get("type", [""])[0] if lp.get("type") else ""

                if not ht.strip() or not lt.strip():
                    continue

                total_common_params += 1

                hcat = categorize_type(ht)
                if hcat in category_agreements:
                    category_agreements[hcat]["total"] += 1

                if are_types_semantically_similar(ht, lt):
                    agreement_count += 1
                    if hcat in category_agreements:
                        category_agreements[hcat]["agreed"] += 1
                else:
                    disagreement_count += 1

    cat_pct = {}
    for cat, cnts in category_agreements.items():
        cat_pct[cat] = (cnts["agreed"] / cnts["total"] * 100) if cnts["total"] > 0 else 0.0

    return {
        "llm_name": llm_name,
        "total_common_params": total_common_params,
        "agreement_count": agreement_count,
        "agreement_percentage": (agreement_count / total_common_params * 100) if total_common_params else 0.0,
        "disagreement_count": disagreement_count,
        "disagreement_percentage": (disagreement_count / total_common_params * 100) if total_common_params else 0.0,
        "category_agreements": cat_pct,
    }

def analyze_file_level_agreement(human_data: Dict, llm_data: Dict, llm_name: str) -> Dict:
    """
    File-level agreement ratio: for each common file,
    min(#params_human, #params_llm) / max(#params_human, #params_llm), averaged.
    """
    common_files = set(human_data.keys()) & set(llm_data.keys())
    file_ratios: List[float] = []

    def _count_non_self(params_list_or_dict) -> int:
        if not isinstance(params_list_or_dict, dict):
            return 0
        total = 0
        for plist in params_list_or_dict.values():
            if not isinstance(plist, list):
                continue
            for p in plist:
                if not isinstance(p, dict):
                    continue
                if p.get("category") == "arg" and str(p.get("name", "")).strip().lower() == "self":
                    continue
                total += 1
        return total

    for filename in common_files:
        hc = _count_non_self(human_data.get(filename, {}))
        lc = _count_non_self(llm_data.get(filename, {}))
        if hc > 0 and lc > 0:
            file_ratios.append(min(hc, lc) / max(hc, lc))

    avg_ratio = np.mean(file_ratios) * 100 if file_ratios else 0.0
    return {"llm_name": llm_name, "total_common_files": len(common_files), "avg_file_agreement": avg_ratio}

# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    all_data = {}
    for name, cfg in LLM_CONFIGS.items():
        data = load_json(cfg["type_info_path"])
        all_data[name] = data
        print(f"Loaded {name}: {len(data)} files")

    human_data = all_data["Human"]

    # Optional: narrow to files that compiled in the untyped baseline pass
    if FILTER_TO_COMPILED:
        compiled_full, compiled_base = load_compiled_true_files(COMPILED_RESULTS_PATH)
        if compiled_full or compiled_base:
            human_data = filter_data_by_compiled(human_data, compiled_full, compiled_base)
            for k in list(all_data.keys()):
                all_data[k] = filter_data_by_compiled(all_data[k], compiled_full, compiled_base)

    print("\n" + "=" * 80)
    print("PRECISION AGREEMENT ANALYSIS: LLMs vs Human")
    print("=" * 80)

    # Per-model results
    results = []
    for llm_name in ["gpt-3.5", "gpt-4o", "o1-mini", "o3-mini", "deepseek", "claude3-sonnet"]:
        if llm_name not in all_data:
            continue
        print(f"\nAnalyzing {llm_name}...")
        pa = analyze_parameter_agreement(human_data, all_data[llm_name], llm_name)
        fa = analyze_file_level_agreement(human_data, all_data[llm_name], llm_name)

        result = {
            "LLM": llm_name,
            "Common_Parameters": pa["total_common_params"],
            "Agreement_Count": pa["agreement_count"],
            "Agreement_Percentage": pa["agreement_percentage"],
            "Disagreement_Count": pa["disagreement_count"],
            "Disagreement_Percentage": pa["disagreement_percentage"],
            "Primitive_Agreement": pa["category_agreements"].get("primitive", 0.0),
            "Container_Agreement": pa["category_agreements"].get("container", 0.0),
            "Generic_Agreement": pa["category_agreements"].get("generic", 0.0),
            "Union_Agreement": pa["category_agreements"].get("union", 0.0),
            "Custom_Agreement": pa["category_agreements"].get("custom", 0.0),
            "File_Level_Agreement": fa["avg_file_agreement"],
        }
        results.append(result)

        print(f"  Common Parameters: {pa['total_common_params']:,}")
        print(f"  Agreement: {pa['agreement_count']:,} ({pa['agreement_percentage']:.1f}%)")
        print(f"  Disagreement: {pa['disagreement_count']:,} ({pa['disagreement_percentage']:.1f}%)")
        print(f"  File-level Agreement: {fa['avg_file_agreement']:.1f}%")

    # ------------------------------------------------------------
    # TRUE SUPERSET UNION (Oracle) of Top-3 models
    # ------------------------------------------------------------
    print("\nAnalyzing TRUE SUPERSET Union (Oracle) of Top-3 Models "
          "(Claude3-Sonnet + O3-mini + DeepSeek)...")

    top3 = ["claude3-sonnet", "o3-mini", "deepseek"]
    union_data: Dict[str, Dict[str, List[Dict]]] = {}

    # 1️⃣  Build a file/function superset of all top-3 models
    all_union_files = set()
    for m in top3:
        all_union_files |= set(all_data.get(m, {}).keys())

    for filename in all_union_files:
        union_data[filename] = {}
        # union across all functions appearing in any model
        all_union_funcs = set()
        for m in top3:
            mdata = all_data.get(m, {})
            if filename in mdata:
                all_union_funcs |= set(mdata[filename].keys())

        # 2️⃣  For every function in this file, collect parameter annotations
        for func_name in all_union_funcs:
            union_data[filename][func_name] = []

            # Collect every unique parameter (category,name) across models
            all_params: Dict[Tuple[str, str], List[Dict]] = {}
            for m in top3:
                mdata = all_data.get(m, {})
                if filename in mdata and func_name in mdata[filename]:
                    for p in mdata[filename][func_name]:
                        if not isinstance(p, dict):
                            continue
                        name = p.get("name", "")
                        cat = p.get("category", "")
                        if cat == "arg" and str(name).lower() == "self":
                            continue
                        key = (cat, name)
                        all_params.setdefault(key, []).append(p)

            # 3️⃣  For each parameter key, pick an annotation
            for (cat, name), candidates in all_params.items():
                # If Human has this param, prefer agreement check
                human_type = ""
                if filename in human_data and func_name in human_data[filename]:
                    for hp in human_data[filename][func_name]:
                        if (hp.get("category"), hp.get("name")) == (cat, name):
                            human_type = hp.get("type", [""])[0] if hp.get("type") else ""
                            break

                # Pick first candidate that agrees with Human, else first non-empty
                selected = None
                for cand in candidates:
                    ctype = cand.get("type", [""])[0] if cand.get("type") else ""
                    if human_type and are_types_semantically_similar(human_type, ctype):
                        selected = cand
                        break
                if not selected:
                    for cand in candidates:
                        ctype = cand.get("type", [""])[0] if cand.get("type") else ""
                        if ctype and ctype.strip():
                            selected = cand
                            break

                if selected:
                    union_data[filename][func_name].append(selected)

    # 4️⃣  Evaluate ensemble performance
    pa = analyze_parameter_agreement(human_data, union_data, "Union of Top 3")
    fa = analyze_file_level_agreement(human_data, union_data, "Union of Top 3")

    union_result = {
        "LLM": "Union of Top 3",
        "Common_Parameters": pa["total_common_params"],
        "Agreement_Count": pa["agreement_count"],
        "Agreement_Percentage": pa["agreement_percentage"],
        "Disagreement_Count": pa["disagreement_count"],
        "Disagreement_Percentage": pa["disagreement_percentage"],
        "Primitive_Agreement": pa["category_agreements"].get("primitive", 0.0),
        "Container_Agreement": pa["category_agreements"].get("container", 0.0),
        "Generic_Agreement": pa["category_agreements"].get("generic", 0.0),
        "Union_Agreement": pa["category_agreements"].get("union", 0.0),
        "Custom_Agreement": pa["category_agreements"].get("custom", 0.0),
        "File_Level_Agreement": fa["avg_file_agreement"],
    }
    results.append(union_result)

    print(f"  Common Parameters: {pa['total_common_params']:,}")
    print(f"  Agreement: {pa['agreement_count']:,} ({pa['agreement_percentage']:.1f}%)")
    print(f"  Disagreement: {pa['disagreement_count']:,} ({pa['disagreement_percentage']:.1f}%)")
    print(f"  File-level Agreement: {fa['avg_file_agreement']:.1f}%")


    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    output_file = "precision_agreement_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "LLM", "Common_Parameters", "Agreement_Count", "Agreement_Percentage",
            "Disagreement_Count", "Disagreement_Percentage", "Primitive_Agreement",
            "Container_Agreement", "Generic_Agreement", "Union_Agreement",
            "Custom_Agreement", "File_Level_Agreement"
        ])
        for r in results:
            writer.writerow([
                r["LLM"],
                r["Common_Parameters"],
                r["Agreement_Count"],
                f"{r['Agreement_Percentage']:.2f}",
                r["Disagreement_Count"],
                f"{r['Disagreement_Percentage']:.2f}",
                f"{r['Primitive_Agreement']:.2f}",
                f"{r['Container_Agreement']:.2f}",
                f"{r['Generic_Agreement']:.2f}",
                f"{r['Union_Agreement']:.2f}",
                f"{r['Custom_Agreement']:.2f}",
                f"{r['File_Level_Agreement']:.2f}",
            ])
    print(f"\nResults saved to {output_file}")

    # Console summary
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = (
        f"{'LLM':<20} {'Common':<8} {'Agree':<8} {'Agree%':<8} "
        f"{'Disagree':<10} {'Disagree%':<10} {'Primitive':<10} {'Container':<10} "
        f"{'Generic':<10} {'Union':<8} {'Custom':<8} {'File%':<8}"
    )
    print(header)
    print("-" * 125)
    for r in results:
        print(
            f"{r['LLM']:<20} "
            f"{r['Common_Parameters']:<8,} "
            f"{r['Agreement_Count']:<8,} "
            f"{r['Agreement_Percentage']:<8.1f} "
            f"{r['Disagreement_Count']:<10,} "
            f"{r['Disagreement_Percentage']:<10.1f} "
            f"{r['Primitive_Agreement']:<10.1f} "
            f"{r['Container_Agreement']:<10.1f} "
            f"{r['Generic_Agreement']:<10.1f} "
            f"{r['Union_Agreement']:<8.1f} "
            f"{r['Custom_Agreement']:<8.1f} "
            f"{r['File_Level_Agreement']:<8.1f}"
        )

if __name__ == "__main__":
    main()
