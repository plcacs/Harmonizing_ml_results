import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration for the 3 LLMs to compare
LLM_CONFIGS = {
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_user_annotated_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deepseek_user_annotated_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_user_annotated_benchmarks.json",
    },
}

# Optional filtering: consider only files with isCompiled == true from mypy results
FILTER_TO_COMPILED: bool = True  # set True to enable filtering
COMPILED_RESULTS_PATH: str = (
    "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
)


def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def is_any_or_blank(type_annotation: str) -> bool:
    """Check if a type annotation is 'Any' or blank/empty."""
    if not type_annotation or not isinstance(type_annotation, str):
        return True

    annotation = type_annotation.strip().lower()
    return annotation in ["any", "tp.any", "typing.any", ""]


def extract_parameter_annotations(type_info: Dict) -> List[bool]:
    """Extract parameter annotations from type info, returning list of is_any_or_blank flags."""
    annotations = []

    if not isinstance(type_info, dict):
        return annotations

    for func_name, items in type_info.items():
        if not isinstance(items, list):
            continue

        for entry in items:
            if not isinstance(entry, dict):
                continue

            tlist = entry.get("type", [])
            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                if isinstance(t0, str):
                    annotations.append(is_any_or_blank(t0))
                else:
                    annotations.append(True)  # Treat non-string as Any/blank
            else:
                annotations.append(True)  # Treat empty list as Any/blank

    return annotations


def load_compiled_true_files(path: str) -> Tuple[Set[str], Set[str]]:
    """Load filenames where isCompiled == true. Returns (full_paths, basenames)."""
    data = load_json(path)
    full_paths: Set[str] = set()
    basenames: Set[str] = set()

    def add_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            return
        norm = name.replace("\\", "/")
        full_paths.add(norm)
        basenames.add(os.path.basename(norm))

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("isCompiled") is True:
                name = entry.get("filename") or entry.get("file") or entry.get("path")
                add_name(name)
    elif isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and entry.get("isCompiled") is True:
                add_name(name)

    return full_paths, basenames


def calculate_agreement_stats(
    llm_annotations: Dict[str, Dict[str, List[bool]]],
) -> Dict[str, float]:
    """Calculate detailed agreement statistics between LLMs."""
    llm_names = list(llm_annotations.keys())
    if len(llm_names) != 3:
        raise ValueError("Expected exactly 3 LLMs for comparison")

    # Get common files across all LLMs
    common_files = set(llm_annotations[llm_names[0]].keys())
    for llm_name in llm_names[1:]:
        common_files = common_files.intersection(set(llm_annotations[llm_name].keys()))

    print(f"Found {len(common_files)} common files across all LLMs")

    total_parameters = 0
    all_agree_any = 0  # All use Any/blank
    two_agree_any = 0  # Two use Any/blank, one uses typed
    all_disagree_any = 0  # All use typed (no Any/blank)

    # Detailed two-agree statistics for Any/blank
    two_agree_any_stats = {
        f"{llm_names[0]}_and_{llm_names[1]}": 0,
        f"{llm_names[0]}_and_{llm_names[2]}": 0,
        f"{llm_names[1]}_and_{llm_names[2]}": 0,
    }

    for filename in common_files:
        # Get annotations for this file from all LLMs
        file_annotations = {}
        for llm_name in llm_names:
            file_annotations[llm_name] = llm_annotations[llm_name][filename]

        # Compare parameter by parameter
        max_params = max(len(anns) for anns in file_annotations.values())

        for param_idx in range(max_params):
            param_annotations = {}
            for llm_name in llm_names:
                if param_idx < len(file_annotations[llm_name]):
                    param_annotations[llm_name] = file_annotations[llm_name][param_idx]
                else:
                    param_annotations[llm_name] = True  # Treat missing as Any/blank

            total_parameters += 1

            # Check Any/blank agreement patterns only
            any_count = sum(param_annotations.values())

            if any_count == 3:  # All use Any/blank
                all_agree_any += 1
            elif any_count == 0:  # All use typed (no Any/blank)
                all_disagree_any += 1
            elif any_count == 2:  # Two use Any/blank, one uses typed
                two_agree_any += 1
                # Find which two agree on Any/blank
                any_llms = [
                    name for name, is_any in param_annotations.items() if is_any
                ]
                if len(any_llms) == 2:
                    key = f"{any_llms[0]}_and_{any_llms[1]}"
                    if key in two_agree_any_stats:
                        two_agree_any_stats[key] += 1
            elif any_count == 1:  # One uses Any/blank, two use typed
                two_agree_any += 1
                # Find which two agree on typed (disagree on Any/blank)
                typed_llms = [
                    name for name, is_any in param_annotations.items() if not is_any
                ]
                if len(typed_llms) == 2:
                    key = f"{typed_llms[0]}_and_{typed_llms[1]}"
                    if key in two_agree_any_stats:
                        two_agree_any_stats[key] += 1

    # Calculate percentages
    stats = {
        "all_agree_any": (
            (all_agree_any / total_parameters * 100) if total_parameters > 0 else 0
        ),
        "two_agree_any": (
            (two_agree_any / total_parameters * 100) if total_parameters > 0 else 0
        ),
        "all_disagree_any": (
            (all_disagree_any / total_parameters * 100) if total_parameters > 0 else 0
        ),
        "total_parameters": total_parameters,
    }

    # Add detailed two-agree percentages for Any/blank
    for key, count in two_agree_any_stats.items():
        stats[f"two_agree_any_{key}"] = (
            (count / total_parameters * 100) if total_parameters > 0 else 0
        )

    return stats


def main():
    """Main analysis function for LLM agreement on Any/blank annotations."""
    print("Loading LLM data...")

    # Load all LLM data
    llm_data = {}
    for llm_name, config in LLM_CONFIGS.items():
        type_info = load_json(config["type_info_path"])
        llm_data[llm_name] = type_info
        print(f"Loaded {llm_name}: {len(type_info)} files")

    # Optional: filter to compiled files only
    compiled_full: Set[str] = set()
    compiled_base: Set[str] = set()
    if FILTER_TO_COMPILED:
        compiled_full, compiled_base = load_compiled_true_files(COMPILED_RESULTS_PATH)
        print(f"Filtering to compiled files: {len(compiled_full)} entries")

    # Extract parameter annotations for each LLM
    llm_annotations = {}
    for llm_name, type_info in llm_data.items():
        print(f"Extracting annotations for {llm_name}...")
        file_annotations = {}

        for filename, file_info in type_info.items():
            if FILTER_TO_COMPILED:
                norm_name = filename.replace("\\", "/")
                base = os.path.basename(norm_name)
                if norm_name not in compiled_full and base not in compiled_base:
                    continue

            annotations = extract_parameter_annotations(file_info)
            file_annotations[filename] = annotations

        llm_annotations[llm_name] = file_annotations
        total_params = sum(len(anns) for anns in file_annotations.values())
        print(
            f"  {llm_name}: {len(file_annotations)} files, {total_params} total parameters"
        )

    # Calculate agreement statistics
    print("\nCalculating agreement statistics...")
    stats = calculate_agreement_stats(llm_annotations)

    # Print results
    print("\n" + "=" * 80)
    print("LLM AGREEMENT ANALYSIS (Any/Blank Usage)")
    print("=" * 80)
    print(f"Total parameters analyzed: {stats['total_parameters']}")
    print(f"All 3 LLMs agree on Any/blank: {stats['all_agree_any']:.1f}%")
    print(f"2 LLMs agree, 1 disagrees on Any/blank: {stats['two_agree_any']:.1f}%")
    print(f"All 3 LLMs disagree on Any/blank: {stats['all_disagree_any']:.1f}%")

    print(f"\nDetailed Two-Agreement Statistics (on Any/blank usage):")
    llm_names = list(LLM_CONFIGS.keys())
    for i in range(len(llm_names)):
        for j in range(i + 1, len(llm_names)):
            key = f"two_agree_any_{llm_names[i]}_and_{llm_names[j]}"
            if key in stats:
                print(
                    f"  {llm_names[i]} & {llm_names[j]} agree on Any/blank: {stats[key]:.1f}%"
                )


if __name__ == "__main__":
    main()
