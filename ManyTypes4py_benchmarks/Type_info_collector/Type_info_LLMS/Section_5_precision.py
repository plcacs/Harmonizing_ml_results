import json
import csv
from typing import Dict, List, Tuple, Set

# Simple, CSV-first script. Uses relative paths and does not auto-run external tools.

# Inputs (relative to this script)
UNTYPED_MYPY_PATH = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"

LLM_MYPY_PATHS: Dict[str, str] = {
    "gpt-3.5": "../../mypy_results/mypy_outputs/mypy_results_gpt35_1st_run_with_errors.json",
    "gpt-4o": "../../mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
    "o1-mini": "../../mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
    "o3-mini": "../../mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
    "deepseek": "../../mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
    "claude3-sonnet": "../../mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
}

TYPE_INFO_PATHS: Dict[str, str] = {
    "gpt-3.5": "./Type_info_gpt35_1st_run_benchmarks.json",
    "gpt-4o": "./Type_info_gpt4o_benchmarks.json",
    "o1-mini": "./Type_info_o1_mini_benchmarks.json",
    "o3-mini": "./Type_info_o3_mini_1st_run_benchmarks.json",
    "deepseek": "./Type_info_deep_seek_benchmarks.json",
    "claude3-sonnet": "./Type_info_claude3_sonnet_1st_run_benchmarks.json",
}

# Output CSVs
PER_FILE_CSV = "./precision_points_per_file.csv"
PER_LLM_CSV = "./precision_points_per_llm.csv"


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_files(untyped_mypy: Dict) -> Set[str]:
    return {fname for fname, info in untyped_mypy.items() if info.get("isCompiled") is True}


def count_any_and_typed_slots(type_info_for_file: Dict) -> Tuple[int, int]:
    """Return (any_slots, typed_slots) over parameters and returns.
    Expects structure: { func_name: [ {"category": "arg"|"return", "type": ["..."]}, ... ] }
    """
    any_slots = 0
    typed_slots = 0
    if not isinstance(type_info_for_file, dict):
        return any_slots, typed_slots

    for _func_name, items in type_info_for_file.items():
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            tlist = entry.get("type", [])
            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                if isinstance(t0, str) and t0.strip():
                    typed_slots += 1
                    if t0.strip().lower() == "any":
                        any_slots += 1
    return any_slots, typed_slots


def compute_non_any_ratio(any_slots: int, typed_slots: int) -> float:
    if typed_slots <= 0:
        return -1.0  # indicates no data
    return 1.0 - (any_slots / typed_slots)


def main() -> None:
    # Load baseline
    untyped_mypy = load_json(UNTYPED_MYPY_PATH)
    baseline_files = build_baseline_files(untyped_mypy)

    # Load LLM artifacts
    llm_mypy: Dict[str, Dict] = {name: load_json(path) for name, path in LLM_MYPY_PATHS.items()}
    llm_typeinfo: Dict[str, Dict] = {name: load_json(path) for name, path in TYPE_INFO_PATHS.items()}

    # Prepare accumulators
    per_llm_points: Dict[str, float] = {name: 0.0 for name in LLM_MYPY_PATHS.keys()}
    per_file_rows: List[List] = []  # [filename, winner_list, points_each, {llm:score...}]

    # Iterate files in baseline
    for filename in baseline_files:
        # For this file, gather each LLM's precision score (non-Any ratio)
        llm_scores: Dict[str, float] = {}
        for llm_name in LLM_MYPY_PATHS.keys():
            # Ensure this LLM produced this file
            if filename not in llm_typeinfo.get(llm_name, {}):
                continue
            typeinfo_file = llm_typeinfo[llm_name][filename]
            any_slots, typed_slots = count_any_and_typed_slots(typeinfo_file)
            score = compute_non_any_ratio(any_slots, typed_slots)
            if score >= 0.0:
                llm_scores[llm_name] = score

        if not llm_scores:
            # No LLM produced typed data for this file
            continue

        # Find max score and award split points
        max_score = max(llm_scores.values())
        winners = [m for m, s in llm_scores.items() if s == max_score]
        split = 1.0 / len(winners)
        for w in winners:
            per_llm_points[w] += split

        # Save per-file details (compact)
        per_file_rows.append([
            filename,
            ";".join(winners),
            f"{split:.3f}",
            *(f"{m}:{llm_scores[m]:.3f}" for m in sorted(llm_scores.keys()))
        ])

    # Write per-file CSV
    with open(PER_FILE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "winners", "points_each", "scores..."])
        writer.writerows(per_file_rows)

    # Write per-LLM CSV
    with open(PER_LLM_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["llm", "precision_points"])
        for llm_name in sorted(per_llm_points.keys()):
            writer.writerow([llm_name, f"{per_llm_points[llm_name]:.3f}"])

    print(f"Wrote {PER_FILE_CSV} and {PER_LLM_CSV}")


if __name__ == "__main__":
    main()
