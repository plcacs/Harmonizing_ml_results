import json
from typing import Dict, Set, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_files(untyped_path: str) -> Set[str]:
    """Replicates the baseline selection used in barplot_for_partial_user_annotated.py.

    Baseline is the set of filenames that compiled successfully in the
    untyped baseline results.
    """
    data = load_json(untyped_path)
    baseline: Set[str] = set()
    for filename, info in data.items():
        if info.get("isCompiled") is True:
            baseline.add(filename)
    return baseline


def compute_truth_table(
    data_a: Dict[str, Dict],
    data_b: Dict[str, Dict],
    baseline_files: Set[str],
) -> Tuple[int, int, int, int]:
    """Return (both_success, only_a, only_b, neither)."""
    both_success = 0
    only_a = 0
    only_b = 0
    neither = 0

    for filename in baseline_files:
        a_info = data_a.get(filename)
        b_info = data_b.get(filename)
        a_ok = bool(a_info and a_info.get("isCompiled") is True)
        b_ok = bool(b_info and b_info.get("isCompiled") is True)

        if a_ok and b_ok:
            both_success += 1
        elif a_ok and not b_ok:
            only_a += 1
        elif b_ok and not a_ok:
            only_b += 1
        else:
            neither += 1

    return both_success, only_a, only_b, neither


def print_truth_table(
    title: str, a_label: str, b_label: str, counts: Tuple[int, int, int, int]
) -> None:
    both_success, only_a, only_b, neither = counts
    print(f"\n=== {title} ===")
    print(f"Both {a_label} & {b_label} success: {both_success}")
    print(f"Only {a_label} success: {only_a}")
    print(f"Only {b_label} success: {only_b}")
    print(f"Neither success: {neither}")


def main() -> None:
    # Baseline (same source as in the referenced script)
    untyped_path = "mypy_outputs/mypy_results_untyped_with_errors.json"

    llm_paths_fullytyped = {
        "o3-mini": "mypy_outputs/mypy_results_o3_mini_user_annotated_with_errors.json",
        "deepseek": "mypy_outputs/mypy_results_deepseek_user_annotated_with_errors.json",
        "claude3 sonnet": "mypy_outputs/mypy_results_claude3_sonnet_user_annotated_with_errors.json",
    }

    llm_paths_partial = {
        "o3-mini": "mypy_outputs/partial_typed/mypy_results_o3_mini_partially_typed_files_with_errors.json",
        "deepseek": "mypy_outputs/partial_typed/mypy_results_deepseek_partially_typed_files_with_errors.json",
        "claude3 sonnet": "mypy_outputs/partial_typed/mypy_results_claude3_sonnet_partially_typed_files_with_errors.json",
    }

    llm_paths_untyped = {
        "o3-mini": "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "deepseek": "mypy_outputs/mypy_results_deepseek_with_errors.json",
        "claude3 sonnet": "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
    }

    baseline_files = build_baseline_files(untyped_path)
    if not baseline_files:
        print("No baseline files found (no compiled entries in untyped baseline).")
        return

    # Preload all datasets
    fullytyped_data = {k: load_json(v) for k, v in llm_paths_fullytyped.items()}
    partial_data = {k: load_json(v) for k, v in llm_paths_partial.items()}
    untyped_data = {k: load_json(v) for k, v in llm_paths_untyped.items()}

    # 1) fullytyped vs untyped
    print("\n######## Truth Tables: fullytyped vs untyped ########")
    for llm in llm_paths_fullytyped.keys():
        counts = compute_truth_table(
            fullytyped_data[llm], untyped_data[llm], baseline_files
        )
        print_truth_table(
            title=f"{llm}: fullytyped vs untyped",
            a_label="fullytyped",
            b_label="untyped",
            counts=counts,
        )

    # 2) partial vs untyped
    print("\n######## Truth Tables: partial vs untyped ########")
    for llm in llm_paths_partial.keys():
        counts = compute_truth_table(
            partial_data[llm], untyped_data[llm], baseline_files
        )
        print_truth_table(
            title=f"{llm}: partial vs untyped",
            a_label="partial",
            b_label="untyped",
            counts=counts,
        )

    # 3) fullytyped vs partial
    print("\n######## Truth Tables: fullytyped vs partial ########")
    for llm in llm_paths_fullytyped.keys():
        counts = compute_truth_table(
            fullytyped_data[llm], partial_data[llm], baseline_files
        )
        print_truth_table(
            title=f"{llm}: fullytyped vs partial",
            a_label="fullytyped",
            b_label="partial",
            counts=counts,
        )


if __name__ == "__main__":
    main()
