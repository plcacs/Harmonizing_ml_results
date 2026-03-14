"""
Analyze the relationship between call graph complexity and mypy success.

Usage:
    python callgraph_vs_mypy.py
"""

import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

CALLGRAPH_JSON = os.path.join(SCRIPT_DIR, "callgraph_gpt5_3_run.json")
MYPY_RESULTS_JSON = os.path.join(
    PARENT_DIR, "GPCE_mypy_results", "mypy_results_gpt5_3_run_with_errors.json"
)
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "callgraph_vs_mypy_gpt5_3_run.csv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_data(callgraph, mypy_results):
    """Match files by basename and return merged rows."""
    cg_by_name = {}
    for rel_path, metrics in callgraph.items():
        if "error" in metrics:
            continue
        basename = os.path.basename(rel_path)
        cg_by_name[basename] = metrics

    rows = []
    for filename, mypy_info in mypy_results.items():
        if filename not in cg_by_name:
            continue
        m = cg_by_name[filename]
        rows.append({
            "file": filename,
            "mypy_pass": mypy_info.get("isCompiled", False),
            "error_count": mypy_info.get("error_count", 0),
            **m,
        })
    return rows


def print_summary(rows):
    """Print comparison of call graph metrics: mypy pass vs fail."""
    pass_rows = [r for r in rows if r["mypy_pass"]]
    fail_rows = [r for r in rows if not r["mypy_pass"]]

    print(f"\nTotal matched files : {len(rows)}")
    print(f"  mypy pass         : {len(pass_rows)}")
    print(f"  mypy fail         : {len(fail_rows)}")

    metrics = [
        "num_functions", "num_call_edges", "max_fan_out", "avg_fan_out",
        "max_fan_in", "avg_fan_in", "max_call_depth", "has_recursion",
        "num_connected_components", "num_classes",
    ]

    def avg(lst, key):
        vals = [r[key] for r in lst]
        if not vals:
            return 0
        if isinstance(vals[0], bool):
            return round(sum(vals) / len(vals) * 100, 1)
        return round(sum(vals) / len(vals), 2)

    print(f"\n{'Metric':<30} {'Pass (avg)':>12} {'Fail (avg)':>12} {'Diff':>10}")
    print("-" * 66)
    for m in metrics:
        p = avg(pass_rows, m)
        f = avg(fail_rows, m)
        if isinstance(p, float) and isinstance(f, float):
            diff = round(f - p, 2)
        else:
            diff = f - p
        unit = "%" if m == "has_recursion" else ""
        print(f"{m:<30} {p:>11}{unit} {f:>11}{unit} {diff:>+10}")


def print_bucket_analysis(rows):
    """Show mypy pass rate bucketed by key metrics."""
    print("\n\n=== Mypy Pass Rate by Metric Buckets ===\n")

    buckets_config = {
        "num_functions": [0, 1, 5, 10, 20, 50, 999],
        "num_call_edges": [0, 1, 5, 10, 20, 50, 999],
        "max_call_depth": [0, 1, 2, 3, 5, 999],
        "max_fan_out": [0, 1, 3, 5, 10, 999],
    }

    for metric, boundaries in buckets_config.items():
        print(f"\n--- {metric} ---")
        print(f"  {'Range':<20} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Pass%':>8}")

        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            label = f"[{lo}, {hi})" if hi != 999 else f"[{lo}+)"
            bucket = [r for r in rows if lo <= r[metric] < hi]
            if not bucket:
                continue
            total = len(bucket)
            passed = sum(1 for r in bucket if r["mypy_pass"])
            failed = total - passed
            rate = round(passed / total * 100, 1)
            print(f"  {label:<20} {total:>6} {passed:>6} {failed:>6} {rate:>7}%")


def save_csv(rows, output_path):
    """Save merged data as CSV for further analysis."""
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, bool):
                    vals.append("1" if v else "0")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"\nCSV saved to {output_path}")


def main():
    callgraph = load_json(CALLGRAPH_JSON)
    mypy_results = load_json(MYPY_RESULTS_JSON)

    rows = merge_data(callgraph, mypy_results)
    if not rows:
        print("No matching files found between callgraph and mypy results.")
        sys.exit(1)

    print_summary(rows)
    print_bucket_analysis(rows)
    save_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
