import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

UNTYPED_MYPY = os.path.join(
    PARENT_DIR, "mypy_results", "mypy_outputs", "mypy_results_untyped_with_errors.json"
)
COMPLEXITY_JSON = os.path.join(
    PARENT_DIR,
    "Human_annotations_detailed_analysis",
    "original_files_complexity_analysis.json",
)
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "selected_500_files.json")

SAMPLE_SIZE = 500


def main():
    with open(UNTYPED_MYPY, "r", encoding="utf-8") as f:
        untyped = json.load(f)
    with open(COMPLEXITY_JSON, "r", encoding="utf-8") as f:
        complexity = json.load(f)

    # Stage 1: keep only files whose untyped version passes mypy
    evaluable = {
        fname: data
        for fname, data in untyped.items()
        if data.get("isCompiled", False) and data.get("error_count", 1) == 0
    }
    print(f"Total files in untyped JSON : {len(untyped)}")
    print(f"Evaluable (untyped passes)  : {len(evaluable)}")

    # Stage 2: gather metrics for ranking
    file_metrics = []
    missing_complexity = 0
    for fname, data in evaluable.items():
        total_params = data["stats"]["total_parameters"]
        comp = complexity.get(fname)
        if comp is None:
            missing_complexity += 1
            total_lines = 0
        else:
            total_lines = comp.get("total_line_count", 0)
        file_metrics.append(
            {
                "filename": fname,
                "total_parameters": total_params,
                "total_line_count": total_lines,
            }
        )

    if missing_complexity:
        print(f"Warning: {missing_complexity} files missing from complexity JSON (line_count set to 0)")

    # Rank by total_parameters (desc) and total_line_count (desc) using sum-of-ranks
    file_metrics.sort(key=lambda x: x["total_parameters"], reverse=True)
    for rank, item in enumerate(file_metrics):
        item["param_rank"] = rank

    file_metrics.sort(key=lambda x: x["total_line_count"], reverse=True)
    for rank, item in enumerate(file_metrics):
        item["line_rank"] = rank

    for item in file_metrics:
        item["combined_rank"] = item["param_rank"] + item["line_rank"]

    file_metrics.sort(key=lambda x: x["combined_rank"])

    if len(file_metrics) < SAMPLE_SIZE:
        print(f"WARNING: only {len(file_metrics)} evaluable files, selecting all")
    selected = file_metrics[: SAMPLE_SIZE]

    print(f"\nSelected {len(selected)} files")
    print(f"  total_parameters range : {selected[-1]['total_parameters']} – {selected[0]['total_parameters']}")
    print(f"  total_line_count range : {selected[-1]['total_line_count']} – {selected[0]['total_line_count']}")

    output = {
        "count": len(selected),
        "selection_criteria": (
            "Top 500 evaluable files by combined rank of total_parameters and "
            "total_line_count. Evaluable = untyped version passes mypy "
            "(isCompiled=True, error_count=0)."
        ),
        "files": [item["filename"] for item in selected],
        "detailed": selected,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
