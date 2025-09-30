import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_error_text(msg: str) -> str:
    # Strip line/column/file specifics in a simple way to group root messages
    # Keep only the error class/message part after ':' if present
    # Also collapse multiple spaces
    core = msg.strip()
    # Remove common prefixes like "error: " or file:line:col:
    # We keep a conservative approach to avoid over-normalizing
    if ": " in core:
        parts = core.split(": ", 1)
        # If it looks like a filename prefix, drop it
        if parts[0].endswith(".py") or parts[0].isdigit():
            core = parts[1]
    core = " ".join(core.split())
    return core


def classify_top_error(errors: List[Any]) -> Optional[str]:
    if not errors:
        return None
    texts: List[str] = []
    for e in errors:
        # mypy JSON may have entries as dicts or raw strings
        if isinstance(e, dict):
            text = e.get("message") or e.get("text") or str(e)
        elif isinstance(e, str):
            text = e
        else:
            text = str(e)
        texts.append(normalize_error_text(text))
    counts = Counter(texts)
    top, _ = counts.most_common(1)[0]
    return top


def status_and_top_error(mypy_result: Dict[str, Any], filename: str) -> Tuple[str, Optional[str]]:
    # Expect a structure with per-file errors. Accept flexible schemas.
    # Try a few common layouts: {"errors": [{"path": ..., "message": ...}, ...]}
    errors_for_file: List[Dict[str, Any]] = []

    # Fast path: pre-indexed by file
    if isinstance(mypy_result, dict) and filename in mypy_result:
        entry = mypy_result.get(filename)
        if isinstance(entry, dict) and "errors" in entry:
            errors_for_file = entry["errors"] or []
        elif isinstance(entry, list):
            errors_for_file = entry

    # Generic scan path
    if not errors_for_file:
        items = []
        if isinstance(mypy_result, dict):
            if "errors" in mypy_result and isinstance(mypy_result["errors"], list):
                items = mypy_result["errors"]
            else:
                # try flatten dict values
                for v in mypy_result.values():
                    if isinstance(v, list):
                        items.extend(v)
        elif isinstance(mypy_result, list):
            items = mypy_result

        for e in items:
            path = e.get("path") or e.get("file") or e.get("filename")
            if not path:
                continue
            if os.path.basename(path) == filename:
                errors_for_file.append(e)

    status = "success" if not errors_for_file else "fail"
    top = classify_top_error(errors_for_file) if errors_for_file else None
    return status, top


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mypy flip results across two runs per model.")
    parser.add_argument("--file", dest="single_file", default=None, help="Optional file name to filter, e.g., schemas_1abb49.py")
    parser.add_argument("--out", dest="out_csv", default="flip_results.csv", help="Output CSV path")
    args = parser.parse_args()

    # Configure models and their two mypy result JSONs (relative paths)
    models: List[Tuple[str, str, str]] = [
        ("gpt4o", 
         os.path.join("..", "mypy_outputs", "mypy_results_gpt4o_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_gpt4o_2nd_run_with_errors.json")),
        ("o1-mini", 
         os.path.join("..", "mypy_outputs", "mypy_results_o1_mini_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_o1_mini_2nd_run_with_errors.json")),
        ("deepseek", 
         os.path.join("..", "mypy_outputs", "mypy_results_deepseek_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_deepseek_2nd_run_with_errors.json")),
        ("claude3_sonnet", 
         os.path.join("..", "mypy_outputs", "mypy_results_claude3_sonnet_1st_run_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_claude3_sonnet_2nd_run_2nd_run_with_errors.json")),
        ("gpt35", 
         os.path.join("..", "mypy_outputs", "mypy_results_gpt35_1st_run_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_gpt35_2nd_run_with_errors.json")),
        ("o3_mini", 
         os.path.join("..", "mypy_outputs", "mypy_results_o3_mini_1st_run_with_errors.json"),
         os.path.join("..", "mypy_outputs", "mypy_results_o3_mini_2nd_run_with_errors.json")),
    ]

    # Load inconsistent files index to focus only on files that differ across runs per model
    index_path = os.path.join(os.path.dirname(__file__), "inconsistent_files_by_model.json")
    inconsistent = read_json(index_path)

    rows: List[Dict[str, Any]] = []
    console_summary: List[str] = []

    for model, run1_path_rel, run2_path_rel in models:
        run1_path = os.path.normpath(os.path.join(os.path.dirname(__file__), run1_path_rel))
        run2_path = os.path.normpath(os.path.join(os.path.dirname(__file__), run2_path_rel))

        if not (os.path.exists(run1_path) and os.path.exists(run2_path)):
            continue

        try:
            run1 = read_json(run1_path)
            run2 = read_json(run2_path)
        except Exception:
            continue

        model_incons = inconsistent.get(model, {}) if isinstance(inconsistent, dict) else {}
        candidate_files: List[str] = []
        for key in ("first_success_only_files", "second_success_only_files"):
            files = model_incons.get(key, []) if isinstance(model_incons, dict) else []
            for f in files:
                if args.single_file and f != args.single_file:
                    continue
                candidate_files.append(f)

        # Deduplicate
        candidate_files = sorted(set(candidate_files))

        for filename in candidate_files:
            run1_status, run1_top = status_and_top_error(run1, filename)
            run2_status, run2_top = status_and_top_error(run2, filename)

            if run1_status == run2_status:
                # Not a flip according to detected status; skip
                continue

            flip_direction = f"{run1_status}->" + ("success" if run2_status == "success" else "fail")

            rows.append({
                "model": model,
                "file": filename,
                "flip_direction": flip_direction,
                "run1_status": run1_status,
                "run2_status": run2_status,
                "run1_error_top": run1_top or "",
                "run2_error_top": run2_top or "",
            })

            if args.single_file and filename == args.single_file:
                msg = f"{model}: {flip_direction} | run1_top='{run1_top or ''}' | run2_top='{run2_top or ''}'"
                console_summary.append(msg)

    # Write single concise CSV
    out_csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.out_csv))
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model",
            "file",
            "flip_direction",
            "run1_status",
            "run2_status",
            "run1_error_top",
            "run2_error_top",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if args.single_file:
        for line in console_summary:
            print(line)
        if not console_summary:
            print("No flips found for the requested file.")
    else:
        print(f"Wrote {len(rows)} flip rows to {out_csv_path}")


if __name__ == "__main__":
    main()

