"""
Per-error analysis: for each mypy error, what strategy did the LLM use to fix it?

Parses mypy errors (with line numbers and error codes), then compares the
error line in the initial LLM version vs the fixed version to classify
the fix strategy.
"""

import json
import os
import re
import glob
import difflib
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MYPY_DIR = os.path.join(PARENT_DIR, "mypy_results", "mypy_outputs")

MODELS = {
    "GPT-5": {
        "log": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "gpt5_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "gpt5_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_gpt5_1st_run_with_errors.json"),
    },
    "DeepSeek": {
        "log": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "deep_seek_2nd_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "deepseek_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_deepseek_2nd_run_with_errors.json"),
    },
    "Claude": {
        "log": os.path.join(PARENT_DIR, "claude_mypy_fix", "fix_log.json"),
        "initial_dir": os.path.join(PARENT_DIR, "claude3_sonnet_1st_run"),
        "fixed_dir": os.path.join(PARENT_DIR, "claude_mypy_fix", "fixed_files"),
        "mypy_json": os.path.join(MYPY_DIR, "mypy_results_claude3_sonnet_1st_run_with_errors.json"),
    },
}

ERROR_PATTERN = re.compile(r"^.+?:(\d+): error: .+\[(.+)\]$")


def find_file(directory, filename):
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def parse_errors(error_list):
    """Extract (line_number, error_code) from mypy error strings."""
    errors = []
    for err in error_list:
        m = ERROR_PATTERN.match(err)
        if m:
            errors.append((int(m.group(1)), m.group(2)))
    return errors


def classify_fix(init_line, fixed_line):
    """Classify how a single error line was fixed."""
    if init_line is None or fixed_line is None:
        return "restructured"

    init_stripped = init_line.strip()
    fixed_stripped = fixed_line.strip()

    if init_stripped == fixed_stripped:
        return "unchanged"

    if "# type: ignore" in fixed_stripped and "# type: ignore" not in init_stripped:
        return "type_ignore_added"

    if "cast(" in fixed_stripped and "cast(" not in init_stripped:
        return "cast_added"

    if ": Any" in fixed_stripped or "-> Any" in fixed_stripped:
        if (": Any" not in init_stripped and "-> Any" not in init_stripped):
            return "changed_to_any"

    ann_removed = False
    if re.search(r":\s*\w", init_stripped) and not re.search(r":\s*\w", fixed_stripped):
        ann_removed = True
    if "->" in init_stripped and "->" not in fixed_stripped:
        ann_removed = True
    if ann_removed:
        return "annotation_removed"

    if init_stripped != fixed_stripped:
        has_ann_change = False
        if ":" in init_stripped and ":" in fixed_stripped:
            has_ann_change = True
        if "->" in init_stripped or "->" in fixed_stripped:
            has_ann_change = True
        if has_ann_change:
            return "type_corrected"
        return "code_modified"

    return "other"


def find_best_match_line(init_lines, fixed_lines, target_line_num):
    """Try to find the matching line in fixed version, handling line shifts."""
    if target_line_num <= len(fixed_lines):
        return fixed_lines[target_line_num - 1]

    if target_line_num <= len(init_lines):
        init_line = init_lines[target_line_num - 1].strip()
        for fixed_line in fixed_lines:
            if fixed_line.strip() == init_line:
                return fixed_line
    return None


def sub_classify_code_modification(init_line, fixed_line):
    """Sub-classify what kind of code change was made for 'code_modified' errors."""
    init_s, fixed_s = init_line.strip(), fixed_line.strip()
    if ("is not None" in fixed_s and "is not None" not in init_s) or \
       ("is None" in fixed_s and "is None" not in init_s):
        return "added_none_check"
    if "isinstance(" in fixed_s and "isinstance(" not in init_s:
        return "added_isinstance"
    if ".get(" in fixed_s and ".get(" not in init_s:
        return "used_dict_get"
    if "if " in fixed_s and "if " not in init_s:
        return "added_conditional"
    if "assert " in fixed_s and "assert " not in init_s:
        return "added_assertion"
    if ("try" in fixed_s or "except" in fixed_s) and \
       ("try" not in init_s and "except" not in init_s):
        return "added_try_except"
    if " or " in fixed_s and " or " not in init_s:
        return "added_fallback"
    if "(" in init_s and "(" in fixed_s:
        return "modified_call_or_args"
    if "=" in init_s and "=" in fixed_s:
        return "modified_assignment"
    return "other_code_change"


def analyze_annotation_removal(init_line, fixed_line):
    """Count how many type annotations were removed from the line."""
    init_s, fixed_s = init_line.strip(), fixed_line.strip()
    removed = 0

    if "->" in init_s and "->" not in fixed_s:
        removed += 1

    if init_s.startswith("def "):
        init_match = re.search(r"\((.*)\)", init_s)
        fixed_match = re.search(r"\((.*)\)", fixed_s)
        if init_match and fixed_match:
            init_params = [p.strip() for p in init_match.group(1).split(",") if p.strip()]
            fixed_params = [p.strip() for p in fixed_match.group(1).split(",") if p.strip()]
            init_ann = sum(1 for p in init_params
                          if ":" in p and not p.startswith("self"))
            fixed_ann = sum(1 for p in fixed_params
                           if ":" in p and not p.startswith("self"))
            removed += max(0, init_ann - fixed_ann)
    else:
        if re.search(r":\s*\w", init_s) and not re.search(r":\s*\w", fixed_s):
            removed += 1

    return removed


def get_file_diff_summary(init_lines, fixed_lines):
    """Categorize line-level changes between two file versions."""
    diff = list(difflib.unified_diff(init_lines, fixed_lines, n=0))
    added = [l[1:].strip() for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in diff if l.startswith("-") and not l.startswith("---")]

    TYPE_KEYWORDS = {"int", "str", "float", "bool", "List", "Dict", "Optional",
                     "Any", "Union", "Tuple", "Set", "Callable", "Sequence", "Type"}
    categories = Counter()
    for line in added + removed:
        if line.startswith("import ") or line.startswith("from "):
            categories["import_change"] += 1
        elif line.startswith("def ") or line.startswith("class "):
            categories["def_or_class_change"] += 1
        elif any(kw in line for kw in TYPE_KEYWORDS) and (":" in line or "->" in line):
            categories["annotation_change_elsewhere"] += 1
        elif line.startswith("#"):
            categories["comment_change"] += 1
        elif "=" in line:
            categories["assignment_change"] += 1
        else:
            categories["other_change"] += 1

    return {"total_added": len(added), "total_removed": len(removed),
            "categories": categories}


def analyze_model(model_name, config):
    for path_key in ["log", "mypy_json"]:
        if not os.path.exists(config[path_key]):
            print(f"{model_name}: {path_key} not found at {config[path_key]}")
            return

    with open(config["log"], "r") as f:
        log = json.load(f)

    with open(config["mypy_json"], "r") as f:
        mypy_data = json.load(f)

    fixed_files = {k: v for k, v in log.items() if v["status"] == "fixed"}

    strategy_counter = Counter()
    strategy_by_error_code = {}
    total_errors = 0
    files_analyzed = 0

    code_mod_subcats = Counter()
    code_mod_examples = []
    unchanged_diff_cats = Counter()
    unchanged_count = 0
    unchanged_examples = []
    ann_removed_dist = Counter()

    for filename in fixed_files:
        if filename not in mypy_data:
            continue
        errors = parse_errors(mypy_data[filename].get("errors", []))
        if not errors:
            continue

        initial_path = find_file(config["initial_dir"], filename)
        fixed_path = os.path.join(config["fixed_dir"], filename)

        if not initial_path or not os.path.exists(fixed_path):
            continue

        init_lines = read_lines(initial_path)
        fixed_lines = read_lines(fixed_path)
        files_analyzed += 1
        file_diff = None

        for line_num, error_code in errors:
            total_errors += 1
            init_line = init_lines[line_num - 1] if line_num <= len(init_lines) else None
            fixed_line = find_best_match_line(init_lines, fixed_lines, line_num)

            strategy = classify_fix(init_line, fixed_line)
            strategy_counter[strategy] += 1

            if error_code not in strategy_by_error_code:
                strategy_by_error_code[error_code] = Counter()
            strategy_by_error_code[error_code][strategy] += 1

            if strategy == "code_modified" and init_line and fixed_line:
                sub = sub_classify_code_modification(init_line, fixed_line)
                code_mod_subcats[sub] += 1
                if len(code_mod_examples) < 15:
                    code_mod_examples.append(
                        (init_line.strip(), fixed_line.strip(), error_code))

            elif strategy == "unchanged":
                unchanged_count += 1
                if file_diff is None:
                    file_diff = get_file_diff_summary(init_lines, fixed_lines)
                    for cat, cnt in file_diff["categories"].items():
                        unchanged_diff_cats[cat] += cnt
                if len(unchanged_examples) < 5:
                    unchanged_examples.append({
                        "line": init_line.strip() if init_line else "N/A",
                        "error_code": error_code,
                        "added": file_diff["total_added"],
                        "removed": file_diff["total_removed"],
                    })

            elif strategy == "annotation_removed" and init_line and fixed_line:
                n = analyze_annotation_removal(init_line, fixed_line)
                ann_removed_dist[n] += 1

    print(f"\n{'='*60}")
    print(f"  {model_name} — Error Fix Strategy Analysis")
    print(f"  Files analyzed: {files_analyzed}, Total errors: {total_errors}")
    print(f"{'='*60}")

    print(f"\n  --- Overall fix strategies ---")
    for strategy, count in strategy_counter.most_common():
        print(f"    {strategy}: {count}/{total_errors} ({100*count/total_errors:.1f}%)")

    print(f"\n  --- Top 10 error codes ---")
    sorted_codes = sorted(strategy_by_error_code.items(), key=lambda x: -sum(x[1].values()))
    for error_code, strategies in sorted_codes[:10]:
        code_total = sum(strategies.values())
        print(f"\n    [{error_code}] ({code_total} errors)")
        for strategy, count in strategies.most_common():
            print(f"      {strategy}: {count} ({100*count/code_total:.1f}%)")

    # --- Q1: code_modified sub-categories ---
    if code_mod_subcats:
        print(f"\n  --- code_modified: sub-categories ---")
        for sub, count in code_mod_subcats.most_common():
            print(f"    {sub}: {count}")
        if code_mod_examples:
            print(f"\n  --- code_modified: examples (up to 15) ---")
            for init_l, fixed_l, ec in code_mod_examples:
                print(f"    [{ec}]")
                print(f"      BEFORE: {init_l}")
                print(f"      AFTER:  {fixed_l}")

    # --- Q2: unchanged — what else changed in the file ---
    if unchanged_count > 0:
        print(f"\n  --- unchanged lines: what else changed ({unchanged_count} errors) ---")
        print(f"    Changes in files (by category, counted once per file):")
        for cat, count in unchanged_diff_cats.most_common():
            print(f"      {cat}: {count}")
        if unchanged_examples:
            print(f"\n    Example unchanged error lines:")
            for ex in unchanged_examples:
                print(f"      [{ex['error_code']}] {ex['line']}")
                print(f"        File diff: +{ex['added']} / -{ex['removed']} lines")

    # --- Q3: annotation_removed — how many per error line ---
    if ann_removed_dist:
        total_ann_rm = sum(ann_removed_dist.values())
        print(f"\n  --- annotation_removed: count per error line ---")
        for n_removed, freq in sorted(ann_removed_dist.items()):
            print(f"    {n_removed} annotation(s) removed: {freq} ({100*freq/total_ann_rm:.1f}%)")


if __name__ == "__main__":
    for model_name, config in MODELS.items():
        analyze_model(model_name, config)
