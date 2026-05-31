import json
import re
from collections import Counter, defaultdict


def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)


def extract_error_code(error_line):
    match = re.search(r"\[([a-z\-]+)\]", error_line)
    return match.group(1) if match else None


NON_TYPE_RELATED_ERRORS = {
    "name-defined", "import", "syntax", "no-redef", "unused-ignore",
    "override-without-super", "redundant-cast", "literal-required",
    "typeddict-unknown-key", "typeddict-item", "truthy-function",
    "str-bytes-safe", "unused-coroutine", "explicit-override",
    "truthy-iterable", "redundant-self", "redundant-await", "unreachable",
}


def is_unprocessed(errors):
    if any(kw in e.lower() for e in errors for kw in ["syntax", "empty_body", "name_defined"]):
        return True
    for e in errors:
        code = extract_error_code(e)
        if code and code in NON_TYPE_RELATED_ERRORS:
            return True
    return False


def get_type_error_codes(errors):
    """Return list of type-related error codes (excludes summary lines)."""
    codes = []
    for e in errors:
        if e.startswith("Found "):
            continue
        code = extract_error_code(e)
        if code and code not in NON_TYPE_RELATED_ERRORS:
            codes.append(code)
    return codes


def error_type_breakdown(name, data):
    """Count error codes across all evaluable files."""
    counter = Counter()
    for file_key, file_data in data.items():
        errors = file_data.get("errors", [])
        if is_unprocessed(errors):
            continue
        counter.update(get_type_error_codes(errors))
    print(f"\n=== Error Type Breakdown: {name} ===")
    print(f"{'Error Code':<30} {'Count':>6}")
    print("-" * 38)
    for code, count in counter.most_common():
        print(f"{code:<30} {count:>6}")
    print(f"{'TOTAL':<30} {sum(counter.values()):>6}")
    return counter


def divergence_analysis(name1, data1, name2, data2):
    """Find files clean in one setting but failing in the other."""
    common_files = set(data1.keys()) & set(data2.keys())

    clean1_fail2 = []
    clean2_fail1 = []
    both_clean = []
    both_fail = []

    for f in sorted(common_files):
        errs1 = data1[f].get("errors", [])
        errs2 = data2[f].get("errors", [])
        if is_unprocessed(errs1) or is_unprocessed(errs2):
            continue

        ec1 = data1[f]["error_count"]
        ec2 = data2[f]["error_count"]

        if ec1 == 0 and ec2 == 0:
            both_clean.append(f)
        elif ec1 == 0 and ec2 > 0:
            clean1_fail2.append(f)
        elif ec1 > 0 and ec2 == 0:
            clean2_fail1.append(f)
        else:
            both_fail.append(f)

    print(f"\n=== Divergence: {name1} vs {name2} ===")
    print(f"Common evaluable files: {len(both_clean) + len(clean1_fail2) + len(clean2_fail1) + len(both_fail)}")
    print(f"Both clean:            {len(both_clean)}")
    print(f"Both fail:             {len(both_fail)}")
    print(f"Clean in {name1} ONLY: {len(clean1_fail2)}")
    print(f"Clean in {name2} ONLY: {len(clean2_fail1)}")

    if clean1_fail2:
        codes = Counter()
        for f in clean1_fail2:
            codes.update(get_type_error_codes(data2[f]["errors"]))
        print(f"\n  Errors in {name2} (for files clean in {name1}):")
        for code, cnt in codes.most_common():
            print(f"    {code:<28} {cnt:>4}")

    if clean2_fail1:
        codes = Counter()
        for f in clean2_fail1:
            codes.update(get_type_error_codes(data1[f]["errors"]))
        print(f"\n  Errors in {name1} (for files clean in {name2}):")
        for code, cnt in codes.most_common():
            print(f"    {code:<28} {cnt:>4}")


if __name__ == "__main__":
    pairs = [
        (
            "gpt5_setting1",
            "mypy_results_gpt5_2_run_with_errors.json",
            "gpt5_setting2",
            "mypy_results_gpt5_1_infer_stub_run_with_errors.json",
        ),
        (
            "deepseek_setting1",
            "mypy_results_deepseek_3_run_with_errors.json",
            "deepseek_setting2",
            "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
        ),
    ]

    for name1, file1, name2, file2 in pairs:
        data1 = load_json_file(file1)
        data2 = load_json_file(file2)

        error_type_breakdown(name1, data1)
        error_type_breakdown(name2, data2)
        divergence_analysis(name1, data1, name2, data2)
        print("\n" + "=" * 60)
