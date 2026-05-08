"""Compare mypy results before vs after union import, per model."""

import json
import re
import os

BASE = os.path.join(os.path.dirname(__file__), "mypy_outputs")

PAIRS = {
    "GPT5": (
        "mypy_results_gpt5_1_infer_stub_run_with_errors.json",
        "mypy_results_gpt5_1_infer_stub_union_import_run_with_errors.json",
    ),
    "DeepSeek": (
        "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
        "mypy_results_deepseek_3_stub_union_import_run_with_errors.json",
    ),
}


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def extract_error_codes(errors):
    codes = []
    for e in errors:
        m = re.search(r"\[([a-z\-]+)\]", e)
        if m:
            codes.append(m.group(1))
    return codes


def analyze_pair(label, before_file, after_file):
    before = load(before_file)
    after = load(after_file)
    common = sorted(set(before) & set(after))

    total_before = total_after = 0
    compiled_before = compiled_after = 0
    improved = worsened = same = 0
    codes_before = {}
    codes_after = {}
    worsened_files = []

    for f in common:
        b, a = before[f], after[f]
        eb, ea = b["error_count"], a["error_count"]
        total_before += eb
        total_after += ea
        compiled_before += b["isCompiled"]
        compiled_after += a["isCompiled"]

        if ea < eb:
            improved += 1
        elif ea > eb:
            worsened += 1
            worsened_files.append((f, eb, ea))
        else:
            same += 1

        for c in extract_error_codes(b["errors"]):
            codes_before[c] = codes_before.get(c, 0) + 1
        for c in extract_error_codes(a["errors"]):
            codes_after[c] = codes_after.get(c, 0) + 1

    n = len(common)
    print(f"\n{'='*60}")
    print(f"  {label}  ({n} files)")
    print(f"{'='*60}")

    print(f"\n1. Total errors:  {total_before} -> {total_after}  (delta: {total_after - total_before:+d})")
    print(f"2. Compiled:      {compiled_before}/{n} ({100*compiled_before/n:.1f}%) -> {compiled_after}/{n} ({100*compiled_after/n:.1f}%)")
    print(f"3. Per-file:      improved={improved}, same={same}, worsened={worsened}")

    all_codes = sorted(set(codes_before) | set(codes_after), key=lambda c: codes_before.get(c, 0), reverse=True)
    print(f"\n4. Error codes (before -> after):")
    for c in all_codes:
        cb, ca = codes_before.get(c, 0), codes_after.get(c, 0)
        delta = ca - cb
        if delta != 0:
            print(f"   [{c:20s}]  {cb:4d} -> {ca:4d}  ({delta:+d})")

    import_codes = ["import", "name-defined"]
    print(f"\n5. Import-related errors:")
    for c in import_codes:
        cb, ca = codes_before.get(c, 0), codes_after.get(c, 0)
        print(f"   [{c:20s}]  {cb:4d} -> {ca:4d}  ({ca - cb:+d})")

    if worsened_files:
        print(f"\n6. Worsened files (top 10):")
        for f, eb, ea in sorted(worsened_files, key=lambda x: x[2]-x[1], reverse=True)[:10]:
            print(f"   {f}: {eb} -> {ea} ({ea-eb:+d})")


for label, (bf, af) in PAIRS.items():
    analyze_pair(label, bf, af)
