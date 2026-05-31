"""
Joined analysis: annotation quality (AST) vs mypy outcomes.

Questions:
  1. Does more annotation lead to more type errors?
  2. Does more Any lead to more mypy success?
  3. Relation between annotation quality and mypy success?

Joins per-file AST annotation stats with mypy result JSONs
on filename stem, filtered to the 500 selected files.
"""

import ast
import json
import re
import statistics
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
MYPY_DIR = BENCH_DIR / "HarmonizingML_mypy_results"
SELECTED_500 = BENCH_DIR / "gpt5_500_sample_runs" / "selected_500_files.json"

NON_TYPE_RELATED_ERRORS = {
    "name-defined", "import", "syntax", "no-redef", "unused-ignore",
    "override-without-super", "redundant-cast", "literal-required",
    "typeddict-unknown-key", "typeddict-item", "truthy-function",
    "str-bytes-safe", "unused-coroutine", "explicit-override",
    "truthy-iterable", "redundant-self", "redundant-await", "unreachable",
}

SETTINGS = [
    {
        "label": "GPT5 setting1 (inline)",
        "src_dir": BENCH_DIR / "gpt5_2_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_gpt5_2_run_with_errors.json",
    },
    {
        "label": "GPT5 setting2 (stub)",
        "src_dir": BENCH_DIR / "gpt5_1_infer_stub_run" / "merged",
        "recursive": False,
        "mypy_json": MYPY_DIR / "mypy_results_gpt5_1_infer_stub_run_with_errors.json",
    },
    {
        "label": "DeepSeek setting1 (inline)",
        "src_dir": BENCH_DIR / "deepseek_3_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_deepseek_3_run_with_errors.json",
    },
    {
        "label": "DeepSeek setting2 (stub)",
        "src_dir": BENCH_DIR / "deepseek_3_stub_run" / "merged",
        "recursive": False,
        "mypy_json": MYPY_DIR / "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_selected_stems():
    with open(SELECTED_500) as f:
        data = json.load(f)
    return {Path(name).stem for name in data["files"]}


def _contains_any(node):
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and _contains_any(child):
            return True
    return False


def _is_exact_any(node):
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    return False


def classify(node):
    if node is None:
        return "blank"
    if _is_exact_any(node):
        return "exact_any"
    if _contains_any(node):
        return "partial_any"
    return "concrete"


def analyze_file_ast(filepath):
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return None

    total_params = 0
    blank = 0
    exact_any = 0
    partial_any = 0
    concrete = 0
    total_returns = 0
    ret_blank = 0
    ret_exact_any = 0
    ret_partial_any = 0
    ret_concrete = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_args = (
            list(getattr(node.args, "posonlyargs", []))
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        extras = [a for a in [node.args.vararg, node.args.kwarg] if a]
        for arg in all_args + extras:
            if arg.arg in ("self", "cls"):
                continue
            total_params += 1
            cat = classify(arg.annotation)
            if cat == "blank":
                blank += 1
            elif cat == "exact_any":
                exact_any += 1
            elif cat == "partial_any":
                partial_any += 1
            else:
                concrete += 1

        total_returns += 1
        rcat = classify(node.returns)
        if rcat == "blank":
            ret_blank += 1
        elif rcat == "exact_any":
            ret_exact_any += 1
        elif rcat == "partial_any":
            ret_partial_any += 1
        else:
            ret_concrete += 1

    total_slots = total_params + total_returns

    # Type coverage: concrete + partial_any (static or partially-static types)
    typed_count = (concrete + ret_concrete) + (partial_any + ret_partial_any)
    # Any coverage: blank + exact_any (unannotated or pure "Any")
    any_or_blank = (blank + ret_blank) + (exact_any + ret_exact_any)

    return {
        "total_slots": total_slots,
        "annotated": total_slots - blank - ret_blank,
        "coverage_pct": (typed_count / total_slots * 100) if total_slots else 0,
        "any_count": any_or_blank,
        "any_pct": (any_or_blank / total_slots * 100) if total_slots else 0,
        "concrete": concrete + ret_concrete,
        "concrete_pct": ((concrete + ret_concrete) / total_slots * 100) if total_slots else 0,
    }


def is_unprocessed(errors):
    if any(kw in e.lower() for e in errors for kw in ["syntax", "empty_body", "name_defined"]):
        return True
    for e in errors:
        m = re.search(r"\[([a-z\-]+)\]", e)
        if m and m.group(1) in NON_TYPE_RELATED_ERRORS:
            return True
    return False


def collect_files(directory, recursive, allowed):
    result = {}
    pattern = directory.rglob("*.py") if recursive else directory.glob("*.py")
    for p in sorted(pattern):
        if p.stem in allowed and p.stem not in result:
            result[p.stem] = p
    return result


# ── Per-file joined data ─────────────────────────────────────────────────────

def build_joined(setting, allowed):
    src_files = collect_files(setting["src_dir"], setting["recursive"], allowed)

    with open(setting["mypy_json"]) as f:
        mypy_data = json.load(f)

    rows = []
    for stem in sorted(allowed):
        mypy_key = stem + ".py"
        mypy_entry = mypy_data.get(mypy_key)
        src_path = src_files.get(stem)

        ast_stats = analyze_file_ast(src_path) if src_path else None

        if mypy_entry is None or ast_stats is None:
            continue

        errors = mypy_entry.get("errors", [])
        if is_unprocessed(errors):
            continue

        rows.append({
            "file": stem,
            "coverage_pct": ast_stats["coverage_pct"],
            "any_pct": ast_stats["any_pct"],
            "concrete_pct": ast_stats["concrete_pct"],
            "total_slots": ast_stats["total_slots"],
            "annotated": ast_stats["annotated"],
            "any_count": ast_stats["any_count"],
            "error_count": mypy_entry["error_count"],
            "is_clean": mypy_entry["error_count"] == 0,
        })

    return rows


# ── Analysis ─────────────────────────────────────────────────────────────────

def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "N/A"


def median_safe(vals):
    return f"{statistics.median(vals):.1f}" if vals else "N/A"


def mean_safe(vals):
    return f"{statistics.mean(vals):.1f}" if vals else "N/A"


def analyze_setting(label, rows):
    clean = [r for r in rows if r["is_clean"]]
    fail = [r for r in rows if not r["is_clean"]]

    print(f"\n{'=' * 65}")
    print(f"  {label}  ({len(rows)} evaluable files)")
    print(f"{'=' * 65}")
    print(f"  Mypy clean: {len(clean)}   Mypy fail: {len(fail)}")
    print()

    print(f"  Q1: Does more annotation -> more errors?")
    print(f"  {'':30s} {'Clean files':>14} {'Failing files':>14}")
    print(f"  {'Median coverage %':<30s} {median_safe([r['coverage_pct'] for r in clean]):>14} {median_safe([r['coverage_pct'] for r in fail]):>14}")
    print(f"  {'Mean coverage %':<30s} {mean_safe([r['coverage_pct'] for r in clean]):>14} {mean_safe([r['coverage_pct'] for r in fail]):>14}")
    print(f"  {'Median annotated slots':<30s} {median_safe([r['annotated'] for r in clean]):>14} {median_safe([r['annotated'] for r in fail]):>14}")
    print()

    print(f"  Q2: Does more Any -> more success?")
    print(f"  {'Median Any %':<30s} {median_safe([r['any_pct'] for r in clean]):>14} {median_safe([r['any_pct'] for r in fail]):>14}")
    print(f"  {'Mean Any %':<30s} {mean_safe([r['any_pct'] for r in clean]):>14} {mean_safe([r['any_pct'] for r in fail]):>14}")
    print(f"  {'Median Any count':<30s} {median_safe([r['any_count'] for r in clean]):>14} {median_safe([r['any_count'] for r in fail]):>14}")
    print()

    print(f"  Q3: Concrete % vs mypy outcome")
    print(f"  {'Median concrete %':<30s} {median_safe([r['concrete_pct'] for r in clean]):>14} {median_safe([r['concrete_pct'] for r in fail]):>14}")
    print(f"  {'Mean concrete %':<30s} {mean_safe([r['concrete_pct'] for r in clean]):>14} {mean_safe([r['concrete_pct'] for r in fail]):>14}")
    print()

    # Bucket by coverage ranges
    print(f"  Mypy clean rate by coverage bucket:")
    buckets = [(0, 50), (50, 80), (80, 95), (95, 100.01)]
    for lo, hi in buckets:
        in_bucket = [r for r in rows if lo <= r["coverage_pct"] < hi]
        clean_in = sum(1 for r in in_bucket if r["is_clean"])
        label_b = f"    [{lo:.0f}%-{hi:.0f}%)"
        print(f"  {label_b:<22s} {clean_in}/{len(in_bucket)}  ({pct(clean_in, len(in_bucket))})")

    # Bucket by Any %
    print(f"\n  Mypy clean rate by Any% bucket:")
    any_buckets = [(0, 0.01), (0.01, 10), (10, 20), (20, 100.01)]
    bucket_labels = ["0% (no Any)", "0-10%", "10-20%", "20%+"]
    for (lo, hi), blabel in zip(any_buckets, bucket_labels):
        in_bucket = [r for r in rows if lo <= r["any_pct"] < hi]
        clean_in = sum(1 for r in in_bucket if r["is_clean"])
        print(f"    {blabel:<22s} {clean_in}/{len(in_bucket)}  ({pct(clean_in, len(in_bucket))})")


# ── Paired comparison ────────────────────────────────────────────────────────

def paired_comparison(label, rows1, rows2):
    map1 = {r["file"]: r for r in rows1}
    map2 = {r["file"]: r for r in rows2}
    common = sorted(set(map1.keys()) & set(map2.keys()))

    if not common:
        print(f"\n  No common evaluable files for {label}")
        return

    more_ann_more_err = 0
    more_ann_less_err = 0
    same = 0

    for f in common:
        r1, r2 = map1[f], map2[f]
        cov_delta = r2["coverage_pct"] - r1["coverage_pct"]
        err_delta = r2["error_count"] - r1["error_count"]
        if abs(cov_delta) < 0.1:
            same += 1
        elif (cov_delta > 0 and err_delta > 0) or (cov_delta < 0 and err_delta < 0):
            more_ann_more_err += 1
        else:
            more_ann_less_err += 1

    print(f"\n{'=' * 65}")
    print(f"  Paired: {label}  ({len(common)} common evaluable files)")
    print(f"{'=' * 65}")
    print(f"  More coverage -> more errors:  {more_ann_more_err}")
    print(f"  More coverage -> fewer errors: {more_ann_less_err}")
    print(f"  Same coverage:                {same}")

    # Files where one is clean and other fails
    s1_clean_s2_fail = [f for f in common if map1[f]["is_clean"] and not map2[f]["is_clean"]]
    s2_clean_s1_fail = [f for f in common if map2[f]["is_clean"] and not map1[f]["is_clean"]]

    if s1_clean_s2_fail:
        cov_deltas = [map2[f]["coverage_pct"] - map1[f]["coverage_pct"] for f in s1_clean_s2_fail]
        any_deltas = [map2[f]["any_pct"] - map1[f]["any_pct"] for f in s1_clean_s2_fail]
        print(f"\n  Clean in setting1, fail in setting2 ({len(s1_clean_s2_fail)} files):")
        print(f"    Mean coverage delta (s2-s1): {mean_safe(cov_deltas)}pp")
        print(f"    Mean Any% delta (s2-s1):     {mean_safe(any_deltas)}pp")

    if s2_clean_s1_fail:
        cov_deltas = [map2[f]["coverage_pct"] - map1[f]["coverage_pct"] for f in s2_clean_s1_fail]
        any_deltas = [map2[f]["any_pct"] - map1[f]["any_pct"] for f in s2_clean_s1_fail]
        print(f"\n  Clean in setting2, fail in setting1 ({len(s2_clean_s1_fail)} files):")
        print(f"    Mean coverage delta (s2-s1): {mean_safe(cov_deltas)}pp")
        print(f"    Mean Any% delta (s2-s1):     {mean_safe(any_deltas)}pp")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    allowed = load_selected_stems()
    print(f"Filtering to {len(allowed)} selected files.\n")

    all_rows = {}
    for s in SETTINGS:
        rows = build_joined(s, allowed)
        all_rows[s["label"]] = rows
        analyze_setting(s["label"], rows)

    pairs = [
        ("GPT5 setting1 (inline)", "GPT5 setting2 (stub)"),
        ("DeepSeek setting1 (inline)", "DeepSeek setting2 (stub)"),
    ]
    for l1, l2 in pairs:
        paired_comparison(f"{l1} vs {l2}", all_rows[l1], all_rows[l2])


if __name__ == "__main__":
    main()
