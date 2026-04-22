"""
Compare annotation coverage and quality across settings.

For each file, parses the AST and collects per-function:
  - parameter annotations (excluding self/cls)
  - return type annotations

Classifications: concrete, exact_any, partial_any, blank.

Directories (relative to ManyTypes4py_benchmarks):
  untyped_benchmarks/          — flat, original untyped .py
  gpt5_2_run/                  — numbered subdirs, setting1
  gpt5_1_infer_stub_run/merged — flat, setting2
  deepseek_3_run/              — numbered subdirs, setting1
  deepseek_3_stub_run/merged   — flat, setting2
"""

import ast
import json
import os
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
SELECTED_500 = BENCH_DIR / "gpt5_500_sample_runs" / "selected_500_files.json"


def load_selected_stems() -> set[str]:
    with open(SELECTED_500, "r") as f:
        data = json.load(f)
    return {Path(name).stem for name in data["files"]}


# ── AST helpers ──────────────────────────────────────────────────────────────

def _contains_any(node: ast.expr) -> bool:
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


def _is_exact_any(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    return False


def classify(node) -> str:
    if node is None:
        return "blank"
    if _is_exact_any(node):
        return "exact_any"
    if _contains_any(node):
        return "partial_any"
    return "concrete"


def _is_parameterized(node) -> bool:
    """True if annotation is generic like List[int], dict[str, Any], etc."""
    if node is None:
        return False
    return isinstance(node, ast.Subscript)


def analyze_file(filepath: Path) -> dict | None:
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None

    stats = {
        "params_blank": 0, "params_exact_any": 0,
        "params_partial_any": 0,
        "params_concrete_bare": 0, "params_concrete_param": 0,
        "returns_blank": 0, "returns_exact_any": 0,
        "returns_partial_any": 0,
        "returns_concrete_bare": 0, "returns_concrete_param": 0,
        "total_params": 0, "total_returns": 0,
        "args_blank": 0, "args_annotated": 0,
        "vararg_blank": 0, "vararg_annotated": 0,
        "kwarg_blank": 0, "kwarg_annotated": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        all_args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            stats["total_params"] += 1
            cat = classify(arg.annotation)
            if cat == "concrete":
                sub = "concrete_param" if _is_parameterized(arg.annotation) else "concrete_bare"
                stats[f"params_{sub}"] += 1
            else:
                stats[f"params_{cat}"] += 1
            if cat == "blank":
                stats["args_blank"] += 1
            else:
                stats["args_annotated"] += 1

        if node.args.vararg:
            stats["total_params"] += 1
            cat = classify(node.args.vararg.annotation)
            if cat == "concrete":
                sub = "concrete_param" if _is_parameterized(node.args.vararg.annotation) else "concrete_bare"
                stats[f"params_{sub}"] += 1
            else:
                stats[f"params_{cat}"] += 1
            if cat == "blank":
                stats["vararg_blank"] += 1
            else:
                stats["vararg_annotated"] += 1

        if node.args.kwarg:
            stats["total_params"] += 1
            cat = classify(node.args.kwarg.annotation)
            if cat == "concrete":
                sub = "concrete_param" if _is_parameterized(node.args.kwarg.annotation) else "concrete_bare"
                stats[f"params_{sub}"] += 1
            else:
                stats[f"params_{cat}"] += 1
            if cat == "blank":
                stats["kwarg_blank"] += 1
            else:
                stats["kwarg_annotated"] += 1

        stats["total_returns"] += 1
        rcat = classify(node.returns)
        if rcat == "concrete":
            rsub = "concrete_param" if _is_parameterized(node.returns) else "concrete_bare"
            stats[f"returns_{rsub}"] += 1
        else:
            stats[f"returns_{rcat}"] += 1

    return stats


# ── File collection ──────────────────────────────────────────────────────────

def collect_flat(directory: Path, allowed: set[str]) -> dict[str, Path]:
    """Flat directory: key = filename stem, filtered to allowed set."""
    result = {}
    for p in directory.glob("*.py"):
        if p.stem in allowed:
            result[p.stem] = p
    return result


def collect_recursive(directory: Path, allowed: set[str]) -> dict[str, Path]:
    """Numbered-subdir directory: key = filename stem (deduped, first wins), filtered."""
    result = {}
    for p in sorted(directory.rglob("*.py")):
        stem = p.stem
        if stem in allowed and stem not in result:
            result[stem] = p
    return result


# ── Aggregation & printing ───────────────────────────────────────────────────

def aggregate(per_file: dict[str, dict | None]) -> dict:
    totals = Counter()
    for stats in per_file.values():
        if stats is None:
            continue
        for k, v in stats.items():
            totals[k] += v
    return dict(totals)


def pct(num, den):
    return f"{num / den * 100:.1f}%" if den else "N/A"


def print_summary(label: str, totals: dict, file_count: int, parse_errors: int):
    tp = totals.get("total_params", 0)
    tr = totals.get("total_returns", 0)

    annotated_params = tp - totals.get("params_blank", 0)
    annotated_returns = tr - totals.get("returns_blank", 0)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Files: {file_count}  (parse errors: {parse_errors})")
    print(f"  Total params: {tp}   Total returns: {tr}")
    print()
    print(f"  --- Coverage ---")
    print(f"  Params annotated:  {annotated_params}/{tp} ({pct(annotated_params, tp)})")
    print(f"  Returns annotated: {annotated_returns}/{tr} ({pct(annotated_returns, tr)})")
    print()
    print(f"    Regular args:  annotated={totals.get('args_annotated',0)}  blank={totals.get('args_blank',0)}")
    print(f"    *args:         annotated={totals.get('vararg_annotated',0)}  blank={totals.get('vararg_blank',0)}")
    print(f"    **kwargs:      annotated={totals.get('kwarg_annotated',0)}  blank={totals.get('kwarg_blank',0)}")
    print()
    print(f"  --- Quality (params) ---")
    print(f"    Concrete bare:   {totals.get('params_concrete_bare',0):>5}  ({pct(totals.get('params_concrete_bare',0), tp)})")
    print(f"    Concrete param:  {totals.get('params_concrete_param',0):>5}  ({pct(totals.get('params_concrete_param',0), tp)})")
    print(f"    Exact Any:       {totals.get('params_exact_any',0):>5}  ({pct(totals.get('params_exact_any',0), tp)})")
    print(f"    Partial Any:     {totals.get('params_partial_any',0):>5}  ({pct(totals.get('params_partial_any',0), tp)})")
    print(f"    Blank:           {totals.get('params_blank',0):>5}  ({pct(totals.get('params_blank',0), tp)})")
    print()
    print(f"  --- Quality (returns) ---")
    print(f"    Concrete bare:   {totals.get('returns_concrete_bare',0):>5}  ({pct(totals.get('returns_concrete_bare',0), tr)})")
    print(f"    Concrete param:  {totals.get('returns_concrete_param',0):>5}  ({pct(totals.get('returns_concrete_param',0), tr)})")
    print(f"    Exact Any:       {totals.get('returns_exact_any',0):>5}  ({pct(totals.get('returns_exact_any',0), tr)})")
    print(f"    Partial Any:     {totals.get('returns_partial_any',0):>5}  ({pct(totals.get('returns_partial_any',0), tr)})")
    print(f"    Blank:           {totals.get('returns_blank',0):>5}  ({pct(totals.get('returns_blank',0), tr)})")


# ── Paired agreement analysis ────────────────────────────────────────────────

def paired_agreement(label, files_a, per_a, files_b, per_b):
    """For common files, compare per-function annotations between two settings."""
    common = set(files_a.keys()) & set(files_b.keys())
    counts = Counter()

    for stem in sorted(common):
        path_a, path_b = files_a[stem], files_b[stem]
        try:
            tree_a = ast.parse(path_a.read_text(encoding="utf-8", errors="ignore"))
            tree_b = ast.parse(path_b.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        funcs_a = _func_map(tree_a)
        funcs_b = _func_map(tree_b)

        for qname in set(funcs_a.keys()) & set(funcs_b.keys()):
            fa, fb = funcs_a[qname], funcs_b[qname]
            _compare_params(fa, fb, counts)
            _compare_returns(fa, fb, counts)

    total = sum(counts.values())
    print(f"\n{'=' * 60}")
    print(f"  Paired Agreement: {label}")
    print(f"{'=' * 60}")
    print(f"  Total annotation slots compared: {total}")
    for key in ["both_blank", "both_agree_typed", "both_agree_any", "a_only", "b_only", "both_differ"]:
        print(f"    {key:<16} {counts[key]:>6}  ({pct(counts[key], total)})")


def _func_map(tree):
    fmap = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fmap[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fmap[f"{node.name}.{child.name}"] = child
    return fmap


def _ann_str(node):
    if node is None:
        return None
    return ast.unparse(node)


def _compare_params(fa, fb, counts):
    args_a = [a for a in fa.args.args + list(getattr(fa.args, "posonlyargs", [])) + fa.args.kwonlyargs if a.arg not in ("self", "cls")]
    args_b = [a for a in fb.args.args + list(getattr(fb.args, "posonlyargs", [])) + fb.args.kwonlyargs if a.arg not in ("self", "cls")]

    names_a = {a.arg: a.annotation for a in args_a}
    names_b = {a.arg: a.annotation for a in args_b}

    for extra in [fa.args.vararg, fa.args.kwarg]:
        if extra:
            names_a[extra.arg] = extra.annotation
    for extra in [fb.args.vararg, fb.args.kwarg]:
        if extra:
            names_b[extra.arg] = extra.annotation

    for name in set(names_a.keys()) & set(names_b.keys()):
        sa, sb = _ann_str(names_a[name]), _ann_str(names_b[name])
        _tally(sa, sb, counts)


def _compare_returns(fa, fb, counts):
    sa, sb = _ann_str(fa.returns), _ann_str(fb.returns)
    _tally(sa, sb, counts)


def _tally(sa, sb, counts):
    if sa is None and sb is None:
        counts["both_blank"] += 1
    elif sa == sb:
        if sa == "Any":
            counts["both_agree_any"] += 1
        else:
            counts["both_agree_typed"] += 1
    elif sa is None:
        counts["b_only"] += 1
    elif sb is None:
        counts["a_only"] += 1
    else:
        counts["both_differ"] += 1


# ── Main ─────────────────────────────────────────────────────────────────────

def run_one(label, directory, recursive, allowed):
    if recursive:
        files = collect_recursive(directory, allowed)
    else:
        files = collect_flat(directory, allowed)

    per_file = {}
    parse_errors = 0
    for stem, path in sorted(files.items()):
        result = analyze_file(path)
        if result is None:
            parse_errors += 1
        per_file[stem] = result

    totals = aggregate(per_file)
    file_count = sum(1 for v in per_file.values() if v is not None)
    print_summary(label, totals, file_count, parse_errors)
    return files, per_file


def main():
    allowed = load_selected_stems()
    print(f"Filtering to {len(allowed)} selected files.\n")

    settings = [
        # ("Untyped (baseline)",        BENCH_DIR / "untyped_benchmarks",              False),
        ("GPT5 setting1 (inline)",    BENCH_DIR / "gpt5_2_run",                      True),
        ("GPT5 setting2 (stub)",      BENCH_DIR / "gpt5_1_infer_stub_run" / "merged", False),
        ("DeepSeek setting1 (inline)", BENCH_DIR / "deepseek_3_run",                 True),
        ("DeepSeek setting2 (stub)",  BENCH_DIR / "deepseek_3_stub_run" / "merged",  False),
    ]

    all_files = {}
    all_per_file = {}
    for label, directory, recursive in settings:
        files, per_file = run_one(label, directory, recursive, allowed)
        all_files[label] = files
        all_per_file[label] = per_file

    pairs = [
        ("GPT5 setting1 (inline)", "GPT5 setting2 (stub)"),
        ("DeepSeek setting1 (inline)", "DeepSeek setting2 (stub)"),
        # ("Untyped (baseline)", "GPT5 setting1 (inline)"),
        # ("Untyped (baseline)", "GPT5 setting2 (stub)"),
        # ("Untyped (baseline)", "DeepSeek setting1 (inline)"),
        # ("Untyped (baseline)", "DeepSeek setting2 (stub)"),
    ]

    for label_a, label_b in pairs:
        paired_agreement(
            f"{label_a} vs {label_b}",
            all_files[label_a], all_per_file[label_a],
            all_files[label_b], all_per_file[label_b],
        )


if __name__ == "__main__":
    main()
