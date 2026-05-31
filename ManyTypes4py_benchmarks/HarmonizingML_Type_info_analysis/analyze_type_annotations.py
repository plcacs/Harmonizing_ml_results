"""
Analyze type annotation quality across strict, unstrict, and stub settings.

For each model (DeepSeek, GPT-5), compares annotation quality in:
  - strict: fully strict LLM-annotated .py files
  - unstrict: unstrict LLM-annotated .py files
  - stub: merged stub+source .py files

Only analyzes the 500 canonical files from 500_untyped_files.

Each parameter/return annotation is classified as:
  concrete, exact_any, partial_any, or blank.

Usage:
    python analyze_type_annotations.py
"""

import ast
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

CANONICAL_DIR = PARENT_DIR / "500_untyped_files"

MODELS = {
    "DeepSeek": {
        "strict": {"dir": PARENT_DIR / "deepseek_4_run", "recursive": True},
        "unstrict": {"dir": PARENT_DIR / "deepseek_3_run", "recursive": True},
        "stub": {"dir": PARENT_DIR / "deepseek_3_stub_run" / "merged", "recursive": False},
    },
    "GPT-5": {
        "strict": {"dir": PARENT_DIR / "gpt5_4_run", "recursive": True},
        "unstrict": {"dir": PARENT_DIR / "gpt5_1st_run", "recursive": True},
        "stub": {"dir": PARENT_DIR / "gpt5_1_infer_stub_run" / "merged", "recursive": False},
    },
}


def get_canonical_filenames() -> set[str]:
    return {f.stem for f in CANONICAL_DIR.iterdir() if f.suffix == ".py"}


def contains_any(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and contains_any(child):
            return True
    return False


def is_exact_any(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    return False


def classify_annotation(node) -> str:
    if node is None:
        return "blank"
    if is_exact_any(node):
        return "exact_any"
    if contains_any(node):
        return "partial_any"
    return "concrete"


def analyze_file(filepath: Path) -> dict | None:
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None

    stats = {
        "total_params": 0,
        "total_returns": 0,
        "params_blank": 0,
        "params_exact_any": 0,
        "params_partial_any": 0,
        "params_concrete": 0,
        "returns_blank": 0,
        "returns_exact_any": 0,
        "returns_partial_any": 0,
        "returns_concrete": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            stats["total_params"] += 1
            cat = classify_annotation(arg.annotation)
            stats[f"params_{cat}"] += 1

        if node.args.vararg:
            stats["total_params"] += 1
            cat = classify_annotation(node.args.vararg.annotation)
            stats[f"params_{cat}"] += 1

        if node.args.kwarg:
            stats["total_params"] += 1
            cat = classify_annotation(node.args.kwarg.annotation)
            stats[f"params_{cat}"] += 1

        stats["total_returns"] += 1
        cat = classify_annotation(node.returns)
        stats[f"returns_{cat}"] += 1

    return stats


def collect_files(directory: Path, recursive: bool, canonical: set[str]) -> dict[str, Path]:
    pattern = "**/*.py" if recursive else "*.py"
    files = {}
    for p in directory.glob(pattern):
        if p.stem in canonical and p.stem not in files:
            files[p.stem] = p
    return files


def aggregate(per_file: dict[str, dict | None]) -> dict:
    totals = {}
    for stats in per_file.values():
        if stats is None:
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    return totals


def pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:.1f}%" if denom else "N/A"


def run_analysis(directory: Path, recursive: bool, canonical: set[str]) -> dict:
    files = collect_files(directory, recursive, canonical)
    per_file = {}
    parse_errors = 0
    for name, path in sorted(files.items()):
        result = analyze_file(path)
        if result is None:
            parse_errors += 1
        per_file[name] = result

    totals = aggregate(per_file)
    analyzed = sum(1 for v in per_file.values() if v is not None)
    return {
        "analyzed": analyzed,
        "parse_errors": parse_errors,
        "missing": len(canonical) - len(files),
        "totals": totals,
    }


def main() -> None:
    canonical = get_canonical_filenames()
    print(f"Canonical files: {len(canonical)}\n")

    for model_name, settings in MODELS.items():
        print(f"\n{'=' * 90}")
        print(f"  {model_name}")
        print(f"{'=' * 90}")

        header = (
            f"{'Setting':<10} {'Files':>6} {'Errs':>5} {'Miss':>5} │ "
            f"{'Params':>6} {'Conc%':>7} {'ExAny%':>7} {'PtAny%':>7} {'Blank%':>7} │ "
            f"{'Rets':>5} {'Conc%':>7} {'ExAny%':>7} {'PtAny%':>7} {'Blank%':>7}"
        )
        print(header)
        print("─" * len(header))

        for setting_name, cfg in settings.items():
            r = run_analysis(cfg["dir"], cfg["recursive"], canonical)
            t = r["totals"]
            tp = t.get("total_params", 0)
            tr = t.get("total_returns", 0)

            print(
                f"{setting_name:<10} {r['analyzed']:>6} {r['parse_errors']:>5} {r['missing']:>5} │ "
                f"{tp:>6} {pct(t.get('params_concrete', 0), tp):>7} "
                f"{pct(t.get('params_exact_any', 0), tp):>7} "
                f"{pct(t.get('params_partial_any', 0), tp):>7} "
                f"{pct(t.get('params_blank', 0), tp):>7} │ "
                f"{tr:>5} {pct(t.get('returns_concrete', 0), tr):>7} "
                f"{pct(t.get('returns_exact_any', 0), tr):>7} "
                f"{pct(t.get('returns_partial_any', 0), tr):>7} "
                f"{pct(t.get('returns_blank', 0), tr):>7}"
            )

        print("─" * len(header))


if __name__ == "__main__":
    main()
