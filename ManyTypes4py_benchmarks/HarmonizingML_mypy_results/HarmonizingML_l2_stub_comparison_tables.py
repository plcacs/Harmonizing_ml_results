"""
Print summary tables for L2 stub experiments (matching analyze_type_annotations.py
and HarmonizingML_Table_strict_unstrict_stub_comparison.py format).

Experiments:
  - GPT-5 L2 stub       (gpt5_l2_stub_run)
  - Claude Opus L2 stub (claude_opus_l2_stub_run)
  - Claude Opus stub    (claude_opus_stub_run, no L2)

Usage:
    python HarmonizingML_l2_stub_comparison_tables.py
    python HarmonizingML_l2_stub_comparison_tables.py --save
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
BENCH = BASE.parent
MYPY_DIR = BASE / "mypy_outputs"
CANONICAL_DIR = BENCH / "500_untyped_files"
OUTPUT_TXT = BENCH / "Analysis_Results" / "Mypy_Typeinfo_l2_stub_comparison.txt"

NON_TYPE_RELATED_ERRORS = {
    "name-defined", "import", "syntax", "no-redef", "unused-ignore",
    "override-without-super", "redundant-cast", "literal-required",
    "typeddict-unknown-key", "typeddict-item", "truthy-function",
    "str-bytes-safe", "unused-coroutine", "explicit-override",
    "truthy-iterable", "redundant-self", "redundant-await", "unreachable",
}

EXPERIMENTS = {
    "GPT-5": {
        "L2 stub": {
            "stub_dir": BENCH / "gpt5_l2_stub_run",
            "mypy_json": MYPY_DIR / "mypy_results_gpt5_l2_stub_run_with_errors.json",
        },
    },
    "Claude Opus 4.6": {
        "L2 stub": {
            "stub_dir": BENCH / "claude_opus_l2_stub_run",
            "mypy_json": MYPY_DIR / "mypy_results_claude_opus_l2_stub_run_with_errors.json",
        },
        "stub": {
            "stub_dir": BENCH / "claude_opus_stub_run",
            "mypy_json": MYPY_DIR / "mypy_results_claude_opus_stub_run_with_errors.json",
        },
    },
}


def get_canonical_filenames() -> list[str]:
    return sorted(f.name for f in CANONICAL_DIR.iterdir() if f.suffix == ".py")


def get_canonical_stems() -> set[str]:
    return {Path(name).stem for name in get_canonical_filenames()}


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


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


def _classify(node) -> str:
    if node is None:
        return "blank"
    if _is_exact_any(node):
        return "exact_any"
    if _contains_any(node):
        return "partial_any"
    return "concrete"


def _analyze_stub_file(path: Path) -> dict | None:
    try:
        source = _strip_markdown_fences(path.read_text(encoding="utf-8", errors="ignore"))
        tree = ast.parse(source, filename=str(path))
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

        for arg in (
            list(getattr(node.args, "posonlyargs", []))
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            if arg.arg in ("self", "cls"):
                continue
            stats["total_params"] += 1
            cat = _classify(arg.annotation)
            stats[f"params_{cat}"] += 1

        if node.args.vararg:
            stats["total_params"] += 1
            cat = _classify(node.args.vararg.annotation)
            stats[f"params_{cat}"] += 1
        if node.args.kwarg:
            stats["total_params"] += 1
            cat = _classify(node.args.kwarg.annotation)
            stats[f"params_{cat}"] += 1

        stats["total_returns"] += 1
        cat = _classify(node.returns)
        stats[f"returns_{cat}"] += 1

    return stats


def analyze_stub_dir(stub_dir: Path, canonical_stems: set[str]) -> dict:
    per_file: dict[str, dict | None] = {}
    parse_errors = 0
    missing = 0

    for stem in canonical_stems:
        stub_path = stub_dir / f"{stem}.pyi"
        if not stub_path.is_file():
            missing += 1
            continue
        result = _analyze_stub_file(stub_path)
        if result is None:
            parse_errors += 1
        per_file[stem] = result

    totals: dict[str, int] = {}
    for stats in per_file.values():
        if stats is None:
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v

    analyzed = sum(1 for v in per_file.values() if v is not None)
    return {
        "analyzed": analyzed,
        "parse_errors": parse_errors,
        "missing": missing,
        "totals": totals,
    }


def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:.1f}%" if denom else "N/A"


def _has_non_type_error(errors: list[str]) -> bool:
    if any(
        kw in error.lower()
        for error in errors
        for kw in ["syntax", "empty_body", "name_defined"]
    ):
        return True
    for error in errors:
        m = re.search(r"\[([a-z\-]+)\]", error)
        if m and m.group(1) in NON_TYPE_RELATED_ERRORS:
            return True
    return False


def analyze_mypy_json(mypy_path: Path, canonical_files: list[str]) -> dict:
    if not mypy_path.is_file():
        return {
            "unprocessed": len(canonical_files),
            "both_success": 0,
            "llm_only_fail": 0,
            "success_rate": 0.0,
            "overall_success": 0.0,
            "missing_json": True,
        }

    with open(mypy_path, encoding="utf-8") as f:
        data = json.load(f)

    unprocessed: list[str] = []
    both_success: list[str] = []
    llm_only_fail: list[str] = []
    missing: list[str] = []

    for fname in canonical_files:
        if fname not in data:
            missing.append(fname)
            continue
        file_data = data[fname]
        if _has_non_type_error(file_data.get("errors", [])):
            unprocessed.append(fname)
            continue
        if file_data.get("error_count", 0) == 0:
            both_success.append(fname)
        else:
            llm_only_fail.append(fname)

    total_unprocessed = len(unprocessed) + len(missing)
    evaluable = len(both_success) + len(llm_only_fail)
    success_rate = 100 * len(both_success) / evaluable if evaluable else 0.0
    overall = (
        100 * len(both_success) / (evaluable + total_unprocessed)
        if (evaluable + total_unprocessed)
        else 0.0
    )
    return {
        "unprocessed": total_unprocessed,
        "both_success": len(both_success),
        "llm_only_fail": len(llm_only_fail),
        "success_rate": success_rate,
        "overall_success": overall,
        "missing_json": False,
    }


def print_type_annotation_tables(canonical_stems: set[str], out) -> None:
    out.write("Type annotation analysis:\n")
    for model_name, settings in EXPERIMENTS.items():
        out.write("=" * 90 + "\n")
        out.write(f"  {model_name}\n")
        out.write("=" * 90 + "\n")
        header = (
            f"{'Setting':<10} {'Files':>6} {'Errs':>5} {'Miss':>5} │ "
            f"{'Params':>6} {'Conc%':>7} {'ExAny%':>7} {'PtAny%':>7} {'Blank%':>7} │ "
            f"{'Rets':>5} {'Conc%':>7} {'ExAny%':>7} {'PtAny%':>7} {'Blank%':>7}\n"
        )
        out.write(header)
        out.write("─" * len(header.rstrip()) + "\n")

        for setting_name, cfg in settings.items():
            r = analyze_stub_dir(cfg["stub_dir"], canonical_stems)
            t = r["totals"]
            tp = t.get("total_params", 0)
            tr = t.get("total_returns", 0)
            out.write(
                f"{setting_name:<10} {r['analyzed']:>6} {r['parse_errors']:>5} {r['missing']:>5} │ "
                f"{tp:>6} {_pct(t.get('params_concrete', 0), tp):>7} "
                f"{_pct(t.get('params_exact_any', 0), tp):>7} "
                f"{_pct(t.get('params_partial_any', 0), tp):>7} "
                f"{_pct(t.get('params_blank', 0), tp):>7} │ "
                f"{tr:>5} {_pct(t.get('returns_concrete', 0), tr):>7} "
                f"{_pct(t.get('returns_exact_any', 0), tr):>7} "
                f"{_pct(t.get('returns_partial_any', 0), tr):>7} "
                f"{_pct(t.get('returns_blank', 0), tr):>7}\n"
            )
        out.write("─" * len(header.rstrip()) + "\n\n")


def print_mypy_tables(canonical_files: list[str], out) -> None:
    out.write("Mypy result analysis:\n\n")
    header = (
        f"{'Model':<16} {'Setting':<10} {'Unprocessed':>12} {'Success':>9} "
        f"{'LLM Fail':>10} {'Success%':>10} {'Overall%':>10} {'FileChanges':>12}\n"
    )
    sep = "-" * len(header.rstrip())

    for model_name, settings in EXPERIMENTS.items():
        out.write(f"{model_name}\n")
        out.write(sep + "\n")
        out.write(header)
        out.write(sep + "\n")
        for setting_name, cfg in settings.items():
            r = analyze_mypy_json(cfg["mypy_json"], canonical_files)
            if r.get("missing_json"):
                out.write(
                    f"{model_name:<16} {setting_name:<10} {'MISSING':>12} "
                    f"{'N/A':>9} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}\n"
                )
                continue
            out.write(
                f"{model_name:<16} {setting_name:<10} {r['unprocessed']:>12} "
                f"{r['both_success']:>9} {r['llm_only_fail']:>10} "
                f"{r['success_rate']:>9.2f}% {r['overall_success']:>9.2f}% {'N/A':>12}\n"
            )
        out.write(sep + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 stub comparison tables")
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Write output to {OUTPUT_TXT}",
    )
    args = parser.parse_args()

    canonical_files = get_canonical_filenames()
    canonical_stems = get_canonical_stems()

    lines: list[str] = []

    class Writer:
        def write(self, s: str) -> None:
            sys.stdout.write(s)
            lines.append(s)

    writer = Writer()
    print_type_annotation_tables(canonical_stems, writer)
    print_mypy_tables(canonical_files, writer)

    if args.save:
        OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_TXT.write_text("".join(lines), encoding="utf-8")
        print(f"Saved to {OUTPUT_TXT}", file=sys.stderr)


if __name__ == "__main__":
    main()
