"""
Shared settings for strict vs unstrict prompt analysis.

Reuses AST analysis and build_joined from annotation_vs_mypy_joined.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from annotation_vs_mypy_joined import (  # noqa: E402
    build_joined,
    load_selected_stems,
    analyze_file_ast,
)

BENCH_DIR = ANALYSIS_DIR.parent
MYPY_DIR = BENCH_DIR / "HarmonizingML_mypy_results" / "mypy_outputs"

SETTINGS = [
    {
        "label": "GPT5 unstrict",
        "src_dir": BENCH_DIR / "gpt5_2_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_gpt5_2_run_with_errors.json",
    },
    {
        "label": "GPT5 strict",
        "src_dir": BENCH_DIR / "gpt5_4_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_gpt5_4_run_with_errors.json",
    },
    {
        "label": "DeepSeek unstrict",
        "src_dir": BENCH_DIR / "deep_seek_2nd_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_deepseek_2nd_run_with_errors.json",
    },
    {
        "label": "DeepSeek strict",
        "src_dir": BENCH_DIR / "deepseek_4_run",
        "recursive": True,
        "mypy_json": MYPY_DIR / "mypy_results_deepseek_4_run_with_errors_strict.json",
    },
]

SETTING_PAIRS = [
    ("GPT5 unstrict",     "GPT5 strict"),
    ("DeepSeek unstrict", "DeepSeek strict"),
]

COLOR_MAP = {
    "GPT5 unstrict":     "#1f77b4",
    "GPT5 strict":       "#ff7f0e",
    "DeepSeek unstrict": "#2ca02c",
    "DeepSeek strict":   "#d62728",
}

SHORT_NAMES = {
    "GPT5 unstrict":     "GPT5-unstrict",
    "GPT5 strict":       "GPT5-strict",
    "DeepSeek unstrict": "DS-unstrict",
    "DeepSeek strict":   "DS-strict",
}


def short(label):
    return SHORT_NAMES.get(label, label)


def load_rows_by_setting():
    allowed = load_selected_stems()
    return {s["label"]: build_joined(s, allowed) for s in SETTINGS}
