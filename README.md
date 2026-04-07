# Harmonizing ML Results

This repository contains the source code, data, scripts, and results for the **HarmonizingML** research project.
The project systematically evaluates how well Large Language Models (LLMs) can generate Python type annotations
(type hints) for previously unannotated code, and compares their output against human-written annotations and
established type-inference tools such as [HiTyper](https://github.com/JohnnyPeng18/HiTyper) and
[Typewriter](https://github.com/typilus/typewriter).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Key Scripts](#key-scripts)
- [Data Files](#data-files)
- [Workflow & Pipeline](#workflow--pipeline)
- [Results Summary](#results-summary)
- [Setup & Installation](#setup--installation)

---

## Project Overview

### Goals

1. **Compare LLM Type-Annotation Quality** – Evaluate five state-of-the-art LLMs
   (Claude 3.5 Sonnet, DeepSeek, O1-mini, O3-mini, GPT-4o) on their ability to add correct Python
   type hints to unannotated source files.
2. **Measure Type Correctness** – Use [mypy](https://mypy.readthedocs.io/) as a ground-truth type
   checker to measure compilation success rates and error counts.
3. **Assess Type Precision** – Track the ratio of non-`Any` annotations as a proxy for type specificity.
4. **Benchmark Against Baselines** – Compare LLM outputs to human annotations and to tool-generated
   annotations (HiTyper, Typewriter, MonkeyType).
5. **Study Code Similarity** – Determine whether different LLMs produce semantically equivalent code
   for the same input.
6. **Measure Performance Impact** – Run the
   [pyperformance](https://github.com/python/pyperformance) suite on typed vs. untyped code to assess
   runtime differences.

### Dataset

The benchmark dataset is derived from
[ManyTypes4Py](https://github.com/saltudelft/many-types-4-py), a large-scale Python dataset of
typed and untyped files collected from GitHub projects that depend on mypy.  The `data/` directory
holds the dataset metadata, repository specs, and type-frequency statistics.

---

## Repository Structure

```
project_root/
│
├── data/                           # Dataset metadata and preparation scripts
│   ├── clone_respos.py             # Clones the top-50 mypy-dependent GitHub repos
│   ├── Code_similarity.py          # Compares code similarity across LLM outputs
│   ├── MT4Py_VTHs.csv              # Visible type-hint frequency table (~1.8 M rows)
│   ├── ManyTypes4PyDataset.spec    # GitHub URLs + commit hashes for ~2,550 benchmark files
│   ├── dataset_split.csv           # Train / validation / test file paths (~23 MB)
│   ├── duplicate_files.txt         # Detected duplicate source files (~30 MB)
│   ├── type_checked_files.txt      # Files successfully checked by mypy (~15 MB)
│   ├── mypy-dependents-by-stars.json  # Top mypy-dependent projects ranked by stars
│   └── repo_analysis.csv           # Cloned-repo metadata (name, file count, size)
│
├── ManyTypes4py_benchmarks/        # Main benchmark suite
│   ├── original_files/             # Unannotated baseline Python files
│   ├── partial_annotated_*/        # Selectively (partially) typed variants
│   ├── {llm_name}_{run}/           # LLM-generated fully-typed variants, one dir per run
│   ├── Human_annotations_*/        # Human-written type annotations (baseline)
│   ├── mypy_results/               # Mypy output JSON files (errors, success/failure)
│   ├── Full_repo_analysis/         # Whole-repository mypy analysis artifacts
│   └── Type_info_collector/        # Post-processed type-annotation statistics
│       ├── Section_04/             # Per-file `Any`-percentage breakdown
│       ├── Section_06/             # Human vs. LLM annotation comparison
│       ├── Type_info_LLMS/         # Per-LLM type-info JSONs
│       └── llm_vs_llm_comparison/  # Cross-LLM comparison artifacts
│
├── pyperformance_benchmarks/       # Runtime performance benchmark suite
│   ├── LLM_Gen_Files/              # LLM-generated benchmark implementations
│   ├── {llm_name}_*_run/           # Per-LLM benchmark execution results
│   └── Type_info_collector/        # Type-annotation stats for benchmark files
│
├── benchmarks/                     # Raw pyperformance benchmark sources (80 + benchmarks)
│   └── bm_{name}/                  # One directory per benchmark (e.g. bm_json_dumps)
│
├── LLM_Generated_code/             # Raw LLM API outputs (DeepSeek, O1-mini, GPT-4o, …)
├── LLM_GEN_CODE_inefficient/       # Intentionally inefficient LLM code variants
├── model_results/                  # Aggregated model evaluation artifacts
├── results/                        # Final per-model analysis JSON files
├── Type_errors_results/             # Detailed mypy type-error JSON files per model
├── cloned_repos/                   # Locally cloned GitHub repositories used as test beds
│
├── important_scripts/              # Core analysis helper scripts
│   ├── Common_file_find.py         # Identifies files shared across multiple LLM runs
│   ├── Analyze_common_file_issue.py # Quality comparison for shared files
│   ├── Run_raw_mypy.py             # Batch mypy runner; outputs per-file error counts
│   ├── extract_type_hints.py       # Extracts and compares type hints (old vs. new run)
│   ├── run_type_analysis.py        # Orchestrates type-analysis pipeline
│   └── old_new_comparison/         # Side-by-side diff artifacts for run pairs
│
├── llm_analysis.py                 # Main LLM comparison script (root level)
├── merge_json.py                   # Merges multiple JSON result files into one
├── requirements.txt                # Pinned Python dependency list
├── environment.yml                 # Conda environment definition (Python 3.10, ppmlenv)
└── correlation_test_results.csv    # Summary statistics & correlation metrics per model
```

---

## Key Scripts

### Root-level

| Script | Purpose |
|---|---|
| `llm_analysis.py` | Central analysis framework. Loads mypy results, any-ratio data, and type-info files for each LLM. Calculates non-`Any` ratios, precision scores, organic typing gains/losses, and compares all models side-by-side. Supports three annotation levels: *original*, *partial*, and *full*. |
| `merge_json.py` | Utility that merges multiple JSON result files produced by separate benchmark runs into a single consolidated file. |

### `important_scripts/`

| Script | Purpose |
|---|---|
| `Common_file_find.py` | Compares syntactic-feature sets (extracted via the `ast` module) across two LLM runs to find files with matching structure in both. |
| `Analyze_commom_file_issue.py` | For files identified as common across runs, counts typed parameters, total parameters, and variables; runs mypy; and writes a detailed JSON report. |
| `Run_raw_mypy.py` | Batch-runs mypy over a directory of Python files, captures per-file error counts and annotation statistics, and saves results as JSON. |
| `extract_type_hints.py` | Parses Python files with the `ast` module to extract function-parameter and return-type annotations; optionally diffs old-run vs. new-run annotation coverage. |
| `run_type_analysis.py` | Orchestrates the type-analysis pipeline by calling the above helpers in sequence. |

### `data/`

| Script | Purpose |
|---|---|
| `clone_respos.py` | Reads `mypy-dependents-by-stars.json`, clones the top-50 repositories, counts Python files, and records metadata in `repo_analysis.csv`. |
| `Code_similarity.py` | Strips type annotations from LLM outputs (DeepSeek, O1-mini, GPT-4o) and uses `difflib` to find file triplets with >90 % semantic similarity across models. |

### `ManyTypes4py_benchmarks/`

| Script pattern | Purpose |
|---|---|
| `Generate_code_using_OPENAI.py` | Calls the OpenAI API (GPT-3.5 / GPT-4) to add type hints to each benchmark file, tracking token usage. |
| `Generate_code_using_deepseek.py` | Same pipeline using the DeepSeek API. |
| `Generate_code_using_o1_mini.py` | Same pipeline using the O1-mini API. |
| `Generate_code_using_claude3_sonnet.py` | Same pipeline using the Anthropic API (Claude 3.5 Sonnet). |
| `AddType_Typewriter_original.py` | Applies the Typewriter tool to the same benchmark files to produce a tool-generated baseline. |

---

## Data Files

| File | Description |
|---|---|
| `correlation_test_results.csv` | Per-model summary: files analysed, mypy compilation success rate, mean `Any` count, Point-Biserial / Spearman / Pearson correlation coefficients and p-values. |
| `simple_correlation_results.csv` | Simplified version of the above with the most important metrics. |
| `data/MT4Py_VTHs.csv` | Visible type-hint frequency table for the ManyTypes4Py dataset (~1.8 M rows). |
| `data/ManyTypes4PyDataset.spec` | GitHub repository URLs and commit hashes for all ~2,550 benchmark files. |
| `data/dataset_split.csv` | Train / validation / test partition for the dataset (~23 MB). |
| `data/repo_analysis.csv` | Metadata for each cloned repository (star count, file count, size on disk). |
| `results/{model}_analysis.json` | Detailed per-model analysis artifacts (non-`Any` ratios, precision scores, etc.). |
| `Type_errors_results/type_errors_{model}.json` | Full mypy error details per model (~50–90 KB each). |

---

## Workflow & Pipeline

```
1. DATA PREPARATION
   └─ clone_respos.py ──► clone top-50 mypy-dependent repos
                      ──► count files ──► repo_analysis.csv

2. CODE GENERATION  (one run per LLM × annotation level)
   └─ Generate_code_using_*.py
         ├─ input : original_files/ (unannotated)
         ├─ output: {llm_name}_{run}/ (fully typed)
         └─ also produces partial_annotated_* variants

3. TYPE CHECKING
   └─ Run_raw_mypy.py / batch mypy
         ├─ collects: error counts, success/failure per file
         └─ stores  : mypy_results/*.json

4. TYPE-HINT EXTRACTION & ANALYSIS
   └─ extract_type_hints.py + run_type_analysis.py
         ├─ counts   : annotated parameters, return types, Any usage
         ├─ calculates: non-Any ratio, type-precision score
         └─ stores   : Type_info_collector/ per-LLM JSONs

5. CODE SIMILARITY
   └─ Code_similarity.py
         ├─ strips annotations ──► difflib comparison
         └─ flags files with >90 % similarity across models

6. PERFORMANCE BENCHMARKING
   └─ pyperformance_benchmarks/
         ├─ runs 80 + benchmarks on typed vs. untyped code
         └─ records execution-time deltas per LLM

7. CROSS-MODEL COMPARISON
   └─ llm_analysis.py
         ├─ loads mypy results + any-ratio + type-info files
         ├─ computes: non-Any ratios, precision, gains/losses
         ├─ compares: original / partial / full annotation levels
         └─ outputs : results/ JSON + correlation_test_results.csv

8. REPORTING
   └─ merge_json.py ──► consolidates JSON artifacts
      matplotlib / seaborn / plotly ──► figures for publication
```

---

## Results Summary

The table below is reproduced from `correlation_test_results.csv`:

| Model | Files Analysed | mypy Success Rate | Mean `Any` Count | Spearman r (errors) | Pearson r (errors) |
|---|---|---|---|---|---|
| Human | 356 | **39.6 %** | 3.6 | -0.133 | 0.229 |
| GPT-4o | 723 | 30.3 % | 6.2 | 0.032 | 0.097 |
| O1-mini | 1 083 | 25.1 % | 7.3 | -0.057 | 0.126 |
| **O3-mini** | **1 458** | **38.5 %** | 10.0 | -0.099 | **0.156** |
| DeepSeek | 601 | 33.9 % | 5.1 | 0.013 | 0.010 |
| Claude 3-Sonnet | 921 | **38.7 %** | 7.0 | -0.020 | 0.049 |

Key observations:
- **O3-mini** and **Claude 3-Sonnet** match or approach the human baseline mypy success rate (~39 %).
- Higher `Any` counts do not reliably predict worse mypy outcomes (weak / near-zero correlations).
- All LLMs generate substantially more annotations than humans, but precision varies widely.

---

## Setup & Installation

### Conda (recommended)

```bash
conda env create -f environment.yml   # creates the 'ppmlenv' environment
conda activate ppmlenv
pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` (or create `.env`) and add your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
```

### Running the analysis

```bash
# 1. Run mypy on a set of LLM-generated files
python important_scripts/Run_raw_mypy.py

# 2. Extract and compare type hints
python important_scripts/extract_type_hints.py

# 3. Run the full cross-model comparison
python llm_analysis.py

# 4. Merge JSON artefacts if needed
python merge_json.py
```
