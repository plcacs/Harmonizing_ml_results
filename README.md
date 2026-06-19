# HarmonizingML – Evaluating LLMs for Python Type Annotation Generation

This repository contains the source code, benchmark scripts, and results for **HarmonizingML**, an empirical research project that evaluates and compares Large Language Models (LLMs) for *automated Python type annotation generation*.

---

## Overview

Python's gradual type system (PEP 484 / PEP 526) allows developers to add type hints to their code, enabling static analysis tools such as [mypy](https://mypy.readthedocs.io/) to catch type errors early. Manually annotating large codebases is tedious; this project investigates how well modern LLMs can automate the task.

The study benchmarks multiple LLMs against each other and against dedicated type-annotation tools (**TypeWriter** and **HiTyper**) across several typing scenarios, measuring:

- **Compilation success rate** – does mypy accept the annotated file without errors?
- **Type precision** – are inferred types specific (e.g. `int`) rather than the catch-all `Any`?
- **Any-type ratio** – what fraction of annotations fall back to `Any`?
- **Code similarity** – how syntactically close are the LLM-generated versions to the originals?
- **Pairwise model comparisons** – where do models agree/disagree?

---

## Models & Tools Evaluated

| Category | Systems |
|---|---|
| **LLMs** | Claude 3 Sonnet, GPT-3.5, GPT-4, GPT-4o, GPT-5, DeepSeek, O1-Mini, O3-Mini, Llama 3.2 |
| **Type-annotation tools** | TypeWriter, HiTyper |
| **Baseline** | Original (untyped) code, human annotations |

---

## Typing Scenarios

Each model/tool is evaluated under the following conditions:

| Scenario | Description |
|---|---|
| **Original** | Baseline – the existing (possibly partially-typed) code from the dataset |
| **Untyped** | All type hints stripped from the original |
| **LLM-generated** | Type annotations added by the LLM (first run) |
| **Partial** | LLM output with partial annotations |
| **Full (user-annotated)** | LLM output aiming for complete annotations |
| **TypeWriter / HiTyper** | Annotations produced by traditional ML tools |

---

## Repository Structure

```
Harmonizing_ml_results/
├── README.md                          # This file
├── requirements.txt                   # Python package dependencies
├── environment.yml                    # Conda environment (Python 3.10)
│
├── llm_analysis.py                    # Main analysis script – compares mypy results,
│                                      #   any-ratios, and type info across all models/scenarios;
│                                      #   outputs LaTeX tables for academic publication
├── merge_json.py                      # Utility to merge multiple JSON result files
├── Command.txt                        # Developer notes and improvement ideas
│
├── data/                              # ManyTypes4Py dataset files
│   ├── README.md                      # Dataset field descriptions
│   ├── dataset_split.csv              # Train / validation / test file paths
│   ├── ManyTypes4PyDataset.spec       # GitHub project URLs and commit hashes
│   ├── MT4Py_VTHs.csv                 # Visible type hints and frequencies (~2 550 files)
│   ├── type_checked_files.txt         # Files successfully checked by mypy
│   ├── duplicate_files.txt            # Duplicate source files in the dataset
│   ├── mypy-dependents-by-stars.json  # Python projects that depend on mypy
│   └── mypy_output/                   # Raw mypy analysis on the original dataset
│
├── ManyTypes4py_benchmarks/           # Core benchmark directory (LLM generation + analysis)
│   ├── Generate_code_using_*.py       # LLM code-generation scripts (one per model)
│   ├── Generate_no_type_version.py    # Generate untyped baseline versions
│   ├── Generate_original_files.py     # Process/copy original files
│   ├── AddType_Typewriter_*.py        # TypeWriter integration scripts
│   ├── mypy_results/                  # mypy output JSONs (per model, per scenario)
│   ├── Type_info_collector/           # Type information extraction & analysis
│   ├── Type_writer_results/           # TypeWriter tool outputs
│   ├── HiTyper_1st_run/               # HiTyper tool outputs
│   ├── Full_repo_analysis/            # Repository-level analysis
│   ├── Human_annotations_detailed_analysis/  # Quality study on human annotations
│   ├── GPCE_mypy_results/             # GPT4 Code Editor mypy results
│   ├── mypy_fix_analysis/             # Analysis of mypy-error corrections
│   ├── callgraph_analysis/            # Call graph / dependency analysis
│   ├── Code_similarity.py             # Code similarity metrics
│   └── Generate_Table1.py             # Generate main publication table
│
├── important_scripts/                 # Utility and post-processing scripts
│   ├── Run_raw_mypy.py                # Execute mypy on raw source files
│   ├── extract_type_hints.py          # Extract type annotation data
│   ├── Analyze_commom_file_issue.py   # Identify common failure patterns
│   ├── Common_file_find.py            # Find files processed by every model
│   └── old_new_comparison/            # Scripts comparing earlier vs later LLM runs
│
├── LLM_Generated_code/                # Generated annotated source files from all LLMs
├── model_results/                     # Consolidated per-model result files
├── results/                           # Final analysis outputs (JSON + CSV)
├── pyperformance_benchmarks/          # Python runtime-performance benchmarks
│   ├── bm_2to3/, bm_asyncio/, …      # 24 individual benchmark suites
│   └── mypy_results/                  # mypy results for the performance suite
├── benchmarks/                        # Additional benchmark datasets
├── Type_errors_results/               # Type-error analysis outputs
└── cloned_repos/                      # Sample cloned GitHub repositories
```

---

## Pipeline / Workflow

```
1. DATASET PREPARATION
   └─► data/  (Python projects from GitHub, processed with libsa4py)
       └─► dataset_split.csv  (train / val / test split)

2. CODE GENERATION
   └─► ManyTypes4py_benchmarks/Generate_code_using_<model>.py
       └─► LLM_Generated_code/  (annotated source files per model)

3. TYPE CHECKING
   └─► Run mypy on original, LLM-generated, TypeWriter, and HiTyper outputs
       └─► ManyTypes4py_benchmarks/mypy_results/  (compilation status + errors)

4. METRICS EXTRACTION
   ├─► Type_info_collector/  →  Type_info_*.json
   ├─► per_file_any_percentage.json  (Any-type ratio per file)
   └─► Code_similarity.py  →  syntactic feature vectors

5. COMPARATIVE ANALYSIS
   └─► llm_analysis.py  (LLM vs LLM, scenario vs scenario, precision scoring)

6. PERFORMANCE BENCHMARKING
   └─► pyperformance_benchmarks/  (runtime impact of type annotations)

7. PUBLICATION OUTPUT
   └─► LaTeX tables generated by llm_analysis.py --latex
```

---

## Getting Started

### 1. Install dependencies

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate harmonizing_ml

# Or using pip
pip install -r requirements.txt
```

### 2. Run the main analysis

```bash
# Analyse results for a specific LLM (e.g. deepseek, claude, o3-mini)
python llm_analysis.py --llm deepseek

# Generate LaTeX-formatted output for the paper
python llm_analysis.py --llm deepseek --latex

# Show delta/difference tables
python llm_analysis.py --llm deepseek --latex --difference
```

### 3. Generate annotated code with an LLM

Each `Generate_code_using_<model>.py` script in `ManyTypes4py_benchmarks/` calls the corresponding model API and writes the annotated Python files to `LLM_Generated_code/`.  
Set the relevant API key in your environment (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) before running.

---

## Key Metrics

| Metric | How it is measured |
|---|---|
| **Compilation success rate** | Fraction of files accepted by `mypy` without errors |
| **Any-type ratio** | Fraction of annotations that are `Any` (lower = more precise) |
| **Precision score** | Custom scoring: specific types score higher than generic ones |
| **Organic gain/loss** | Files that newly pass / newly fail mypy vs the baseline |
| **Common-file success** | Success rate restricted to files processable by *all* models |

---

## Dataset

The project uses the **ManyTypes4Py** dataset – a large curated collection of Python source files with type annotations sourced from popular open-source GitHub projects.  
See [`data/README.md`](data/README.md) for a full description of the dataset files.

---

## Dependencies

Core libraries used throughout the project:

- **mypy** – static type checker
- **pandas / numpy / scipy** – data manipulation and statistics
- **matplotlib / seaborn / plotly** – visualisation
- **openai / anthropic** – LLM API clients
- **transformers / torch** – HuggingFace / PyTorch (used by TypeWriter and HiTyper)
- **ast / astor** – Python AST manipulation
- **networkx** – call-graph analysis
- **scikit-learn** – machine-learning utilities

See [`requirements.txt`](requirements.txt) for the full list with pinned versions.

---

## Citation

If you use this work, please cite the accompanying paper (details to be added upon publication).
