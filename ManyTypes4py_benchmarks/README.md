# ManyTypes4py benchmarks — experiments and scripts

This folder holds the **ManyTypes4py-style benchmark corpus** (cloned open-source Python projects), **generated datasets** (untyped / partially typed / LLM-annotated), **mypy and type-info artifacts**, and **analysis scripts** for comparing LLMs, human-style partial typing, static tools, and code structure.

**How to use this README:** Edit sections below when you add a new run directory, JSON path, or driver script. Keep the **“Artifact directories”** list in sync with top-level folders that are *outputs* (not third-party repos).

---

## End-to-end pipeline (typical)

1. **Select files** → JSON lists under `Files_not_for_root_directories/` (e.g. `filtered_python_files.json`, `grouped_file_paths.json`).
2. **Corpus** → `Generate_no_type_version.py` / `Generate_original_files.py` → `untyped_benchmarks/`, `original_files/`.
3. **Annotate** → `Generate_code_using_*.py` (or subdirectory variants) → per-model folders (`deep_seek_2nd_run/`, `gpt5_1st_run/`, …).
4. **Check types** → `Run_mypy_on_llm_2.py` (or `Run_mypy_on_llm.py`, `Human_annotations_detailed_analysis/Run_mypy_on_llm_percent.py`) → JSON under `mypy_results/`, `GPCE_mypy_results/`.
5. **Optional repair** → `deepseek_mypy_fix/`, `gpt5_mypy_fix/`, `claude_mypy_fix/`.
6. **Metrics / papers** → `Type_info_collector/` scripts, repo-root `llm_analysis.py` (parent folder), tables in `Generate_Table1.py`, `Table_3_generate.py`, etc.

---

## Shared inputs (`Files_not_for_root_directories/`)

| File (examples) | Role |
|-----------------|------|
| `filtered_python_files.json` | Paths grouped by annotation-density buckets (`50+`, `30-50`, …) for corpus generation. |
| `grouped_file_paths.json` | Batched untyped paths (groups `1`–`18`) for most **full-file LLM annotation** scripts. |
| `processed_files_*.txt` | Append-only logs of completed paths per run (resume / skip). |
| `*_model_timings*.json` | Per-request or per-file timing logs. |

Paths in JSON are often **relative to this directory**; run scripts with **`ManyTypes4py_benchmarks` as the current working directory** unless a script sets paths otherwise.

---

## Artifact directories (experiment outputs)

These are **top-level folders you own** (contrast with cloned projects like `pandas/`, `aiohttp/`).

### Baseline corpus

| Directory | Description |
|-----------|-------------|
| `untyped_benchmarks/` | AST-stripped copies; names `{stem}_{hash6}.py` align with originals. |
| `original_files/` | **Typed originals** copied with the **same** `{stem}_{hash6}.py` names as untyped (see `Generate_original_files.py`). |
| `Hundrad_original_typed_benchmarks/` | Subset sampled from compiled untyped files (see `collect_top_100_files.py` — samples up to 200 for reproducibility). |
| `Hundrad_renamed_benchmarks/` | Renamed / reorganized variant of the hundred-file subset (for controlled comparisons). |
| `archive_files/` | Archived copies / intermediate dumps (ad hoc). |

### Partial human-style typing

| Directory | Description |
|-----------|-------------|
| `partially_typed_files/` | Files with **some** annotations removed (input for “complete the typing” LLM runs). Built by `create_partially_typed_files.py` (uses JSON + AST stripping counts). |
| `partial_annoated_code_generation_by_LLM/` | Scripts + outputs: LLMs finish typing from `partially_typed_files/` (`*_partially_typed_files/` per model). |

### Full-file LLM annotation (main benchmark runs)

Outputs are usually `/<run_name>/<group_id>/<filename>.py`.

| Pattern / folder | Model / notes |
|------------------|----------------|
| `deep_seek/`, `deep_seek_2nd_run/` | DeepSeek chat runs (2nd run is the common “production” batch). |
| `claude3_sonnet_1st_run/`, `_2nd_run/`, `_3_run/`, `_4_run/` | Claude 3.7 Sonnet successive passes / ablations. |
| `gpt35_1st_run/`, `gpt35_2nd_run/`, `gpt35_renamed_output/` | GPT-3.5 family. |
| `gpt4o/`, `gpt4o_2nd_run/`, `gpt4o_llm_only_errors/` | GPT-4o + filtered error subsets. |
| `gpt5_1st_run/`, `gpt5_2_run/`, `gpt5_3_run/`, `gpt5_4_run/` | GPT-5 successive runs. |
| `o1_mini/`, `o1_mini_2nd_run/`, `o1_mini_renamed_output/`, `o1_mini_llm_only_errors/` | o1-mini. |
| `o3_mini_1st_run/`, `o3_mini_2nd_run/`, `o3_mini_3rd_run/`, `o3_mini_renamed_output/` | o3-mini. |
| `llama3_1_8B_1st_run/`, `llama3_1_8B_2nd_run/` | Local / Llama-class runs (nested layout). |
| `*_renamed_output/`, `*_renamed_output_2/` | Outputs after **rename** normalization (pairs with `rename_functions.py` / analysis). |
| `*_stub_run/` (`deepseek_1_stub_run`, `gpt5_1_stub_run`, …) | Smaller smoke / stub experiments. |
| `gpt5_500_sample_runs/` | Drivers like `generate_*_sample_run.py` for **500-file** style samples (DeepSeek / Claude / GPT-5). |
| `deep_seek_llm_only_errors/` etc. | Subsets where mypy failures are **typing-only** / LLM-focused (fed by `Find_llm_errors_files.py` + JSON under `mypy_results/Filtered_type_errors/`). |

### User-annotated condition (“human” full annotation)

| Directory | Description |
|-----------|-------------|
| `deepseek_user_annotated/`, `claude3_sonnet_user_annotated/`, `o3_mini_user_annotated/` | Human user completes / corrects annotations (paired with `Generate_code_using_LLM_user_annotated/` scripts). |

### Static / tool baselines

| Directory | Description |
|-----------|-------------|
| `hityper_outputs/`, `HiTyper_json_files/` | HiTyper CLI inference over `untyped_benchmarks/` (`Generate_type_using_HiTyper.py`); JSON type dumps for merging. |
| `HiTyper_1st_run/` | Packaged HiTyper-oriented run + `Run_raw_mypy.py` variant for that tree. |
| `MonkeyType/`, `Type_writer_results/` | MonkeyType / TypeWriter-related experiments and outputs. |

### Mypy results and repairs

| Directory | Description |
|-----------|-------------|
| `mypy_results/` | Large JSON corpora + analysis scripts (tables, Venn, filtered errors) — **see dedicated section *mypy_results* below.** |
| `GPCE_mypy_results/` | Mypy JSON for GPCE-style extra runs (e.g. `deepseek_3_run`, `deepseek_4_run` as wired in `Run_mypy_on_llm_2.py`). |
| `deepseek_mypy_fix/`, `gpt5_mypy_fix/`, `claude_mypy_fix/` | **LLM repair loops**: strict typing-only prompts, `fixed_files/` / `fixed_files_strict_edit_rules/`, logs. |
| `mypy_fix_analysis/` | Qualitative / diff analysis of fix behavior (example before/after snippets). |
| `reprocess_unprocessed/` | Regenerated outputs for files that failed or were skipped in earlier passes. |

### Human annotation percentage study

| Directory | Description |
|-----------|-------------|
| `Human_annotations_detailed_analysis/` | `generate_percentage_variations.py` builds `ten_percent_typed/` … `ninety_percent_typed/`; `Generate_code_using_*_percent.py` runs LLMs per level; `Run_mypy_on_llm_percent.py` + `analyze_mypy_results.py` aggregate. |

### Structure and alternative tasks

| Directory | Description |
|-----------|-------------|
| `Full_repo_analysis/` | Larger-context / full-repo annotation (`Generate_code_using_Claude.py` etc.) with `untyped_version_large`-style inputs. |
| `LLM-WT/` | Organized mirrors of model outputs (by model: `deepseek/`, `claude/`, `gpt4o/`, `o1mini/`, `o3mini/`); use with `collect_llm_wt_files.py` and `mypy_results/Section_04` JSONs. |
| `callgraph_analysis/` | `generate_callgraph.py` + comparisons to mypy outcomes (`callgraph_vs_mypy.py`). |
| `complexity_of_source_codes/` | Cyclomatic / file complexity vs mypy success. |
| `codesimilarty/`, `GPCE_AST_analysis/` | Similarity and AST-focused side studies. |
| `Type_info_collector/` | Type-info JSON metrics, precision, **Any**-rate, semantic comparison, plots — **see dedicated section *Type_info_collector* below.** |

---

## `mypy_results/` (scripts and layout)

**Role:** Stores **mypy batch JSON** (per file: `isCompiled`, `errors`, `stats` with parameter counts) and **downstream analysis scripts** for paper sections (tables, Venn diagrams, bar plots, filtered error sets).

### Main subfolders (data + layout)

| Subfolder | Contents / purpose |
|-----------|-------------------|
| `mypy_outputs/` | Large JSON files produced by `Run_mypy_on_llm*.py` — e.g. per-model `mypy_results_*_with_errors.json`, partial / user-annotated variants. |
| `Filtered_type_errors/` | **Merged** JSON per model (`merged_*.json`): files where errors are treated as **type-focused** after comparing with analysis filters; **`analyze_filtered_mypy_results.py`** summarizes them. |
| `Section_04/` | Core **compilation / consistency / Table 1–2** style analysis (see script list below). Also holds JSONs such as `both_success_files.json`, `llm_only_failure_files.json` used by repair / collection scripts. |
| `Section_5_LLM_VS_LLM/` | **LLM vs LLM:** Venn diagrams (`Ven_diagram*.py`), bar plots by annotation ratio or total parameters. |
| `Section_6_Human_VS_LLM/` | **Human (user-annotated / partial) vs LLM:** bar plots, `Table_8_truth_tables_llm_compilation.py`, human vs top-3 LLM Venn. |
| `Section_07/`, `Section_08/` | Extra section-specific plots / renamed-file analysis (`Section_08/Renamed_file_analysis.py`). |
| `split_original_files/` | Splits / overlap of **original** mypy corpora; **`analyze_model_overlap.py`** compares which files compile across models. |
| `type_coverage_bins/`, `benchmarks_annotations/`, `analysis_outputs/` | Binned or derived coverage / annotation artifacts (pairs with `generate_coverage_bins.py` where applicable). |
| `old_type_coverage_files/` | Legacy **`TypeCoveragePlot*.py`** scripts for older coverage plots. |
| `Deep_analysis_of_mypy/Further_analysis/` | Ad hoc deep dives (sample modules, import checks, etc.). |
| `archived/` | Older table scripts (e.g. `Table_1.py`). |

### Top-level scripts (`mypy_results/*.py`)

| Script | What it does |
|--------|----------------|
| **`Filter_only_type_erros_files.py`** | For each model, intersects **analysis JSON** (`analysis_*.json`) with full mypy output vs **untyped** baseline; writes **`Filtered_type_errors/merged_*.json`** (subset used for “LLM-only” / typing-centric error studies and `Find_llm_errors_files.py`). |
| **`analyze_model_overlap_new.py`** | **Overlap** of which files compile / fail across models (newer entry point; see also `split_original_files/analyze_model_overlap.py`). |
| **`BarPlot.py`**, **`create_stacked_bar_chart.py`** | Generic bar / stacked charts from aggregated mypy stats. |
| **`generate_coverage_bins.py`** | Builds **coverage bins** (feeds `type_coverage_bins/` or related summaries). |
| **`Merge_code_similarity_and_mypy_results.py`** | Joins **code similarity** features with mypy outcomes for joint analysis. |
| **`param_distribution_analysis.py`** | Distribution of **parameter counts** / annotations vs mypy success. |
| **`Renamed_file_analysis.py`** | Compares mypy results for **renamed** vs original file corpora. |

### `mypy_results/Section_04/` (paper-style breakdown)

| Script | What it does |
|--------|----------------|
| **`split_parameter_annotations.py`** | Splits `mypy_results_original_files_with_errors.json`-style data into files with **no param annotations** vs **some** (uses `stats.total_parameters` / `parameters_with_annotations`). |
| **`Table_1_analysis.py`**, **`Table_1_simplified.py`**, **`Table_1_split_analysis.py`**, **`Table_1_parameter_count_analysis.py`**, **`Table_1_analysis_filter_both_fail_first.py`** | Variants of **Table 1** aggregates (compilation rates, filters, parameter strata). |
| **`Table_2_consistency_analysis.py`** | **Consistency** across runs / conditions for Table 2. |
| **`flip_results.py`**, **`flip_summary.py`** | Files that **flip** compile ↔ fail between two conditions; summaries. |
| **`compilation_consistency_plot.py`**, **`compilation_consistency_plot_ccn.py`** | Plots of compilation consistency (second adds **CCN** / complexity linkage). |

### `mypy_results/Section_5_LLM_VS_LLM/` and `Section_6_Human_VS_LLM/`

- **Section 5:** **`Ven_diagram*.py`** (overlap of success sets across LLMs), **`barplot_compiled_by_annotation_ratio.py`**, **`barplot_compiled_by_total_parameters.py`**.
- **Section 6:** Same style bar plots plus **partial / user-annotated** variants (`barplot_for_partial_user_annotated.py`, `barplot_for_user_annotated_total_parameter.py`), **`Ven_diagram_human_vs_top3LLM.py`**, **`Table_8_truth_tables_llm_compilation.py`**.

### `mypy_results/Filtered_type_errors/`

- **`analyze_filtered_mypy_results.py`** — Reads merged filtered JSONs and prints / saves summary stats for **typing-filtered** error corpora.

---

## `Type_info_collector/` (scripts and layout)

**Role:** Consumes **type-info JSON** (from repo-root-style `Type_info_collector.py` runs or `Type_info_LLMS/*.json`) to compute **`Any` usage**, **precision vs ground truth**, **agreement** between LLMs, **semantic** type equivalence, and **memorization**-style analyses. Outputs are usually new JSON/CSV/plots beside the script or in named subfolders.

### Root-level scripts

| Script | What it does |
|--------|----------------|
| **`semantic_type_comparison.py`** | Maps “raw” type-string pairs through a **safe** typing-aware comparison; writes enriched JSON under **`semantic_comparison_results/`** (`type_comparison_semantic_*.json`). |
| **`Compare_LLM_vs_LLM.py`**, **`Comapre_LLM_VS_Human.py`** | Cross-model or **LLM vs human** type-info comparisons (aggregates / reports — exact metrics in-file). |
| **`parameter_vs_any_analysis.py`** | Relates **annotation coverage** to **`Any`** usage. |
| **`hypothesis_testing_fixed.py`** | Statistical tests on collected metrics (e.g. differences in Any-rate / precision). |
| **`visualize_any_usage.py`**, **`visualization_analysis.py`**, **`simple_rectangle_plot.py`** | Plotting helpers for Any / metric summaries. |
| **`examine_json.py`**, **`debug_semantic.py`**, **`test_subtype.py`** | Inspection / debugging utilities for JSON layout and subtyping checks. |

### `Type_info_collector/Section_04/`

| Script | What it does |
|--------|----------------|
| **`calculate_any_rate.py`**, **`calculate_empty_rate.py`**, **`calculate_instability_rate.py`** | **Any-ratio**, empty annotation rates, **instability** across two runs. |
| **`File_level_any_analysis.py`**, **`File_level_precision_analysis.py`** | Per-file **Any** and **precision** summaries. |
| **`Precision_in_two_runs.py`**, **`Table_4_precsion_in_two_run.py`**, **`Table_4_precsion_in_two_run_file_level.py`** | **Precision** when comparing two annotation runs (file vs corpus level). |
| **`simple_any_ratio_comparison.py`**, **`Table_03_any_empty_rate.py`**, **`Table_03_two_parts.py`**, **`Table_03_partially_typed_user_annotated_any_rate.py`** | **Table 3**-style **Any** / empty-rate tables, including **partial** and **user-annotated** splits. |
| **`analyze_type_coverage.py`**, **`analyze_type_replacement.py`**, **`analyze_three_models_union.py`** | Coverage / replacement patterns; **three-model** unions. |

### `Type_info_collector/Section_06/`

| Script | What it does |
|--------|----------------|
| **`any_ratio_analysis.py`**, **`any_ratio_analysis_plot_partial_type.py`**, **`any_ratio_analysis_user_annotation.py`** | **Any** ratio trends with plots (partial / user annotation conditions). |
| **`analyze_any_simple.py`**, **`analyze_any_final.py`**, **`analyze_any_ratio_changes.py`**, **`Common_Any_analysis.py`** | Iterations of **Any** deep dives and aggregates. |
| **`analyze_type_replacement.py`** | Which types were replaced by `Any` / others between conditions. |
| **`llm_agreement_analysis.py`**, **`precision_agreement_analysis.py`** | **Pairwise LLM agreement** on types; **precision agreement** across annotators/models. |
| **`debug_test.py`**, **`simple_test.py`**, **`test_file_loading.py`** | Tests / sanity checks. |

### `Type_info_collector/Sections_05/`

| Script | What it does |
|--------|----------------|
| **`Section_5_precision.py`**, **`Section_5_precision_plot.py`**, **`Section_5_precision_plot_binary.py`** | Core **Section 5 precision** numbers and plots. |
| **`Section5_LLM_precision_comparison.py`**, **`Section5_LLM_precision_comparison_with_plot.py`**, **`Section5_binary_precision_comparison.py`**, **`Section5_precision_plot.py`**, **`Section5_precision_plot_binary.py`** | **LLM vs LLM** precision comparisons (with / without plots); binary variants. |
| **`winner_group_counts.py`** | Counts which model “wins” per file/group under chosen precision rules. |

### `Type_info_collector/Section_08_LLM_memorized/`

**Theme:** Whether LLM outputs **mirror training / prior types** (memorization) vs **generic** inference.

| Script | What it does |
|--------|----------------|
| **`calculate_any_rate.py`**, **`any_ratio_analysis.py`** | Any metrics specialized to **memorization** corpora. |
| **`semantic_type_analysis.py`**, **`analyze_type_replacement.py`** | Semantic / replacement analysis for memorization setting. |
| **`precision_agreement_analysis.py`** | Agreement on “memorized” vs human / original types. |
| **`Compare_LLM_file_level_meomorization.py`** | File-level **memorization** comparison (filename reflects typo “meomorization”). |

### `Type_info_collector/Any_Anaysis/` (note: folder spelling)

| Script | What it does |
|--------|----------------|
| **`any_type_analysis.py`**, **`any_ratio_analysis.py`**, **`any_analysis_summary.py`**, **`analyze_type_patterns.py`** | **`Any`** usage patterns and summaries. |
| **`P_test_for_Any_type.py`** | Significance tests related to **`Any`** rates. |

### `Type_info_collector/llm_vs_llm_comparison/` and `semantic_comparison_results/`

| Script | What it does |
|--------|----------------|
| **`llm_vs_llm_comparison/analyze_semantic_results.py`** | Summarizes **semantic** LLM–LLM comparison outputs. |
| **`semantic_comparison_results/analyze_semantic_results.py`**, **`extract_optional_differences.py`** | Drill into **`Optional`** / union differences after semantic normalization. |

---

## Root-level driver scripts (`ManyTypes4py_benchmarks/*.py`)

Brief index; open each file for exact paths and flags.

### Corpus and renaming

- **`Generate_no_type_version.py`** — Untyped corpus + `grouped_file_paths.json` + `File_analysis_result.json`.
- **`Generate_original_files.py`** — Typed originals with hash-aligned names under `original_files/`.
- **`copy_and_create_untyped.py`**, **`output_no_types.py`**, **`Remove_Type_annotation.py`** — Alternate or legacy stripping / copy utilities.
- **`rename_functions.py`** — Rename functions for controlled comparisons (`*_renamed_output` pipelines).

### LLM full-file annotation (API batch)

- **`Generate_code_using_deepseek.py`** — `grouped_file_paths.json` → `deep_seek_2nd_run/`; parses ` ```python ` fences; timings + processed log.
- **`Generate_code_using_claude3_sonnet.py`**, **`Generate_code_using_OPENAI_gpt5.py`**, **`Generate_code_using_o3_mini.py`**, **`Generate_code_using_o1_mini.py`**, **`Generate_code_using_OPENAI_gpt35.py`**, **`Generate_code_using_OPENAI_gpt4o.py`**, **`Generate_code_using_OPENAI.py`**, **`Generate_code_using_llama32.py`** — Same structural pattern; **check `OUTPUT_DIR` / `PROCESSED_FILES_LOG` / model id** inside each file.
- **`Generate_code_using_*_renamed.py`** — Same as above but wired to **renamed** corpus paths.

### Partial typing and sampling

- **`create_partially_typed_files.py`** — Builds `partially_typed_files/` from originals + JSON configuration.
- **`partial_annoated_code_generation_by_LLM/Generate_code_using_*_renamed.py`** — LLM completes types from partial inputs.

### Mypy runners

- **`Run_mypy_on_llm.py`** — Recursive mypy over a tree; writes one JSON (older / simpler driver).
- **`Run_mypy_on_llm_2.py`** — Same idea + **parameter annotation stats** + `--python-version=3.10`; **`__main__`** block chooses **input dirs** and **output JSON** (edit there for new runs).
- **`Run_raw_mypy.py`** — Standalone mypy batch helper (see also `HiTyper_1st_run/Run_raw_mypy.py`).

### HiTyper / merge

- **`Generate_type_using_HiTyper.py`** — Runs `hityper infer` on every file under `untyped_benchmarks/` into `hityper_outputs/`.
- **`merge_hityper_types.py`** — Merges HiTyper JSON predictions into code / unified type views (see file for merge rules).

### Type info extraction (file-level JSON)

- **`Type_info_collector.py`** — Walks AST (with optional syntax recovery), emits per-function arg/return type strings for downstream **Any** and precision metrics.

### Utilities and side studies

- **`collect_top_100_files.py`** — Samples compiled untyped files into `Hundrad_original_typed_benchmarks/`.
- **`Find_llm_errors_files.py`** — Copies files for **LLM-only** error sets into `*_llm_only_errors/` from merged JSON.
- **`collect_llm_wt_files.py`** — Copies files listed in `mypy_results/Section_04` JSONs from model run trees into a consolidated layout for **LLM-WT** analysis.
- **`Parameter_count_group.py`**, **`generate_mapping.py`**, **`Generate_mypy_results_summary.py`**, **`benchmark_summary.py`** — Grouping, mapping, and summary tables.
- **`Generate_Table1.py`**, **`Table_3_generate.py`** — Paper-style table generation.
- **`Code_similarity.py`**, **`Code_similarity_feature_extract.py`**, **`complexity_analyzer_using_LIZARD.py`** — Similarity and complexity features.
- **`AddType_Typewriter_2.py`**, **`AddType_Typewriter_original.py`** — TypeWriter-style search / application experiments (AST + mypy feedback).
- **`Analyze_type_annotation.py`** — Ad hoc annotation analysis.
- **`Check_anthrpic_models.py`**, **`Check_model_load.py`**, **`OpenAIAPI.py`** — API / model sanity checks.

### Subfolders with their own drivers

- **`Generate_code_using_LLM_user_annotated/`** — Per-model scripts for the **user-annotated** condition.
- **`Human_annotations_detailed_analysis/`** — Percent-of-annotations sweep + LLM + mypy.
- **`Full_repo_analysis/`** — Large-input full-repo Claude / DeepSeek / o3-mini runs.
- **`gpt5_500_sample_runs/`** — Sample-run generators for scaling experiments.
- **`callgraph_analysis/`**, **`complexity_of_source_codes/`**, **`Type_info_collector/`**, **`mypy_fix_analysis/`** — Analysis-only code.

---

## Repo-root analysis (outside this folder)

- **`../llm_analysis.py`** — Aggregates **original / partial / user-annotated** mypy + Any + type-info JSON for named LLMs; CLI comparison and LaTeX output.

---

## Table/Figure script mapping

- **Table 2:** `mypy_results/Section_04/Table_1_analysis_filter_both_fail_first.py`
- **Table 3:** `mypy_results/Section_04/Table_2_consistency_analysis.py`
- **Table 4:** `Type_info_collector/Section_04/Table_03_any_empty_rate.py`
- **Table 5:** `Type_info_collector/Section_04/Table_4_precsion_in_two_run_file_level.py`, `Type_info_collector/Section_04/Table_4_precsion_in_two_run.py`
- **Table 7:** `Type_info_collector/Section_06/analyze_type_replacement.py`
- **Table 8:** `Type_info_collector/Section_06/precision_agreement_analysis.py`
- **Table 9:** `Type_info_collector/Section_06/analyze_any_ratio_changes.py`
- **Figure 5:** `mypy_results/Section_6_Human_VS_LLM/Ven_diagram_human_vs_top3LLM.py`
- **Figure 6:** `mypy_results/Section_6_Human_VS_LLM/barplot_compiled_by_total_parameters.py`
- **Figure 7:** `Type_info_collector/Section_06/any_ratio_analysis.py`
- **Figure 8:** `mypy_results/Section_6_Human_VS_LLM/barplot_for_partial_user_annotated.py`, `mypy_results/Section_6_Human_VS_LLM/barplot_for_user_annotated_total_parameter.py`
- **Figure 9:** `Type_info_collector/Section_06/any_ratio_analysis_user_annotation.py`, `Type_info_collector/Section_06/any_ratio_analysis_plot_partial_type.py`

---

## Cloned benchmark projects

Top-level folders such as `pandas/`, `aiohttp/`, `black/`, … are **upstream repositories** used as source locations for real-world files. They are **not** experiment outputs; do not list them as “runs” unless you add a note here for a specific paper snapshot.

---

## Maintenance checklist (when you add something new)

- [ ] Add a row under **Artifact directories** (or extend an existing table).
- [ ] If you add a new top-level `Generate_code_using_*.py`, add one line under **LLM full-file annotation**.
- [ ] If JSON locations change, update **Shared inputs** and any script that hardcodes paths.
- [ ] Regenerate or note new **`mypy_results/`** or **`Type_info_collector/`** consumers in **`llm_analysis.py`** if the paper pipeline depends on them.
- [ ] Add any new **`mypy_results/**/*.py`** or **`Type_info_collector/**/*.py`** to the matching section above (or one-line “see file docstring” note).
