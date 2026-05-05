# AST Comparison Report: GPT-5.2 vs Untyped Baseline

## Summary Statistics

- **Total files compared**: 499
- **Files with structural changes**: 35
- **Files with parse errors**: 6
- **Percentage with changes**: 7.01%

### Key Insights
- Out of 499 files, **35 (7.01%)** had structural differences
- **6** files could not be parsed (syntax errors in GPT-5.2 output)
- Most changes involve **control flow modifications**, **class additions**, or **parameter changes**

---

## ⚠️ Parse Errors - GPT-5.2 Generated Invalid Syntax

These files could not be analyzed because the generated code has syntax errors:

### Folder 6

**test_base_fc0d64.py**

```
Error: invalid syntax (<unknown>, line 108)
```


### Folder 8

**web_rtc_831f2f.py**

```
Error: invalid syntax (<unknown>, line 33)
```


### Folder 9

**test_rocksdb_e6756b.py**

```
Error: invalid syntax (<unknown>, line 91)
```


### Folder 12

**__init___4ea406.py**

```
Error: invalid syntax (<unknown>, line 367)
```


### Folder 14

**test_interval_range_6fec67.py**

```
Error: unterminated string literal (detected at line 199) (<unknown>, line 199)
```


### Folder 16

**entryexitanalysis_7f8dee.py**

```
Error: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-1: malformed \N character escape (<unknown>, line 226)
```


---

## Folder 1 - 3 File(s) with Changes

### alarm_control_panel_e80274.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 31 → 32 **(+1 added)**

### monitor_7b218d.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `WebRequestState`
- ✅ **Class added**: `RebalanceState`
- ✅ **Class added**: `EventState`
- ✅ **Class added**: `AssignmentState`

### test_value_counts_77b3d1.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `rebuild_index`
  - Original params: `df`
  - Modified params: `s`

---

## Folder 2 - 3 File(s) with Changes

### __init___d345bd.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 22 → 23 **(+1 added)**

### _models_dd672f.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 77 → 78 **(+1 added)**

### test_transition_77bcb3.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `BlockLike`
- ✅ **Class added**: `BodyLike`
- ✅ **Class added**: `MessageLike`

---

## Folder 3 - 3 File(s) with Changes

### exposed_entities_a550ba.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `ExposedEntityJSON`
- ✅ **Class added**: `AssistantPreferencesJSON`

### imports_c4205d.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `import_module`
  - Original params: `inference_state, import_names, parent_module_value, sys_path`
  - Modified params: `inference_state, import_names, parent_module_value, sys_path, prefer_stubs`

### test_na_values_e137af.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `ParserLike`

---

## Folder 5 - 3 File(s) with Changes

### test_client_1103c5.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `mock_tls_insecure_set`
  - Original params: `insecure_param`
  - Modified params: `insecure_param_val`

### test_hyperopt_ac1e12.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `ResultMetrics`

### test_template_ceacd4.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_raise_exception_on_error`
  - Original params: `hass`
  - Modified params: ``

---

## Folder 6 - 2 File(s) with Changes

### base_c73d2b.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 20 → 21 **(+1 added)**
  - `try_except`: 2 → 3 **(+1 added)**

### manifest_47c52e.py

**Changes detected**: class_added_in_typed, parameter_mismatch

#### What Changed:

- ✅ **Class added**: `HasUniqueId`
- ⚙️ **Function signature changed**: `_map_nodes_to_map_resources`
  - Original params: `cls, nodes_map`
  - Modified params: `self, nodes_map`

---

## Folder 7 - 3 File(s) with Changes

### api_tests_b98962.py

**Changes detected**: method_added_in_typed

#### What Changed:

- ✅ **Method added** to `BaseTestChartDataApi`: `assert_row_count`

### prometheus_180961.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 1 → 2 **(+1 added)**

### test_usecols_basic_f18d3b.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `Parser`

---

## Folder 10 - 4 File(s) with Changes

### base_value_caa0cc.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 22 → 23 **(+1 added)**

### metrics_b562d7.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 9 → 11 **(+2 added)**

### superset_factory_util_7e174c.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `insert_model`
  - Original params: `dashboard`
  - Modified params: `model`

### test_cli_3a2b0d.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 0 → 1 **(+1 added)**

---

## Folder 11 - 1 File(s) with Changes

### test_range_fc6fe9.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_abs_returns_rangeindex`
  - Original params: `rng, exp_rng`
  - Modified params: `rng`

---

## Folder 12 - 4 File(s) with Changes

### datastructures_0a7b11.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 49 → 50 **(+1 added)**

### randomized_block_tests_8fb192.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `Scenario`
- ✅ **Class added**: `Transition`

### test_constructors_c6bb5c.py

**Changes detected**: class_missing_in_typed, class_added_in_typed

#### What Changed:

- ❌ **Class removed**: `List`
- ✅ **Class added**: `List_`

### test_frame_60c296.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 13 → 14 **(+1 added)**

---

## Folder 13 - 1 File(s) with Changes

### __init___e33fa3.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 15 → 16 **(+1 added)**

---

## Folder 14 - 3 File(s) with Changes

### mccabe_5703d2.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 20 → 23 **(+3 added)**

### test_blackouts_53e928.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `AlertLike`

### test_constraints_cdcb72.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `_normalize_whitespace`
  - Original params: `input`
  - Modified params: `input_str`

---

## Folder 15 - 2 File(s) with Changes

### specs_7c250e.py

**Changes detected**: class_added_in_typed

#### What Changed:

- ✅ **Class added**: `RunParams`

### test_init_75a7d5.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `async_describe_events`
  - Original params: `hass, async_describe_event`
  - Modified params: `self, hass, async_describe_event`

---

## Folder 16 - 2 File(s) with Changes

### evaluators_b911b8.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `calculate_dataframe_hash`
  - Original params: `df, eval_name`
  - Modified params: `df, eval_name_inner`

### test_sql_f4958a.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 146 → 145 **(-1 removed)**

---

## Folder 17 - 1 File(s) with Changes

### test_stata_db4b21.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_chunked_categorical`
  - Original params: `version, temp_file`
  - Modified params: `self, version, temp_file`

---

