# AST Comparison Report: GPT-5.4 vs Untyped Baseline

## Summary Statistics

- **Total files compared**: 499
- **Files with structural changes**: 8
- **Files with parse errors**: 3
- **Percentage with changes**: 1.60%

### Key Insights
- Out of 499 files, **8 (1.60%)** had structural differences
- **3** files could not be parsed (syntax errors in GPT-5.4 output)
- Most changes involve **control flow modifications**, **class additions**, or **parameter changes**

---

## ⚠️ Parse Errors - GPT-5.4 Generated Invalid Syntax

These files could not be analyzed because the generated code has syntax errors:

### Folder 6

**conftest_641a68.py**

```
Error: expression cannot contain assignment, perhaps you meant "=="? (<unknown>, line 97)
```


### Folder 14

**test_validation_2c8d12.py**

```
Error: closing parenthesis ')' does not match opening parenthesis '{' (<unknown>, line 219)
```


### Folder 16

**cache_policies_2cc3d9.py**

```
Error: cannot assign to keyword argument unpacking (<unknown>, line 110)
```


---

## Folder 2 - 1 File(s) with Changes

### awsclient_860c67.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `update_function`
  - Original params: `self, function_name, zip_contents, environment_variables, runtime, tags, xray, timeout, memory_size, role_arn, subnet_ids, security_group_ids, layers`
  - Modified params: `self, function_name, zip_contents, environment_variables, runtime, tags, xray, timeout, memory_size, security_group_ids, subnet_ids, layers`

---

## Folder 6 - 2 File(s) with Changes

### manifest_47c52e.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `_map_nodes_to_map_resources`
  - Original params: `cls, nodes_map`
  - Modified params: `self, nodes_map`

### test_payments_dde789.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_api_payments_with_lock_timeout`
  - Original params: `api_server_test_instance, raiden_network, token_addresses, pfs_mock`
  - Modified params: `api_server_test_instance, raiden_network, token_addresses, pfs_mock, deposit`

---

## Folder 11 - 2 File(s) with Changes

### test_ipython_b46143.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_load_extension_not_in_kedro_project`
  - Original params: `self, mocker, caplog`
  - Modified params: `self, mocker, caplog, ipython`

### test_stack_unstack_b54311.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `test_stack_empty_frame`
  - Original params: `dropna, future_stack`
  - Modified params: `dropna, future_stack, int_frame`

---

## Folder 15 - 2 File(s) with Changes

### forms_6e49d8.py

**Changes detected**: method_removed_in_typed, method_added_in_typed

#### What Changed:

- ❌ **Method removed** from `LoggingSetPasswordForm`: `clean_new_password1`
- ✅ **Method added** to `LoggingSetPasswordForm`: `clean_new_password`

### test_init_75a7d5.py

**Changes detected**: parameter_mismatch

#### What Changed:

- ⚙️ **Function signature changed**: `async_describe_events`
  - Original params: `hass, async_describe_event`
  - Modified params: `self, hass, async_describe_event`

---

## Folder 16 - 1 File(s) with Changes

### test_sql_f4958a.py

**Changes detected**: control_flow_mismatch

#### What Changed:

- 🔄 **Control flow structure modified**:
  - `if_statements`: 146 → 147 **(+1 added)**
  - `with_statements`: 181 → 184 **(+3 added)**

---

