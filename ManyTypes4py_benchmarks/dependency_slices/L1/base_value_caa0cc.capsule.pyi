from typing import Any

# === Third-party dependency: jedi ===
# Used symbols: debug

# === Third-party dependency: jedi.cache ===
def memoize_method(method) -> Any: ...

# === Third-party dependency: jedi.inference ===
# Used symbols: analysis, finder

# === Third-party dependency: jedi.inference.arguments ===
class ValuesArguments(AbstractArguments):
    def __init__(self, values_list) -> Any: ...

# === Third-party dependency: jedi.inference.cache ===
def inference_state_as_method_param_cache() -> Any: ...

# === Third-party dependency: jedi.inference.compiled ===
# Used symbols: CompiledValueName

# === Third-party dependency: jedi.inference.gradual.annotation ===
def merge_type_var_dicts(base_dict, new_dict) -> Any: ...

# === Third-party dependency: jedi.inference.gradual.conversion ===
def convert_values(values, only_stubs = ..., prefer_stubs = ..., ignore_compiled = ...) -> Any: ...

# === Third-party dependency: jedi.inference.helpers ===
class SimpleGetItemNotFound(Exception): ...

# === Third-party dependency: jedi.inference.lazy_value ===
class LazyKnownValues(AbstractLazyValue): ...
def get_merged_lazy_value(lazy_values) -> Any: ...

# === Third-party dependency: jedi.inference.names ===
class ValueName(ValueNameMixin, AbstractTreeName): ...

# === Third-party dependency: jedi.inference.utils ===
def safe_property(func) -> Any: ...

# === Third-party dependency: jedi.parser_utils ===
def clean_scope_docstring(scope_node) -> Any: ...

# === Third-party dependency: parso.python.tree ===
class Name(_LeafWithoutNewlines): ...