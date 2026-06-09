from typing import Any

# === Third-party dependency: jedi ===
# Used symbols: debug

# === Third-party dependency: jedi.cache ===
def memoize_method(method) -> Any: ...

# === Third-party dependency: jedi.inference ===
# Used symbols: docstrings

# === Third-party dependency: jedi.inference.base_value ===
class Value(HelperValueMixin):
    def name(self) -> Any: ...
class ValueSet:
    def __init__(self, iterable) -> Any: ...
NO_VALUES: ValueSet

# === Third-party dependency: jedi.inference.cache ===
def inference_state_function_cache(default = ...) -> Any: ...

# === Third-party dependency: jedi.inference.compiled ===
def builtin_from_name(inference_state, string) -> Any: ...

# === Third-party dependency: jedi.inference.compiled.access ===
# Used symbols: _sentinel

# === Third-party dependency: jedi.inference.context ===
class CompiledContext(ValueContext): ...
class CompiledModuleContext(CompiledContext): ...

# === Third-party dependency: jedi.inference.filters ===
class AbstractFilter:
    def get(self, name) -> Any: ...
    def values(self) -> Any: ...

# === Third-party dependency: jedi.inference.helpers ===
def reraise_getitem_errors(*exception_classes) -> Any: ...

# === Third-party dependency: jedi.inference.lazy_value ===
class LazyKnownValue(AbstractLazyValue): ...

# === Third-party dependency: jedi.inference.names ===
class AbstractNameDefinition:
    def infer(self) -> Any: ...
    def api_type(self) -> Any: ...
class ValueNameMixin:
class ParamNameInterface(_ParamMixin):
    def star_count(self) -> Any: ...

# === Third-party dependency: jedi.inference.signature ===
class BuiltinSignature(AbstractSignature): ...

# === Third-party dependency: jedi.inference.utils ===
def to_list(func) -> Any: ...

# === Third-party dependency: jedi.inference.value ===
# Used symbols: CompiledInstance