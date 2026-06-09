from typing import Any

# === Third-party dependency: jedi ===
# Used symbols: debug, settings

# === Third-party dependency: jedi.file_io ===
class FolderIO(AbstractFolderIO):
    def get_file_io(self, name) -> Any: ...

# === Third-party dependency: jedi.inference ===
# Used symbols: analysis, compiled, helpers, sys_path

# === Third-party dependency: jedi.inference.base_value ===
class ValueSet:
    def __init__(self, iterable) -> Any: ...
NO_VALUES: ValueSet

# === Third-party dependency: jedi.inference.cache ===
def inference_state_method_cache(default = ...) -> Any: ...

# === Third-party dependency: jedi.inference.compiled.subprocess.functions ===
class ImplicitNSInfo: ...

# === Third-party dependency: jedi.inference.gradual.conversion ===
def convert_values(values, only_stubs = ..., prefer_stubs = ..., ignore_compiled = ...) -> Any: ...

# === Third-party dependency: jedi.inference.gradual.typeshed ===
def import_module_decorator(func) -> Any: ...
def parse_stub_module(inference_state, file_io) -> Any: ...
def create_stub_module(inference_state, grammar, python_value_set, stub_module_node, file_io, import_names) -> Any: ...

# === Third-party dependency: jedi.inference.names ===
class ImportName(AbstractNameDefinition): ...
class SubModuleName(ImportName): ...

# === Third-party dependency: jedi.inference.utils ===
def unite(iterable) -> Any: ...

# === Third-party dependency: jedi.inference.value ===
# Used symbols: ModuleValue

# === Third-party dependency: jedi.inference.value.namespace ===
class ImplicitNamespaceValue(Value, SubModuleDictMixin):
    def __init__(self, inference_state, string_names, paths) -> Any: ...

# === Third-party dependency: jedi.parser_utils ===
def get_cached_code_lines(grammar, path) -> Any: ...

# === Third-party dependency: jedi.plugins ===
plugin_manager: _PluginManager

# === Third-party dependency: parso.python ===
# Used symbols: tree

# === Third-party dependency: parso.tree ===
def search_ancestor(node: 'NodeOrLeaf', *node_types: str) -> 'Optional[BaseNode]': ...