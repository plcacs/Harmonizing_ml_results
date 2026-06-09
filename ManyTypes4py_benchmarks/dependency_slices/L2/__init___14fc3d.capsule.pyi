from typing import Any

# === Third-party dependency: jedi ===
# Used symbols: cache, debug, settings

# === Third-party dependency: jedi.api ===
# Used symbols: classes, helpers, interpreter, refactoring

# === Third-party dependency: jedi.api.completion ===
class Completion:
    def __init__(self, inference_state, module_context, code_lines, position, signatures_callback, fuzzy = ...) -> Any: ...
    def complete(self) -> Any: ...
def search_in_module(inference_state, module_context, names, wanted_names, wanted_type, complete = ..., fuzzy = ..., ignore_imports = ..., convert = ...) -> Any: ...

# === Third-party dependency: jedi.api.environment ===
class InterpreterEnvironment(_SameEnvironmentMixin, _BaseEnvironment):
    ...

# === Third-party dependency: jedi.api.errors ===
def parso_to_jedi_errors(grammar, module_node) -> Any: ...

# === Third-party dependency: jedi.api.helpers ===
def validate_line_column(func) -> Any: ...

# === Third-party dependency: jedi.api.keywords ===
class KeywordName(AbstractArbitraryName):
    ...

# === Third-party dependency: jedi.api.project ===
class Project:
    def __init__(self, path, *, environment_path = ..., load_unsafe_extensions = ..., sys_path = ..., added_sys_path = ..., smart_sys_path = ...) -> None: ...
def get_default_project(path = ...) -> Any: ...

# === Third-party dependency: jedi.file_io ===
class KnownContentFileIO(file_io.KnownContentFileIO, FileIOFolderMixin):
    ...

# === Third-party dependency: jedi.inference ===
class InferenceState:
    def __init__(self, project, environment = ..., script_path = ...) -> Any: ...
# re-export: from jedi.inference import imports

# === Third-party dependency: jedi.inference.arguments ===
def try_iter_content(types, depth = ...) -> Any: ...

# === Third-party dependency: jedi.inference.base_value ===
class ValueSet: ...

# === Third-party dependency: jedi.inference.gradual.conversion ===
def convert_names(names, only_stubs = ..., prefer_stubs = ..., prefer_stub_to_compiled = ...) -> Any: ...
def convert_values(values, only_stubs = ..., prefer_stubs = ..., ignore_compiled = ...) -> Any: ...

# === Third-party dependency: jedi.inference.gradual.utils ===
def load_proper_stub_module(inference_state, grammar, file_io, import_names, module_node) -> Any: ...

# === Third-party dependency: jedi.inference.helpers ===
def infer_call_of_leaf(context, leaf, cut_own_trailer = ...) -> Any: ...

# === Third-party dependency: jedi.inference.references ===
def find_references(module_context, tree_name, only_in_module = ...) -> Any: ...

# === Third-party dependency: jedi.inference.syntax_tree ===
def tree_name_to_values(inference_state, context, tree_name) -> Any: ...

# === Third-party dependency: jedi.inference.sys_path ===
def transform_path_to_dotted(sys_path, module_path) -> Any: ...

# === Third-party dependency: jedi.inference.utils ===
def to_list(func) -> Any: ...

# === Third-party dependency: jedi.inference.value ===
# Used symbols: ModuleValue

# === Third-party dependency: jedi.inference.value.iterable ===
def unpack_tuple_to_dict(context, types, exprlist) -> Any: ...

# === Third-party dependency: jedi.parser_utils ===
def get_executable_nodes(node, last_added = ...) -> Any: ...

# === Third-party dependency: parso ===
# Used symbols: split_lines

# === Third-party dependency: parso.python ===
# Used symbols: tree