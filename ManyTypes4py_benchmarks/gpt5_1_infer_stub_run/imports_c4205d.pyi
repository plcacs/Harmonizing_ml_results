from typing import Any, Iterator, Optional, Sequence, Tuple, Union, List, Set
from parso.python import tree
from jedi.inference.base_value import ValueSet
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.value import ModuleValue
from jedi.inference.value.namespace import ImplicitNamespaceValue

NameOrStr = Union[str, tree.Name]

class ModuleCache:
    _name_cache: dict[tuple[str, ...], ValueSet]

    def __init__(self) -> None: ...
    def add(self, string_names: Optional[tuple[str, ...]], value_set: ValueSet) -> None: ...
    def get(self, string_names: tuple[str, ...]) -> Optional[ValueSet]: ...

def infer_import(context: Any, tree_name: tree.Name) -> ValueSet: ...
def goto_import(context: Any, tree_name: tree.Name) -> Union[List[Any], Set[str]]: ...

def _prepare_infer_import(
    module_context: Any, tree_name: tree.Name
) -> tuple[Optional[NameOrStr], tuple[NameOrStr, ...], int, ValueSet]: ...
def _add_error(value: Any, name: Any, message: str) -> None: ...
def _level_to_base_import_path(
    project_path: str, directory: str, level: int
) -> tuple[Optional[list[str]], Optional[str]]: ...

class Importer:
    def __init__(
        self,
        inference_state: Any,
        import_path: Sequence[NameOrStr],
        module_context: Any,
        level: int = ...,
    ) -> None: ...

    @property
    def _str_import_path(self) -> tuple[str, ...]: ...
    def _sys_path_with_modifications(self, is_completion: bool) -> list[str]: ...
    def follow(self) -> ValueSet: ...
    def _get_module_names(
        self, search_path: Optional[Sequence[str]] = ..., in_module: Any = ...
    ) -> list[ImportName]: ...
    def completion_names(self, inference_state: Any, only_modules: bool = ...) -> list[Any]: ...

def import_module_by_names(
    inference_state: Any,
    import_names: Sequence[NameOrStr],
    sys_path: Optional[Sequence[str]] = ...,
    module_context: Optional[Any] = ...,
    prefer_stubs: bool = ...,
) -> ValueSet: ...

def import_module(
    inference_state: Any,
    import_names: tuple[str, ...],
    parent_module_value: Optional[Any],
    sys_path: Sequence[str],
    prefer_stubs: bool = ...,
) -> ValueSet: ...

def _load_python_module(
    inference_state: Any,
    file_io: Any,
    import_names: Optional[tuple[str, ...]] = ...,
    is_package: bool = ...,
) -> ModuleValue: ...

def _load_builtin_module(
    inference_state: Any,
    import_names: Optional[tuple[str, ...]] = ...,
    sys_path: Optional[Sequence[str]] = ...,
) -> Optional[Any]: ...

def load_module_from_path(
    inference_state: Any,
    file_io: Any,
    import_names: Optional[tuple[str, ...]] = ...,
    is_package: Optional[bool] = ...,
) -> Any: ...

def load_namespace_from_path(inference_state: Any, folder_io: Any) -> ImplicitNamespaceValue: ...

def follow_error_node_imports_if_possible(context: Any, name: tree.Name) -> Optional[ValueSet]: ...

def iter_module_names(
    inference_state: Any,
    module_context: Any,
    search_path: Sequence[str],
    module_cls: type[ImportName] = ...,
    add_builtin_modules: bool = ...,
) -> Iterator[ImportName]: ...