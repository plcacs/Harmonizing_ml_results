from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from jedi.file_io import FolderIO
from jedi.inference.base_value import ValueSet
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.value.namespace import ImplicitNamespaceValue

class ModuleCache:
    _name_cache: dict[tuple[str, ...] | None, ValueSet]
    def __init__(self) -> None: ...
    def add(self, string_names: tuple[str, ...] | None, value_set: ValueSet) -> None: ...
    def get(self, string_names: tuple[str, ...]) -> ValueSet | None: ...

def infer_import(context: Any, tree_name: Any) -> ValueSet: ...
def goto_import(context: Any, tree_name: Any) -> list[Any] | set[Any]: ...
def _prepare_infer_import(
    module_context: Any, tree_name: Any
) -> tuple[str | None, tuple[Any, ...], int, ValueSet]: ...
def _add_error(value: Any, name: Any, message: str) -> None: ...
def _level_to_base_import_path(
    project_path: str, directory: str, level: int
) -> tuple[list[str] | None, str | None]: ...

class Importer:
    level: int
    import_path: Any
    _inference_state: Any
    _module_context: Any
    _fixed_sys_path: list[str] | None
    _infer_possible: bool

    def __init__(
        self,
        inference_state: Any,
        import_path: Any,
        module_context: Any,
        level: int = ...,
    ) -> None: ...
    @property
    def _str_import_path(self) -> tuple[str, ...]: ...
    def _sys_path_with_modifications(self, is_completion: bool) -> list[str]: ...
    def follow(self) -> ValueSet: ...
    def _get_module_names(
        self,
        search_path: list[str] | None = ...,
        in_module: Any | None = ...,
    ) -> list[Any]: ...
    def completion_names(
        self, inference_state: Any, only_modules: bool = ...
    ) -> list[Any]: ...

def import_module_by_names(
    inference_state: Any,
    import_names: Any,
    sys_path: list[str] | None = ...,
    module_context: Any | None = ...,
    prefer_stubs: bool = ...,
) -> ValueSet: ...
def import_module(
    inference_state: Any,
    import_names: tuple[str, ...],
    parent_module_value: Any,
    sys_path: list[str],
) -> ValueSet: ...
def _load_python_module(
    inference_state: Any,
    file_io: Any,
    import_names: tuple[str, ...] | None = ...,
    is_package: bool = ...,
) -> Any: ...
def _load_builtin_module(
    inference_state: Any,
    import_names: tuple[str, ...] | None = ...,
    sys_path: list[str] | None = ...,
) -> Any | None: ...
def load_module_from_path(
    inference_state: Any,
    file_io: Any,
    import_names: tuple[str, ...] | None = ...,
    is_package: bool | None = ...,
) -> Any: ...
def load_namespace_from_path(
    inference_state: Any, folder_io: FolderIO
) -> ImplicitNamespaceValue: ...
def follow_error_node_imports_if_possible(
    context: Any, name: Any
) -> ValueSet | None: ...
def iter_module_names(
    inference_state: Any,
    module_context: Any,
    search_path: list[str],
    module_cls: type = ...,
    add_builtin_modules: bool = ...,
) -> Any: ...