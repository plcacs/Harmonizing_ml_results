from pathlib import Path
from typing import Iterator, Optional, Sequence

from parso.python.tree import Name
from jedi.file_io import FileIO, FolderIO
from jedi.inference.base_value import ValueSet
from jedi.inference.context import ModuleContext
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.value import ModuleValue
from jedi.inference.value.namespace import ImplicitNamespaceValue
from jedi.inference.gradual.typeshed import StubModuleValue
from jedi.inference import InferenceState


class ModuleCache:
    _name_cache: dict[tuple[str, ...], ValueSet]

    def __init__(self) -> None: ...
    def add(self, string_names: tuple[str, ...] | None, value_set: ValueSet) -> None: ...
    def get(self, string_names: tuple[str, ...]) -> ValueSet | None: ...


def infer_import(context: object, tree_name: Name) -> ValueSet: ...

def goto_import(context: object, tree_name: Name) -> list[object] | set[object]: ...

def _prepare_infer_import(
    module_context: ModuleContext,
    tree_name: Name,
) -> tuple[str | None, tuple[str | Name, ...], int, ValueSet]: ...

def _add_error(value: object, name: object, message: str) -> None: ...

def _level_to_base_import_path(
    project_path: str,
    directory: str,
    level: int,
) -> tuple[list[str] | None, str | None]: ...


class Importer:
    level: int
    import_path: tuple[str | Name, ...] | list[str | Name]
    _inference_state: InferenceState
    _module_context: ModuleContext
    _fixed_sys_path: list[str] | None
    _infer_possible: bool

    def __init__(
        self,
        inference_state: InferenceState,
        import_path: tuple[str | Name, ...] | Sequence[str | Name],
        module_context: ModuleContext,
        level: int = ...,
    ) -> None: ...

    @property
    def _str_import_path(self) -> tuple[str, ...]: ...

    def _sys_path_with_modifications(self, is_completion: bool) -> list[str]: ...

    def follow(self) -> ValueSet: ...

    def _get_module_names(
        self,
        search_path: list[str] | None = ...,
        in_module: object | None = ...,
    ) -> list[ImportName | SubModuleName]: ...

    def completion_names(
        self,
        inference_state: InferenceState,
        only_modules: bool = ...,
    ) -> list[object]: ...


def import_module_by_names(
    inference_state: InferenceState,
    import_names: tuple[str | Name, ...] | Sequence[str | Name],
    sys_path: list[str] | None = ...,
    module_context: ModuleContext | None = ...,
    prefer_stubs: bool = ...,
) -> ValueSet: ...

def import_module(
    inference_state: InferenceState,
    import_names: tuple[str, ...],
    parent_module_value: object | None,
    sys_path: list[str],
) -> ValueSet: ...

def _load_python_module(
    inference_state: InferenceState,
    file_io: FileIO,
    import_names: tuple[str, ...] | None = ...,
    is_package: bool = ...,
) -> ModuleValue: ...

def _load_builtin_module(
    inference_state: InferenceState,
    import_names: tuple[str, ...] | None = ...,
    sys_path: list[str] | None = ...,
) -> object | None: ...

def load_module_from_path(
    inference_state: InferenceState,
    file_io: FileIO,
    import_names: tuple[str, ...] | None = ...,
    is_package: bool | None = ...,
) -> ModuleValue | StubModuleValue: ...

def load_namespace_from_path(
    inference_state: InferenceState,
    folder_io: FolderIO,
) -> ImplicitNamespaceValue: ...

def follow_error_node_imports_if_possible(
    context: object,
    name: Name,
) -> ValueSet | None: ...

def iter_module_names(
    inference_state: InferenceState,
    module_context: ModuleContext,
    search_path: list[str],
    module_cls: type = ...,
    add_builtin_modules: bool = ...,
) -> Iterator[ImportName | SubModuleName]: ...