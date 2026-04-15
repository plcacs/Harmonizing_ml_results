import os
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Union,
    Iterator,
    Any,
    Set,
    Sequence,
    Iterable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from parso.python import tree
    from parso.tree import BaseNode
    from jedi.inference.base_value import ValueSet, Value
    from jedi.inference.value import ModuleValue
    from jedi.inference.context import ModuleContext, Context
    from jedi.inference.names import AbstractName
    from jedi.file_io import FileIO, FolderIO
    from jedi.inference import InferenceState
    from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
    from jedi.inference.gradual.typeshed import StubModuleValue

class ModuleCache:
    _name_cache: dict

    def __init__(self) -> None: ...
    def add(self, string_names: Optional[Tuple[str, ...]], value_set: "ValueSet") -> None: ...
    def get(self, string_names: Tuple[str, ...]) -> Optional["ValueSet"]: ...

def infer_import(context: "Context", tree_name: "tree.Name") -> "ValueSet": ...
def goto_import(context: "Context", tree_name: "tree.Name") -> List["AbstractName"]: ...
def _prepare_infer_import(
    module_context: "ModuleContext", 
    tree_name: "tree.Name"
) -> Tuple[Optional[str], Tuple[str, ...], int, "ValueSet"]: ...
def _add_error(value: Optional["Value"], name: Any, message: str) -> None: ...
def _level_to_base_import_path(
    project_path: str, 
    directory: str, 
    level: int
) -> Tuple[Optional[List[str]], Optional[str]]: ...

class Importer:
    _inference_state: "InferenceState"
    level: int
    _module_context: "ModuleContext"
    _fixed_sys_path: Optional[List[str]]
    _infer_possible: bool
    import_path: Sequence[Union[str, "tree.Name"]]

    def __init__(
        self, 
        inference_state: "InferenceState", 
        import_path: Sequence[Union[str, "tree.Name"]], 
        module_context: "ModuleContext", 
        level: int = 0
    ) -> None: ...
    @property
    def _str_import_path(self) -> Tuple[str, ...]: ...
    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]: ...
    def follow(self) -> "ValueSet": ...
    def _get_module_names(
        self, 
        search_path: Optional[List[str]] = None, 
        in_module: Optional["Value"] = None
    ) -> List["AbstractName"]: ...
    def completion_names(
        self, 
        inference_state: "InferenceState", 
        only_modules: bool = False
    ) -> List["AbstractName"]: ...

def import_module_by_names(
    inference_state: "InferenceState",
    import_names: Sequence[Union[str, "tree.Name"]],
    sys_path: Optional[List[str]] = None,
    module_context: Optional["ModuleContext"] = None,
    prefer_stubs: bool = True
) -> "ValueSet": ...

def import_module(
    inference_state: "InferenceState",
    import_names: Tuple[str, ...],
    parent_module_value: Optional["Value"],
    sys_path: List[str]
) -> "ValueSet": ...

def _load_python_module(
    inference_state: "InferenceState",
    file_io: "FileIO",
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: bool = False
) -> "ModuleValue": ...

def _load_builtin_module(
    inference_state: "InferenceState",
    import_names: Optional[Tuple[str, ...]] = None,
    sys_path: Optional[List[str]] = None
) -> Optional["Value"]: ...

def load_module_from_path(
    inference_state: "InferenceState",
    file_io: "FileIO",
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: Optional[bool] = None
) -> Union["ModuleValue", "StubModuleValue"]: ...

def load_namespace_from_path(
    inference_state: "InferenceState",
    folder_io: "FolderIO"
) -> "Value": ...

def follow_error_node_imports_if_possible(
    context: "Context",
    name: "tree.Name"
) -> Optional["ValueSet"]: ...

def iter_module_names(
    inference_state: "InferenceState",
    module_context: "ModuleContext",
    search_path: List[str],
    module_cls: type = ...,
    add_builtin_modules: bool = True
) -> Iterator["AbstractName"]: ...