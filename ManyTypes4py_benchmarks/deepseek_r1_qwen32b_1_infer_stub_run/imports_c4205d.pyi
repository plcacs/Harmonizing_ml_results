"""
Stub file for 'imports_c4205d' module.
"""

import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Iterable,
    Iterator,
    Any,
    Optional,
    Type,
    TypeVar,
    overload,
    TYPE_CHECKING,
)
from parso.python.tree import Name as TreeName
from parso.tree import Node as TreeNode
from jedi import debug
from jedi import settings
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.value.namespace import ImplicitNamespaceValue, ModuleValue
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FolderIO
from jedi.inference.cache import InferenceState

if TYPE_CHECKING:
    from jedi.api import Context

class ModuleCache:
    _name_cache: Dict[Tuple[str, ...], ValueSet]

    def __init__(self) -> None:
        ...

    def add(self, string_names: Tuple[str, ...], value_set: ValueSet) -> None:
        ...

    def get(self, string_names: Tuple[str, ...]) -> Optional[ValueSet]:
        ...

@inference_state_method_cache(default=NO_VALUES)
def infer_import(context: Context, tree_name: TreeName) -> ValueSet:
    ...

@inference_state_method_cache(default=Set[str])
def goto_import(context: Context, tree_name: TreeName) -> Set[str]:
    ...

def _prepare_infer_import(module_context: Context, tree_name: TreeName) -> Tuple[Optional[str], Tuple[str, ...], int, ValueSet]:
    ...

class Importer:
    _inference_state: InferenceState
    _module_context: Context
    level: int
    import_path: Tuple[Union[str, TreeName], ...]
    _fixed_sys_path: Optional[List[str]]
    _infer_possible: bool

    def __init__(self, inference_state: InferenceState, import_path: Tuple[Union[str, TreeName], ...], module_context: Context, level: int = 0) -> None:
        ...

    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        ...

    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        ...

    def follow(self) -> ValueSet:
        ...

    def completion_names(self, inference_state: InferenceState, only_modules: bool = False) -> List[Union[ImportName, SubModuleName]]:
        ...

    def _get_module_names(self, search_path: Optional[List[str]] = None, in_module: Optional[bool] = None) -> List[Union[ImportName, SubModuleName]]:
        ...

def import_module_by_names(inference_state: InferenceState, import_names: Tuple[Union[str, TreeName], ...], sys_path: Optional[List[str]] = None, module_context: Optional[Context] = None, prefer_stubs: bool = True) -> ValueSet:
    ...

@plugin_manager.decorate()
@import_module_decorator
def import_module(inference_state: InferenceState, import_names: Tuple[str, ...], parent_module_value: Optional[Any], sys_path: List[str]) -> ValueSet:
    ...

def _load_python_module(inference_state: InferenceState, file_io: FolderIO, import_names: Optional[Tuple[str, ...]] = None, is_package: bool = False) -> ModuleValue:
    ...

def _load_builtin_module(inference_state: InferenceState, import_names: Optional[Tuple[str, ...]] = None, sys_path: Optional[List[str]] = None) -> Optional[ModuleValue]:
    ...

def load_module_from_path(inference_state: InferenceState, file_io: FolderIO, import_names: Optional[Tuple[str, ...]] = None, is_package: Optional[bool] = None) -> Optional[ModuleValue]:
    ...

def load_namespace_from_path(inference_state: InferenceState, folder_io: FolderIO) -> ImplicitNamespaceValue:
    ...

def follow_error_node_imports_if_possible(context: Context, name: TreeName) -> Optional[ValueSet]:
    ...

def iter_module_names(inference_state: InferenceState, module_context: Context, search_path: List[str], module_cls: Type[Union[ImportName, SubModuleName]] = ImportName, add_builtin_modules: bool = True) -> Iterator[Union[ImportName, SubModuleName]]:
    ...