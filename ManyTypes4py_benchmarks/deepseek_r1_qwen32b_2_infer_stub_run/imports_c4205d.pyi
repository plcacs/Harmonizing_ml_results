"""
Stub file for jedi.inference.imports
"""

import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)
from jedi import Context
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.value.namespace import ImplicitNamespaceValue
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.inference.gradual.typeshed import StubModuleValue
from jedi.inference.utils import unite
from jedi.inference.cache import inference_state_method_cache

class ModuleCache:
    def __init__(self) -> None:
        ...
    
    def add(self, string_names: Union[str, Tuple[str, ...]], value_set: ValueSet) -> None:
        ...
    
    def get(self, string_names: Union[str, Tuple[str, ...]]) -> Optional[ValueSet]:
        ...

@inference_state_method_cache(default=NO_VALUES)
def infer_import(context: Context, tree_name: Any) -> ValueSet:
    ...

@inference_state_method_cache(default=[])
def goto_import(context: Context, tree_name: Any) -> Set[str]:
    ...

def _prepare_infer_import(module_context: Context, tree_name: Any) -> Tuple[Optional[str], Tuple[str, ...], int, ValueSet]:
    ...

def _add_error(value: Any, name: Any, message: str) -> None:
    ...

def _level_to_base_import_path(project_path: str, directory: str, level: int) -> Tuple[Optional[List[str]], Optional[str]]:
    ...

class Importer:
    def __init__(self, inference_state: Any, import_path: Tuple[str, ...], module_context: Context, level: int = 0) -> None:
        ...
    
    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        ...
    
    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        ...
    
    def follow(self) -> ValueSet:
        ...
    
    def _get_module_names(self, search_path: Optional[List[str]] = None, in_module: Any = None) -> List[Union[ImportName, SubModuleName]]:
        ...
    
    def completion_names(self, inference_state: Any, only_modules: bool = False) -> List[Union[ImportName, SubModuleName]]:
        ...

def import_module_by_names(inference_state: Any, import_names: Tuple[str, ...], sys_path: Optional[List[str]] = None, module_context: Optional[Context] = None, prefer_stubs: bool = True) -> ValueSet:
    ...

@plugin_manager.decorate()
@import_module_decorator
def import_module(inference_state: Any, import_names: Tuple[str, ...], parent_module_value: Any, sys_path: List[str]) -> ValueSet:
    ...

def _load_python_module(inference_state: Any, file_io: Any, import_names: Optional[Tuple[str, ...]] = None, is_package: bool = False) -> Any:
    ...

def _load_builtin_module(inference_state: Any, import_names: Optional[Tuple[str, ...]] = None, sys_path: Optional[List[str]] = None) -> Any:
    ...

def load_module_from_path(inference_state: Any, file_io: Any, import_names: Optional[Tuple[str, ...]] = None, is_package: Optional[bool] = None) -> Any:
    ...

def load_namespace_from_path(inference_state: Any, folder_io: Any) -> ImplicitNamespaceValue:
    ...

def follow_error_node_imports_if_possible(context: Context, name: Any) -> Optional[ValueSet]:
    ...

def iter_module_names(inference_state: Any, module_context: Context, search_path: List[str], module_cls: Any = ImportName, add_builtin_modules: bool = True) -> Iterable[Any]:
    ...