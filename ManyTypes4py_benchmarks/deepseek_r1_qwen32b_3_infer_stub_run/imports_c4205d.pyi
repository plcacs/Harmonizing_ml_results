"""
Stub file for jedi.inference.imports
"""

from typing import (
    Any, Dict, Iterable, List, Optional, Set, Tuple, Union, overload
)
from pathlib import Path
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.typeshed import (
    parse_stub_module, create_stub_module, import_module_decorator
)
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.sys_path import sys_path
from jedi.inference.value import ModuleValue
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FolderIO
from jedi import settings, debug
import parso

class ModuleCache:
    def __init__(self) -> None:
        ...
    
    def add(self, string_names: Tuple[str, ...], value_set: ValueSet) -> None:
        ...
    
    def get(self, string_names: Tuple[str, ...]) -> Optional[ValueSet]:
        ...

@inference_state_method_cache(default=NO_VALUES)
def infer_import(context: Any, tree_name: parso.tree.Name) -> ValueSet:
    ...

@inference_state_method_cache(default=[])
def goto_import(context: Any, tree_name: parso.tree.Name) -> Set[str]:
    ...

def _prepare_infer_import(
    module_context: Any,
    tree_name: parso.tree.Name
) -> Tuple[Optional[str], Tuple[str, ...], int, ValueSet]:
    ...

def _add_error(value: Any, name: Any, message: str) -> None:
    ...

def _level_to_base_import_path(
    project_path: str,
    directory: str,
    level: int
) -> Tuple[Optional[List[str]], Optional[str]]:
    ...

class Importer:
    def __init__(
        self,
        inference_state: Any,
        import_path: Tuple[Union[str, parso.tree.Name], ...],
        module_context: Any,
        level: int = 0
    ) -> None:
        ...
    
    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        ...
    
    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        ...
    
    def follow(self) -> ValueSet:
        ...
    
    def _get_module_names(
        self,
        search_path: Optional[List[str]] = None,
        in_module: Any = None
    ) -> List[Union[ImportName, SubModuleName]]:
        ...
    
    def completion_names(self, only_modules: bool = False) -> List[ImportName]:
        ...

def import_module_by_names(
    inference_state: Any,
    import_names: Tuple[Union[str, parso.tree.Name], ...],
    sys_path: Optional[List[str]] = None,
    module_context: Any = None,
    prefer_stubs: bool = True
) -> ValueSet:
    ...

@plugin_manager.decorate()
@import_module_decorator
def import_module(
    inference_state: Any,
    import_names: Tuple[str, ...],
    parent_module_value: Any,
    sys_path: List[str]
) -> ValueSet:
    ...

def _load_python_module(
    inference_state: Any,
    file_io: FolderIO,
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: bool = False
) -> ModuleValue:
    ...

def _load_builtin_module(
    inference_state: Any,
    import_names: Optional[Tuple[str, ...]] = None,
    sys_path: Optional[List[str]] = None
) -> Optional[ModuleValue]:
    ...

def load_module_from_path(
    inference_state: Any,
    file_io: FolderIO,
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: Optional[bool] = None
) -> ModuleValue:
    ...

def load_namespace_from_path(
    inference_state: Any,
    folder_io: FolderIO
) -> Any:
    ...

def follow_error_node_imports_if_possible(
    context: Any,
    name: parso.tree.Name
) -> Optional[ValueSet]:
    ...

def iter_module_names(
    inference_state: Any,
    module_context: Any,
    search_path: List[str],
    module_cls: type = ImportName,
    add_builtin_modules: bool = True
) -> Iterable[Union[ImportName, SubModuleName]]:
    ...