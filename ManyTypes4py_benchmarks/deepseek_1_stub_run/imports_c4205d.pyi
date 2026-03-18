```python
import os
from pathlib import Path
from typing import Any, Optional, Tuple, List, Iterator, Union
from parso.python import tree
from parso.tree import BaseNode
from jedi.inference.base_value import ValueSet
from jedi.inference.value import ModuleValue
from jedi.file_io import FileIO, FolderIO
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.inference.context import ModuleContext, ValueContext

class ModuleCache:
    _name_cache: Any = ...
    def __init__(self) -> None: ...
    def add(self, string_names: Any, value_set: ValueSet) -> None: ...
    def get(self, string_names: Any) -> Any: ...

def infer_import(context: Any, tree_name: Any) -> ValueSet: ...
def goto_import(context: Any, tree_name: Any) -> List[Any]: ...
def _prepare_infer_import(module_context: Any, tree_name: Any) -> Tuple[Any, ...]: ...
def _add_error(value: Any, name: Any, message: str) -> None: ...
def _level_to_base_import_path(project_path: Any, directory: str, level: int) -> Tuple[Any, Any]: ...

class Importer:
    def __init__(
        self,
        inference_state: Any,
        import_path: Tuple[Any, ...],
        module_context: Any,
        level: int = 0
    ) -> None: ...
    @property
    def _str_import_path(self) -> Tuple[str, ...]: ...
    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]: ...
    def follow(self) -> ValueSet: ...
    def _get_module_names(
        self,
        search_path: Optional[List[str]] = None,
        in_module: Any = None
    ) -> List[Any]: ...
    def completion_names(
        self,
        inference_state: Any,
        only_modules: bool = False
    ) -> List[Any]: ...

def import_module_by_names(
    inference_state: Any,
    import_names: Tuple[Any, ...],
    sys_path: Optional[List[str]] = None,
    module_context: Optional[Any] = None,
    prefer_stubs: bool = True
) -> ValueSet: ...

def import_module(
    inference_state: Any,
    import_names: Tuple[str, ...],
    parent_module_value: Optional[Any],
    sys_path: List[str]
) -> ValueSet: ...

def _load_python_module(
    inference_state: Any,
    file_io: FileIO,
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: bool = False
) -> ModuleValue: ...

def _load_builtin_module(
    inference_state: Any,
    import_names: Optional[Tuple[str, ...]] = None,
    sys_path: Optional[List[str]] = None
) -> Optional[Any]: ...

def load_module_from_path(
    inference_state: Any,
    file_io: FileIO,
    import_names: Optional[Tuple[str, ...]] = None,
    is_package: Optional[bool] = None
) -> Any: ...

def load_namespace_from_path(
    inference_state: Any,
    folder_io: FolderIO
) -> Any: ...

def follow_error_node_imports_if_possible(
    context: Any,
    name: Any
) -> Optional[ValueSet]: ...

def iter_module_names(
    inference_state: Any,
    module_context: Any,
    search_path: List[str],
    module_cls: Any = ...,
    add_builtin_modules: bool = True
) -> Iterator[Any]: ...
```