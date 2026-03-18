```python
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

class Script:
    _orig_path: Optional[Union[str, Path]]
    path: Optional[Path]
    _inference_state: Any
    _module_node: Any
    _code_lines: List[str]
    _code: str

    def __init__(
        self,
        code: Optional[str] = None,
        *,
        path: Optional[Union[str, Path]] = None,
        environment: Optional[Any] = None,
        project: Optional[Any] = None,
    ) -> None: ...
    def _get_module(self) -> Any: ...
    def _get_module_context(self) -> Any: ...
    def __repr__(self) -> str: ...
    def complete(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        fuzzy: bool = False,
    ) -> List[Any]: ...
    def infer(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Any]: ...
    def goto(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        follow_imports: bool = False,
        follow_builtin_imports: bool = False,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Any]: ...
    def search(self, string: str, *, all_scopes: bool = False) -> List[Any]: ...
    def _search_func(
        self,
        string: str,
        all_scopes: bool = False,
        complete: bool = False,
        fuzzy: bool = False,
    ) -> List[Any]: ...
    def complete_search(self, string: str, **kwargs: Any) -> List[Any]: ...
    def help(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> List[Any]: ...
    def get_references(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]: ...
    def get_signatures(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> List[Any]: ...
    def get_context(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any: ...
    def _analysis(self) -> List[Any]: ...
    def get_names(self, **kwargs: Any) -> List[Any]: ...
    def get_syntax_errors(self) -> List[Any]: ...
    def _names(
        self,
        all_scopes: bool = False,
        definitions: bool = True,
        references: bool = False,
    ) -> List[Any]: ...
    def rename(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        new_name: str,
    ) -> Any: ...
    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = None,
        until_column: Optional[int] = None,
    ) -> Any: ...
    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = None,
        until_column: Optional[int] = None,
    ) -> Any: ...
    def inline(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any: ...

class Interpreter(Script):
    _allow_descriptor_getattr_default: bool
    namespaces: List[Dict[str, Any]]

    def __init__(
        self,
        code: str,
        namespaces: List[Dict[str, Any]],
        *,
        project: Optional[Any] = None,
        **kwds: Any,
    ) -> None: ...
    def _get_module_context(self) -> Any: ...

def preload_module(*modules: str) -> None: ...
def set_debug_function(
    func_cb: Optional[Callable[..., Any]] = ...,
    warnings: bool = True,
    notices: bool = True,
    speed: bool = True,
) -> None: ...
```