import sys
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

from jedi.api import classes
from jedi.api.project import Project
from jedi.inference import InferenceState


class Script:
    _orig_path: Optional[Union[str, Path]]
    path: Optional[Path]
    _inference_state: InferenceState
    _module_node: Any
    _code_lines: List[str]
    _code: str

    def __init__(
        self,
        code: Optional[Union[str, bytes]] = ...,
        *,
        path: Optional[Union[str, Path]] = ...,
        environment: Optional[Any] = ...,
        project: Optional[Project] = ...,
    ) -> None: ...
    def _get_module(self) -> Any: ...
    def _get_module_context(self) -> Any: ...
    def __repr__(self) -> str: ...
    def complete(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        fuzzy: bool = ...,
    ) -> List[classes.Completion]: ...
    def infer(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]: ...
    def goto(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        follow_imports: bool = ...,
        follow_builtin_imports: bool = ...,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]: ...
    def search(
        self, string: str, *, all_scopes: bool = ...
    ) -> List[classes.Name]: ...
    def _search_func(
        self,
        string: str,
        all_scopes: bool = ...,
        complete: bool = ...,
        fuzzy: bool = ...,
    ) -> List[Any]: ...
    def complete_search(
        self, string: str, **kwargs: Any
    ) -> List[classes.Completion]: ...
    def help(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> List[classes.Name]: ...
    def get_references(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        **kwargs: Any,
    ) -> List[classes.Name]: ...
    def get_signatures(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> List[classes.Signature]: ...
    def get_context(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> classes.Name: ...
    def _analysis(self) -> List[Any]: ...
    def get_names(self, **kwargs: Any) -> List[classes.Name]: ...
    def get_syntax_errors(self) -> List[Any]: ...
    def _names(
        self,
        all_scopes: bool = ...,
        definitions: bool = ...,
        references: bool = ...,
    ) -> List[Any]: ...
    def rename(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        new_name: str,
    ) -> Any: ...
    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...,
    ) -> Any: ...
    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...,
    ) -> Any: ...
    def inline(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> Any: ...


class Interpreter(Script):
    _allow_descriptor_getattr_default: bool
    namespaces: List[dict[str, Any]]

    def __init__(
        self,
        code: Union[str, bytes],
        namespaces: Sequence[Any],
        *,
        project: Optional[Project] = ...,
        **kwds: Any,
    ) -> None: ...
    def _get_module_context(self) -> Any: ...


def preload_module(*modules: str) -> None: ...
def set_debug_function(
    func_cb: Callable[..., Any] = ...,
    warnings: bool = ...,
    notices: bool = ...,
    speed: bool = ...,
) -> None: ...