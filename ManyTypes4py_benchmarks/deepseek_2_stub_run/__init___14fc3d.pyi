```python
from typing import Any, Optional, List, Iterator, Union, overload
from pathlib import Path
from jedi.api import classes
from jedi.api.environment import Environment, InterpreterEnvironment
from jedi.api.project import Project
from jedi.api.refactoring import Refactoring

def set_debug_function(
    func_cb: Any = ...,
    warnings: bool = ...,
    notices: bool = ...,
    speed: bool = ...
) -> None: ...

def preload_module(*modules: str) -> None: ...

class Script:
    path: Optional[Path]
    _orig_path: Any
    _inference_state: Any
    _module_node: Any
    _code_lines: Any
    _code: Any

    def __init__(
        self,
        code: Optional[str] = ...,
        *,
        path: Optional[Union[str, Path]] = ...,
        environment: Optional[Environment] = ...,
        project: Optional[Project] = ...
    ) -> None: ...

    def __repr__(self) -> str: ...

    @overload
    def complete(self) -> List[classes.Completion]: ...
    @overload
    def complete(self, line: int) -> List[classes.Completion]: ...
    @overload
    def complete(self, line: int, column: int) -> List[classes.Completion]: ...
    def complete(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        fuzzy: bool = ...
    ) -> List[classes.Completion]: ...

    @overload
    def infer(self) -> List[classes.Name]: ...
    @overload
    def infer(self, line: int) -> List[classes.Name]: ...
    @overload
    def infer(self, line: int, column: int) -> List[classes.Name]: ...
    def infer(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...
    ) -> List[classes.Name]: ...

    @overload
    def goto(self) -> List[classes.Name]: ...
    @overload
    def goto(self, line: int) -> List[classes.Name]: ...
    @overload
    def goto(self, line: int, column: int) -> List[classes.Name]: ...
    def goto(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        follow_imports: bool = ...,
        follow_builtin_imports: bool = ...,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...
    ) -> List[classes.Name]: ...

    def search(self, string: str, *, all_scopes: bool = ...) -> Iterator[classes.Name]: ...

    def complete_search(
        self,
        string: str,
        *,
        all_scopes: bool = ...,
        fuzzy: bool = ...
    ) -> Iterator[classes.Completion]: ...

    @overload
    def help(self) -> List[classes.Name]: ...
    @overload
    def help(self, line: int) -> List[classes.Name]: ...
    @overload
    def help(self, line: int, column: int) -> List[classes.Name]: ...
    def help(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...
    ) -> List[classes.Name]: ...

    @overload
    def get_references(self) -> List[classes.Name]: ...
    @overload
    def get_references(self, line: int) -> List[classes.Name]: ...
    @overload
    def get_references(self, line: int, column: int) -> List[classes.Name]: ...
    def get_references(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        **kwargs: Any
    ) -> List[classes.Name]: ...

    @overload
    def get_signatures(self) -> List[classes.Signature]: ...
    @overload
    def get_signatures(self, line: int) -> List[classes.Signature]: ...
    @overload
    def get_signatures(self, line: int, column: int) -> List[classes.Signature]: ...
    def get_signatures(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...
    ) -> List[classes.Signature]: ...

    @overload
    def get_context(self) -> classes.Name: ...
    @overload
    def get_context(self, line: int) -> classes.Name: ...
    @overload
    def get_context(self, line: int, column: int) -> classes.Name: ...
    def get_context(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...
    ) -> classes.Name: ...

    def get_names(
        self,
        *,
        all_scopes: bool = ...,
        definitions: bool = ...,
        references: bool = ...
    ) -> List[classes.Name]: ...

    def get_syntax_errors(self) -> List[Any]: ...

    def rename(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        new_name: str
    ) -> Refactoring: ...

    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...
    ) -> Refactoring: ...

    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...
    ) -> Refactoring: ...

    def inline(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...
    ) -> Refactoring: ...

class Interpreter(Script):
    _allow_descriptor_getattr_default: bool
    namespaces: List[dict]

    def __init__(
        self,
        code: str,
        namespaces: List[dict],
        *,
        project: Optional[Project] = ...,
        **kwds: Any
    ) -> None: ...
```