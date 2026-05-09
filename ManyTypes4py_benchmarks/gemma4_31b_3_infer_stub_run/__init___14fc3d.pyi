import pathlib
from typing import Any, Callable, Iterable, List, Optional, Union, overload

from jedi.api.completion import Completion
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import Project
from jedi.api.classes import Name
from jedi.api.refactoring import Refactoring
from jedi.api.errors import SyntaxError
from jedi.api.signatures import Signature

class Script:
    """
    A Script is the base for completions, goto or whatever you want to do with
    Jedi.
    """
    path: Optional[pathlib.Path]

    def __init__(
        self,
        code: Optional[str] = None,
        *,
        path: Optional[Union[str, pathlib.Path]] = None,
        environment: Optional[Any] = None,
        project: Optional[Project] = None,
    ) -> None: ...

    def complete(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        fuzzy: bool = False,
    ) -> List[Completion]: ...

    def infer(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Name]: ...

    def goto(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        *,
        follow_imports: bool = False,
        follow_builtin_imports: bool = False,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Name]: ...

    def search(self, string: str, *, all_scopes: bool = False) -> Iterable[Name]: ...

    def complete_search(self, string: str, **kwargs: Any) -> List[Completion]: ...

    def help(self, line: Optional[int] = None, column: Optional[int] = None) -> List[Name]: ...

    def get_references(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Name]: ...

    def get_signatures(self, line: Optional[int] = None, column: Optional[int] = None) -> List[Signature]: ...

    def get_context(self, line: Optional[int] = None, column: Optional[int] = None) -> Name: ...

    def get_names(self, **kwargs: Any) -> List[Name]: ...

    def get_syntax_errors(self) -> List[SyntaxError]: ...

    def rename(self, line: Optional[int] = None, column: Optional[int] = None, *, new_name: str) -> Refactoring: ...

    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = None,
        until_column: Optional[int] = None,
    ) -> Refactoring: ...

    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = None,
        until_column: Optional[int] = None,
    ) -> Refactoring: ...

    def inline(self, line: Optional[int] = None, column: Optional[int] = None) -> Refactoring: ...

class Interpreter(Script):
    """
    Jedi's API for Python REPLs.
    """
    namespaces: List[dict[str, Any]]

    def __init__(
        self,
        code: str,
        namespaces: List[dict[str, Any]],
        *,
        project: Optional[Project] = None,
        **kwds: Any,
    ) -> None: ...

def preload_module(*modules: str) -> None: ...

def set_debug_function(
    func_cb: Optional[Callable[[str], None]] = None,
    warnings: bool = True,
    notices: bool = True,
    speed: bool = True,
) -> None: ...