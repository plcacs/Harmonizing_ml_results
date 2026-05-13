from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from jedi.api.classes import Completion, Name, Signature
from jedi.api.environment import Environment as _Environment
from jedi.api.refactoring import Refactoring

class Script:
    code: str | bytes | None
    path: Path | None

    def __init__(
        self,
        code: str | None = None,
        *,
        path: str | Path | None = None,
        environment: _Environment | None = None,
        project: Any | None = None,
    ) -> None: ...

    def complete(
        self,
        line: int | None = None,
        column: int | None = None,
        *,
        fuzzy: bool = False,
    ) -> list[Completion]: ...

    def infer(
        self,
        line: int | None = None,
        column: int | None = None,
        *,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> list[Name]: ...

    def goto(
        self,
        line: int | None = None,
        column: int | None = None,
        *,
        follow_imports: bool = False,
        follow_builtin_imports: bool = False,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> list[Name]: ...

    def search(self, string: str, *, all_scopes: bool = False) -> Iterator[Name]: ...

    def complete_search(self, string: str, **kwargs: Any) -> Iterator[Completion]: ...

    def help(self, line: int | None = None, column: int | None = None) -> list[Name]: ...

    def get_references(
        self,
        line: int | None = None,
        column: int | None = None,
        **kwargs: Any,
    ) -> list[Name]: ...

    def get_signatures(self, line: int | None = None, column: int | None = None) -> list[Signature]: ...

    def get_context(self, line: int | None = None, column: int | None = None) -> Name: ...

    def get_names(self, **kwargs: Any) -> list[Name]: ...

    def get_syntax_errors(self) -> list[Any]: ...

    def rename(
        self,
        line: int | None = None,
        column: int | None = None,
        *,
        new_name: str,
    ) -> Refactoring: ...

    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: int | None = None,
        until_column: int | None = None,
    ) -> Refactoring: ...

    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: int | None = None,
        until_column: int | None = None,
    ) -> Refactoring: ...

    def inline(self, line: int | None = None, column: int | None = None) -> Refactoring: ...

class Interpreter(Script):
    namespaces: list[dict[str, Any]]

    def __init__(
        self,
        code: str,
        namespaces: list[dict[str, Any]],
        *,
        project: Any | None = None,
        **kwds: Any,
    ) -> None: ...

def preload_module(*modules: str) -> None: ...

def set_debug_function(
    func_cb: Any = ...,
    warnings: bool = True,
    notices: bool = True,
    speed: bool = True,
) -> None: ...