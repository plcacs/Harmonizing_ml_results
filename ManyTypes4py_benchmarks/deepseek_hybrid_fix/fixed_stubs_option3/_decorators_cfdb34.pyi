from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = None,
    klass: type[Warning] | None = None,
    stacklevel: int = 2,
    msg: str | None = None,
) -> Callable[..., Any]: ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]: ...

def _format_argument_list(allow_args: list[str]) -> str: ...

def future_version_msg(version: str | None) -> str: ...

def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]: ...

def doc(
    *docstrings: None | str | Callable[..., Any],
    **params: Any,
) -> Callable[[F], F]: ...

class Substitution:
    def __init__(self, *args: str, **kwargs: str) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    def __init__(self, addendum: str, join: str = "", indents: int = 0) -> None: ...
    def __call__(self, func: F) -> F: ...

def indent(text: str | None, indents: int = 1) -> str: ...

def set_module(module: str | None) -> Callable[[F], F]: ...

__all__ = [
    "Appender",
    "Substitution",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
]