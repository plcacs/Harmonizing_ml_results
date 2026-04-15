from __future__ import annotations
from collections.abc import Callable, Mapping
from functools import wraps
import inspect
from typing import Any, TypeVar, overload, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from pandas._typing import F, T

__all__: list[str] = ...

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = ...,
    klass: type[Warning] | None = ...,
    stacklevel: int = ...,
    msg: str | None = ...,
) -> Callable[..., Any]: ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]: ...

def _format_argument_list(allow_args: list[str] | tuple[str, ...] | int) -> str: ...

def future_version_msg(version: str | None) -> str: ...

def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = ...,
    name: str | None = ...,
) -> Callable[[F], F]: ...

@overload
def doc(*docstrings: None | str | Callable[..., Any], **params: str) -> Callable[[F], F]: ...
@overload
def doc(*docstrings: None | str | Callable[..., Any]) -> Callable[[F], F]: ...

class Substitution:
    def __init__(self, *args: str, **kwargs: str) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    def __init__(self, addendum: str, join: str = ..., indents: int = ...) -> None: ...
    def __call__(self, func: F) -> F: ...

def indent(text: str | None, indents: int = ...) -> str: ...

def set_module(module: str | None) -> Callable[[F], F]: ...

# Re-export from pandas._libs.properties
cache_readonly: Any = ...