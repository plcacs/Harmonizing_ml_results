```pyi
from __future__ import annotations

from functools import wraps as wraps
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._typing import F, T

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

def doc(*docstrings: str | Callable[..., Any] | None, **params: Any) -> Callable[[F], F]: ...

class Substitution:
    params: tuple[Any, ...] | dict[str, Any]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    addendum: str
    join: str
    def __init__(self, addendum: str, join: str = '', indents: int = 0) -> None: ...
    def __call__(self, func: F) -> F: ...

def indent(text: str | Any, indents: int = 1) -> str: ...

def set_module(module: str | None) -> Callable[[F], F]: ...

__all__: list[str]
```