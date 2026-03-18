```python
from __future__ import annotations
from typing import Any, Callable, TypeVar, overload
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")
_F = TypeVar("_F")
_T = TypeVar("_T")

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
    mapping: dict[Any, Any] | Callable[..., Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[[_F], _F]: ...

def _format_argument_list(allow_args: list[str] | tuple[str, ...] | int) -> str: ...

def future_version_msg(version: str | None) -> str: ...

def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = ...,
    name: str | None = ...,
) -> Callable[[_F], _F]: ...

def doc(
    *docstrings: None | str | Callable[..., Any],
    **params: Any,
) -> Callable[[_F], _F]: ...

class Substitution:
    params: tuple[Any, ...] | dict[str, Any]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: _F) -> _F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    addendum: str
    join: str
    def __init__(self, addendum: str, join: str = ..., indents: int = ...) -> None: ...
    def __call__(self, func: _F) -> _F: ...

def indent(text: str, indents: int = ...) -> str: ...

def set_module(module: str | None) -> Callable[[_F], _F]: ...

cache_readonly: Any = ...
```