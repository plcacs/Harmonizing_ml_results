from collections.abc import Callable, Mapping
from typing import Any
from pandas._libs.properties import cache_readonly
from pandas._typing import F

def deprecate(
    name: str,
    alternative: F,
    version: str,
    alt_name: str | None = ...,
    klass: type[Warning] | None = ...,
    stacklevel: int = ...,
    msg: str | None = ...,
) -> F: ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]: ...

def _format_argument_list(allow_args: list[str]) -> str: ...

def future_version_msg(version: str | None) -> str: ...

def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = ...,
    name: str | None = ...,
) -> Callable[[F], F]: ...

def doc(*docstrings: str | None | Callable[..., Any], **params: Any) -> Callable[[F], F]: ...

class Substitution:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    def __init__(self, addendum: str | None, join: str = ..., indents: int = ...) -> None: ...
    def __call__(self, func: F) -> F: ...

def indent(text: object, indents: int = ...) -> str: ...

__all__: list[str]

def set_module(module: str | None) -> Callable[[F], F]: ...