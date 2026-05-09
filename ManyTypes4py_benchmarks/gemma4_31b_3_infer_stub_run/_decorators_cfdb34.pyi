from __future__ import annotations
from typing import Any, Callable, Optional, Union, overload, Type, cast
from collections.abc import Mapping

# F and T are internal pandas types, usually representing Callable or TypeVar
# In a stub, we use Callable or Any for these generic wrappers.
F = Callable[..., Any]

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: Optional[str] = ...,
    klass: Optional[Type[Warning]] = ...,
    stacklevel: int = 2,
    msg: Optional[str] = ...,
) -> Callable[..., Any]: ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: Optional[str],
    mapping: Optional[Union[Mapping[Any, Any], Callable[[Any], Any]]] = ...,
    stacklevel: int = 2,
) -> Callable[[Callable[..., Any]], F]: ...

def _format_argument_list(allow_args: list[str]) -> str: ...

def future_version_msg(version: Optional[str]) -> str: ...

def deprecate_nonkeyword_arguments(
    version: Optional[str],
    allowed_args: Optional[list[str]] = ...,
    name: Optional[str] = ...,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

def doc(
    *docstrings: Any,
    **params: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

class Substitution:
    params: Union[tuple[Any, ...], Mapping[str, Any]]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    addendum: str
    join: str
    def __init__(self, addendum: str, join: str = '', indents: int = 0) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

def indent(text: str, indents: int = 1) -> str: ...

def set_module(module: Optional[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

# Re-exporting from pandas._libs.properties as per __all__
from pandas._libs.properties import cache_readonly

__all__ = [
    'Appender',
    'Substitution',
    'cache_readonly',
    'deprecate',
    'deprecate_kwarg',
    'deprecate_nonkeyword_arguments',
    'doc',
    'future_version_msg',
]