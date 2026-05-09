from __future__ import annotations
from functools import wraps
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    F,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)
import warnings
from pandas._typing import T

__all__: List[str] = [
    "Appender",
    "Substitution",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
]

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: Optional[str] = None,
    klass: Optional[Union[Callable[..., Any], type]] = None,
    stacklevel: int = 2,
    msg: Optional[str] = None,
) -> Callable[..., Any]:
    ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: Optional[str] = None,
    mapping: Optional[Union[Dict[Any, Any], Callable[[Any], Any]]] = None,
    stacklevel: int = 2,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def deprecate_nonkeyword_arguments(
    version: Optional[str] = None,
    allowed_args: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def doc(
    *docstrings: Optional[Union[str, Callable[..., Any]]],
    **params: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def future_version_msg(version: Optional[str] = None) -> str:
    ...

class Substitution:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

class Appender:
    def __init__(self, addendum: str, join: str = "", indents: int = 0) -> None:
        ...

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        ...

def indent(text: str, indents: int = 1) -> str:
    ...

def set_module(module: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...