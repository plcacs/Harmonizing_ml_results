from __future__ import annotations
from functools import wraps
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    F,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import F, T

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: Optional[str] = None,
    klass: type = FutureWarning,
    stacklevel: int = 2,
    msg: Optional[str] = None,
) -> F:
    ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: Union[str, None],
    mapping: Optional[Union[Dict[Any, Any], Callable[[Any], Any]]] = None,
    stacklevel: int = 2,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def _format_argument_list(allow_args: List[str]) -> str:
    ...

def future_version_msg(version: Optional[str]) -> str:
    ...

def deprecate_nonkeyword_arguments(
    version: Optional[str],
    allowed_args: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def doc(*docstrings: str, **params: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

class Substitution:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        ...
    
    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

class Appender:
    def __init__(self, addendum: str, join: str = '', indents: int = 0) -> None:
        ...
    
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        ...

def indent(text: str, indents: int = 1) -> str:
    ...

def set_module(module: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...