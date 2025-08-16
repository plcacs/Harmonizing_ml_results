from __future__ import annotations
from functools import wraps
import inspect
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import F, T
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

def deprecate(name: str, alternative: Callable, version: str, alt_name: Optional[str] = None, klass: Optional[Warning] = None, stacklevel: int = 2, msg: Optional[str] = None) -> Callable:
    ...

def deprecate_kwarg(old_arg_name: str, new_arg_name: str, mapping: Optional[Mapping] = None, stacklevel: int = 2) -> Callable:
    ...

def _format_argument_list(allow_args: list) -> str:
    ...

def future_version_msg(version: Optional[str]) -> str:
    ...

def deprecate_nonkeyword_arguments(version: Optional[str], allowed_args: Optional[list] = None, name: Optional[str] = None) -> Callable:
    ...

def doc(*docstrings: Optional[str], **params: str) -> Callable:
    ...

class Substitution:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __call__(self, func: Callable) -> Callable:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

class Appender:
    def __init__(self, addendum: str, join: str = '', indents: int = 0) -> None:
        ...

    def __call__(self, func: Callable) -> Callable:
        ...

def indent(text: str, indents: int = 1) -> str:
    ...

def set_module(module: str) -> Callable:
    ...
