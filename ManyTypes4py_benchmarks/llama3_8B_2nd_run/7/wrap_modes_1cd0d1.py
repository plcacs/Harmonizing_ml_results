from enum import enum
from inspect import signature
from typing import Any, Callable, Dict, List
import isort.comments

_wrap_modes: Dict[str, Callable] = {}

def from_string(value: str) -> Any:
    return getattr(WrapModes, str(value), None) or WrapModes(int(value))

def formatter_from_string(name: str) -> Callable:
    return _wrap_modes.get(name.upper(), grid)

def _wrap_mode_interface(statement: str, imports: List[str], white_space: str, indent: str, line_length: int, comments: List[str], line_separator: str, comment_prefix: str, include_trailing_comma: bool, remove_comments: bool) -> str:
    """Defines the common interface used by all wrap mode functions"""
    return ''

def _wrap_mode(function: Callable) -> Callable:
    """Registers an individual wrap mode. Function name and order are significant and used for
    creating enum.
    """
    _wrap_modes[function.__name__.upper()] = function
    function.__signature__ = signature(_wrap_mode_interface)
    function.__annotations__ = _wrap_mode_interface.__annotations__
    return function

@_wrap_mode
def grid(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def hanging_indent(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_hanging_indent(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_grid(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_grid_grouped(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_grid_grouped_no_comma(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def noqa(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_hanging_indent_bracket(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def vertical_prefix_from_module_import(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def hanging_indent_with_parentheses(**interface: Dict[str, Any]) -> str:
    ...

@_wrap_mode
def backslash_grid(**interface: Dict[str, Any]) -> str:
    ...

WrapModes = enum.Enum('WrapModes', {wrap_mode: index for index, wrap_mode in enumerate(_wrap_modes.keys())})
