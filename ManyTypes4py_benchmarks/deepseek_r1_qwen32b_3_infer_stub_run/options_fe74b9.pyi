"""Stub file for tornado.options module."""

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TextIO,
    AnyStr,
    overload,
)
import datetime
from tornado.options import OptionParser

class Error(Exception):
    pass

class OptionParser:
    _options: Dict[str, Any]
    _parse_callbacks: List[Callable[[], None]]
    __dict__: Dict[str, Any]

    def __init__(self) -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __contains__(self, name: str) -> bool:
        ...

    def __getitem__(self, name: str) -> Any:
        ...

    def __setitem__(self, name: str, value: Any) -> None:
        ...

    def items(self) -> Iterable[Tuple[str, Any]]:
        ...

    def groups(self) -> Set[str]:
        ...

    def group_dict(self, group: Optional[str] = None) -> Dict[str, Any]:
        ...

    def as_dict(self) -> Dict[str, Any]:
        ...

    def define(
        self,
        name: str,
        default: Any = ...,
        type: Any = ...,
        help: Optional[str] = ...,
        metavar: Optional[str] = ...,
        multiple: bool = ...,
        group: Optional[str] = ...,
        callback: Optional[Callable[[Any], None]] = ...,
    ) -> None:
        ...

    def parse_command_line(
        self,
        args: Optional[List[str]] = ...,
        final: bool = ...,
    ) -> List[str]:
        ...

    def parse_config_file(
        self,
        path: str,
        final: bool = ...,
    ) -> None:
        ...

    def print_help(self, file: Optional[TextIO] = ...) -> None:
        ...

    def add_parse_callback(self, callback: Callable[[], None]) -> None:
        ...

    def run_parse_callbacks(self) -> None:
        ...

    def mockable(self) -> _Mockable:
        ...

class _Mockable:
    def __init__(self, options: OptionParser) -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...

    def __delattr__(self, name: str) -> None:
        ...

class _Option:
    UNSET: Any
    name: str
    type: Any
    help: Optional[str]
    metavar: Optional[str]
    multiple: bool
    file_name: Optional[str]
    group_name: Optional[str]
    callback: Optional[Callable[[Any], None]]
    default: Any
    _value: Any

    def __init__(
        self,
        name: str,
        default: Any = ...,
        type: Any = ...,
        help: Optional[str] = ...,
        metavar: Optional[str] = ...,
        multiple: bool = ...,
        file_name: Optional[str] = ...,
        group_name: Optional[str] = ...,
        callback: Optional[Callable[[Any], None]] = ...,
    ) -> None:
        ...

    def value(self) -> Any:
        ...

    def parse(self, value: str) -> Any:
        ...

    def set(self, value: Any) -> None:
        ...

    def _parse_datetime(self, value: str) -> datetime.datetime:
        ...

    def _parse_timedelta(self, value: str) -> datetime.timedelta:
        ...

    def _parse_bool(self, value: str) -> bool:
        ...

    def _parse_string(self, value: str) -> str:
        ...

def define(
    name: str,
    default: Any = ...,
    type: Any = ...,
    help: Optional[str] = ...,
    metavar: Optional[str] = ...,
    multiple: bool = ...,
    group: Optional[str] = ...,
    callback: Optional[Callable[[Any], None]] = ...,
) -> None:
    ...

def parse_command_line(
    args: Optional[List[str]] = ...,
    final: bool = ...,
) -> List[str]:
    ...

def parse_config_file(
    path: str,
    final: bool = ...,
) -> None:
    ...

def print_help(file: Optional[TextIO] = ...) -> None:
    ...

def add_parse_callback(callback: Callable[[], None]) -> None:
    ...

options: OptionParser