import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import Any, Iterator, Iterable, Tuple, Set, Dict, Callable, List, TextIO, Optional

class Error(Exception):
    """Exception raised by errors in the options module."""
    pass

class OptionParser:
    """A collection of options, a dictionary with object-like access.

    Normally accessed via static functions in the `tornado.options` module,
    which reference a global instance.
    """

    def __init__(self) -> None:
        self._options: Dict[str, '_Option'] = {}
        self._parse_callbacks: List[Callable[[], None]] = []
        self.define('help', type=bool, help='show this help information', callback=self._help_callback)

    def _normalize_name(self, name: str) -> str:
        return name.replace('_', '-')

    def __getattr__(self, name: str) -> Any:
        name = self._normalize_name(name)
        if isinstance(self._options.get(name), _Option):
            return self._options[name].value()
        raise AttributeError('Unrecognized option %r' % name)

    def __setattr__(self, name: str, value: Any) -> None:
        name = self._normalize_name(name)
        if isinstance(self._options.get(name), _Option):
            return self._options[name].set(value)
        raise AttributeError('Unrecognized option %r' % name)

    def __iter__(self) -> Iterator[str]:
        return (opt.name for opt in self._options.values())

    def __contains__(self, name: str) -> bool:
        name = self._normalize_name(name)
        return name in self._options

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def __setitem__(self, name: str, value: Any) -> None:
        return self.__setattr__(name, value)

    def items(self) -> Iterable[Tuple[str, Any]]:
        return [(opt.name, opt.value()) for name, opt in self._options.items()]

    def groups(self) -> Set[str]:
        return {opt.group_name for opt in self._options.values()}

    def group_dict(self, group: str) -> Dict[str, Any]:
        return {opt.name: opt.value() for name, opt in self._options.items() if not group or group == opt.group_name}

    def as_dict(self) -> Dict[str, Any]:
        return {opt.name: opt.value() for name, opt in self._options.items()}

    def define(self, name: str, default: Any = None, type: Any = None, help: str = None, metavar: str = None, multiple: bool = False, group: str = None, callback: Callable[[Any], None] = None) -> None:
        ...

    def parse_command_line(self, args: Optional[List[str]] = None, final: bool = True) -> List[str]:
        ...

    def parse_config_file(self, path: str, final: bool = True) -> None:
        ...

    def print_help(self, file: Optional[TextIO] = None) -> None:
        ...

    def _help_callback(self, value: bool) -> None:
        ...

    def add_parse_callback(self, callback: Callable[[], None]) -> None:
        ...

    def run_parse_callbacks(self) -> None:
        ...

    def mockable(self) -> '_Mockable':
        ...

class _Mockable:
    ...

class _Option:
    UNSET = object()

    def __init__(self, name: str, default: Any = None, type: Any = None, help: str = None, metavar: str = None, multiple: bool = False, file_name: Optional[str] = None, group_name: Optional[str] = None, callback: Optional[Callable[[Any], None]] = None) -> None:
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

options = OptionParser()
