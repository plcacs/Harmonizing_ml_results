import datetime
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
    overload,
)

class Error(Exception):
    """Exception raised by errors in the options module."""
    ...

class OptionParser:
    """A collection of options, a dictionary with object-like access."""

    def __init__(self) -> None:
        ...

    def _normalize_name(self, name: str) -> str:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...

    def __iter__(self) -> Iterable[str]:
        ...

    def __contains__(self, name: str) -> bool:
        ...

    def __getitem__(self, name: str) -> Any:
        ...

    def __setitem__(self, name: str, value: Any) -> None:
        ...

    def items(self) -> List[Tuple[str, Any]]:
        """An iterable of (name, value) pairs."""
        ...

    def groups(self) -> Set[str]:
        """The set of option-groups created by ``define``."""
        ...

    def group_dict(self, group: str) -> Dict[str, Any]:
        """The names and values of options in a group."""
        ...

    def as_dict(self) -> Dict[str, Any]:
        """The names and values of all options."""
        ...

    @overload
    def define(
        self,
        name: str,
        default: None = None,
        type: Optional[type] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        multiple: bool = False,
        group: Optional[str] = None,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        ...

    @overload
    def define(
        self,
        name: str,
        default: Any,
        type: Optional[type] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        multiple: bool = False,
        group: Optional[str] = None,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        ...

    def define(
        self,
        name: str,
        default: Any = None,
        type: Optional[type] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        multiple: bool = False,
        group: Optional[str] = None,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """Defines a new command line option."""
        ...

    def parse_command_line(
        self, args: Optional[List[str]] = None, final: bool = True
    ) -> List[str]:
        """Parses all options given on the command line (defaults to `sys.argv`)."""
        ...

    def parse_config_file(self, path: str, final: bool = True) -> None:
        """Parses and loads the config file at the given path."""
        ...

    def print_help(self, file: Optional[TextIO] = None) -> None:
        """Prints all the command line options to stderr (or another file)."""
        ...

    def _help_callback(self, value: bool) -> None:
        ...

    def add_parse_callback(self, callback: Callable[[], None]) -> None:
        """Adds a parse callback, to be invoked when option parsing is done."""
        ...

    def run_parse_callbacks(self) -> None:
        ...

    def mockable(self) -> "_Mockable":
        """Returns a wrapper around self that is compatible with `unittest.mock.patch`."""
        ...

class _Mockable:
    """`mock.patch` compatible wrapper for `OptionParser`."""

    def __init__(self, options: OptionParser) -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...

    def __delattr__(self, name: str) -> None:
        ...

class _Option:
    UNSET: Any = ...

    def __init__(
        self,
        name: str,
        default: Optional[Any] = None,
        type: Optional[type] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        multiple: bool = False,
        file_name: Optional[str] = None,
        group_name: Optional[str] = None,
        callback: Optional[Callable[[Any], None]] = None,
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

options: OptionParser = ...
"""Global options object.

All defined options are available as attributes on this object.
"""

@overload
def define(
    name: str,
    default: None = None,
    type: Optional[type] = None,
    help: Optional[str] = None,
    metavar: Optional[str] = None,
    multiple: bool = False,
    group: Optional[str] = None,
    callback: Optional[Callable[[Any], None]] = None,
) -> None:
    ...

@overload
def define(
    name: str,
    default: Any,
    type: Optional[type] = None,
    help: Optional[str] = None,
    metavar: Optional[str] = None,
    multiple: bool = False,
    group: Optional[str] = None,
    callback: Optional[Callable[[Any], None]] = None,
) -> None:
    ...

def define(
    name: str,
    default: Any = None,
    type: Optional[type] = None,
    help: Optional[str] = None,
    metavar: Optional[str] = None,
    multiple: bool = False,
    group: Optional[str] = None,
    callback: Optional[Callable[[Any], None]] = None,
) -> None:
    """Defines an option in the global namespace.

    See `OptionParser.define`.
    """
    ...

def parse_command_line(
    args: Optional[List[str]] = None, final: bool = True
) -> List[str]:
    """Parses global options from the command line.

    See `OptionParser.parse_command_line`.
    """
    ...

def parse_config_file(path: str, final: bool = True) -> None:
    """Parses global options from a config file.

    See `OptionParser.parse_config_file`.
    """
    ...

def print_help(file: Optional[TextIO] = None) -> None:
    """Prints all the command line options to stderr (or another file).

    See `OptionParser.print_help`.
    """
    ...

def add_parse_callback(callback: Callable[[], None]) -> None:
    """Adds a parse callback, to be invoked when option parsing is done.

    See `OptionParser.add_parse_callback`
    """
    ...

def define_logging_options(options: OptionParser) -> None:
    ...