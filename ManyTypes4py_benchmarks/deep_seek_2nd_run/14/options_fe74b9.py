"""A command line parsing module that lets modules define their own options.

This module is inspired by Google's `gflags
<https://github.com/google/python-gflags>`_. The primary difference
with libraries such as `argparse` is that a global registry is used so
that options may be defined in any module (it also enables
`tornado.log` by default). The rest of Tornado does not depend on this
module, so feel free to use `argparse` or other configuration
libraries if you prefer them.
"""
import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import Any, Iterator, Iterable, Tuple, Set, Dict, Callable, List, TextIO, Optional, Union, Type, TypeVar, Generic, cast, overload

T = TypeVar('T')

class Error(Exception):
    """Exception raised by errors in the options module."""
    pass

class OptionParser:
    """A collection of options, a dictionary with object-like access.

    Normally accessed via static functions in the `tornado.options` module,
    which reference a global instance.
    """

    def __init__(self) -> None:
        self.__dict__['_options']: Dict[str, _Option] = {}
        self.__dict__['_parse_callbacks']: List[Callable[[], None]] = []
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

    def items(self) -> List[Tuple[str, Any]]:
        """An iterable of (name, value) pairs.

        .. versionadded:: 3.1
        """
        return [(opt.name, opt.value()) for name, opt in self._options.items()]

    def groups(self) -> Set[str]:
        """The set of option-groups created by ``define``.

        .. versionadded:: 3.1
        """
        return {opt.group_name for opt in self._options.values()}

    def group_dict(self, group: Optional[str]) -> Dict[str, Any]:
        """The names and values of options in a group.

        Useful for copying options into Application settings::

            from tornado.options import define, parse_command_line, options

            define('template_path', group='application')
            define('static_path', group='application')

            parse_command_line()

            application = Application(
                handlers, **options.group_dict('application'))

        .. versionadded:: 3.1
        """
        return {opt.name: opt.value() for name, opt in self._options.items() if not group or group == opt.group_name}

    def as_dict(self) -> Dict[str, Any]:
        """The names and values of all options.

        .. versionadded:: 3.1
        """
        return {opt.name: opt.value() for name, opt in self._options.items()}

    def define(self, name: str, default: Any = None, type: Optional[Type] = None, help: Optional[str] = None, 
               metavar: Optional[str] = None, multiple: bool = False, group: Optional[str] = None, 
               callback: Optional[Callable[[Any], None]] = None) -> None:
        """Defines a new command line option."""
        normalized = self._normalize_name(name)
        if normalized in self._options:
            raise Error('Option %r already defined in %s' % (normalized, self._options[normalized].file_name))
        frame = sys._getframe(0)
        if frame is not None:
            options_file = frame.f_code.co_filename
            if frame.f_back is not None and frame.f_back.f_code.co_filename == options_file and (frame.f_back.f_code.co_name == 'define'):
                frame = frame.f_back
            assert frame.f_back is not None
            file_name = frame.f_back.f_code.co_filename
        else:
            file_name = '<unknown>'
        if file_name == options_file:
            file_name = ''
        if type is None:
            if not multiple and default is not None:
                type = default.__class__
            else:
                type = str
        if group:
            group_name = group
        else:
            group_name = file_name
        option = _Option(name, file_name=file_name, default=default, type=type, help=help, metavar=metavar, multiple=multiple, group_name=group_name, callback=callback)
        self._options[normalized] = option

    def parse_command_line(self, args: Optional[List[str]] = None, final: bool = True) -> List[str]:
        """Parses all options given on the command line (defaults to
        `sys.argv`)."""
        if args is None:
            args = sys.argv
        remaining = []
        for i in range(1, len(args)):
            if not args[i].startswith('-'):
                remaining = args[i:]
                break
            if args[i] == '--':
                remaining = args[i + 1:]
                break
            arg = args[i].lstrip('-')
            name, equals, value = arg.partition('=')
            name = self._normalize_name(name)
            if name not in self._options:
                self.print_help()
                raise Error('Unrecognized command line option: %r' % name)
            option = self._options[name]
            if not equals:
                if option.type == bool:
                    value = 'true'
                else:
                    raise Error('Option %r requires a value' % name)
            option.parse(value)
        if final:
            self.run_parse_callbacks()
        return remaining

    def parse_config_file(self, path: str, final: bool = True) -> None:
        """Parses and loads the config file at the given path."""
        config = {'__file__': os.path.abspath(path)}
        with open(path, 'rb') as f:
            exec_in(native_str(f.read()), config, config)
        for name in config:
            normalized = self._normalize_name(name)
            if normalized in self._options:
                option = self._options[normalized]
                if option.multiple:
                    if not isinstance(config[name], (list, str)):
                        raise Error('Option %r is required to be a list of %s or a comma-separated string' % (option.name, option.type.__name__))
                if type(config[name]) is str and (option.type is not str or option.multiple):
                    option.parse(config[name])
                else:
                    option.set(config[name])
        if final:
            self.run_parse_callbacks()

    def print_help(self, file: Optional[TextIO] = None) -> None:
        """Prints all the command line options to stderr (or another file)."""
        if file is None:
            file = sys.stderr
        print('Usage: %s [OPTIONS]' % sys.argv[0], file=file)
        print('\nOptions:\n', file=file)
        by_group: Dict[str, List[_Option]] = {}
        for option in self._options.values():
            by_group.setdefault(option.group_name, []).append(option)
        for filename, o in sorted(by_group.items()):
            if filename:
                print('\n%s options:\n' % os.path.normpath(filename), file=file)
            o.sort(key=lambda option: option.name)
            for option in o:
                prefix = self._normalize_name(option.name)
                if option.metavar:
                    prefix += '=' + option.metavar
                description = option.help or ''
                if option.default is not None and option.default != '':
                    description += ' (default %s)' % option.default
                lines = textwrap.wrap(description, 79 - 35)
                if len(prefix) > 30 or len(lines) == 0:
                    lines.insert(0, '')
                print('  --%-30s %s' % (prefix, lines[0]), file=file)
                for line in lines[1:]:
                    print('%-34s %s' % (' ', line), file=file)
        print(file=file)

    def _help_callback(self, value: bool) -> None:
        if value:
            self.print_help()
            sys.exit(0)

    def add_parse_callback(self, callback: Callable[[], None]) -> None:
        """Adds a parse callback, to be invoked when option parsing is done."""
        self._parse_callbacks.append(callback)

    def run_parse_callbacks(self) -> None:
        for callback in self._parse_callbacks:
            callback()

    def mockable(self) -> '_Mockable':
        """Returns a wrapper around self that is compatible with
        `unittest.mock.patch`."""
        return _Mockable(self)

class _Mockable:
    """`mock.patch` compatible wrapper for `OptionParser`."""

    def __init__(self, options: OptionParser) -> None:
        self.__dict__['_options'] = options
        self.__dict__['_originals']: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._options, name)

    def __setattr__(self, name: str, value: Any) -> None:
        assert name not in self._originals, "don't reuse mockable objects"
        self._originals[name] = getattr(self._options, name)
        setattr(self._options, name, value)

    def __delattr__(self, name: str) -> None:
        setattr(self._options, name, self._originals.pop(name))

class _Option:
    UNSET = object()

    def __init__(self, name: str, default: Any = None, type: Optional[Type] = None, help: Optional[str] = None, 
                 metavar: Optional[str] = None, multiple: bool = False, file_name: Optional[str] = None, 
                 group_name: Optional[str] = None, callback: Optional[Callable[[Any], None]] = None) -> None:
        if default is None and multiple:
            default = []
        self.name = name
        if type is None:
            raise ValueError('type must not be None')
        self.type = type
        self.help = help
        self.metavar = metavar
        self.multiple = multiple
        self.file_name = file_name
        self.group_name = group_name
        self.callback = callback
        self.default = default
        self._value: Any = _Option.UNSET

    def value(self) -> Any:
        return self.default if self._value is _Option.UNSET else self._value

    def parse(self, value: str) -> Any:
        _parse = {datetime.datetime: self._parse_datetime, datetime.timedelta: self._parse_timedelta, bool: self._parse_bool, basestring_type: self._parse_string}.get(self.type, self.type)
        if self.multiple:
            self._value = []
            for part in value.split(','):
                if issubclass(self.type, numbers.Integral):
                    lo_str, _, hi_str = part.partition(':')
                    lo = _parse(lo_str)
                    hi = _parse(hi_str) if hi_str else lo
                    self._value.extend(range(lo, hi + 1))
                else:
                    self._value.append(_parse(part))
        else:
            self._value = _parse(value)
        if self.callback is not None:
            self.callback(self._value)
        return self.value()

    def set(self, value: Any) -> None:
        if self.multiple:
            if not isinstance(value, list):
                raise Error('Option %r is required to be a list of %s' % (self.name, self.type.__name__))
            for item in value:
                if item is not None and (not isinstance(item, self.type)):
                    raise Error('Option %r is required to be a list of %s' % (self.name, self.type.__name__))
        elif value is not None and (not isinstance(value, self.type)):
            raise Error('Option %r is required to be a %s (%s given)' % (self.name, self.type.__name__, type(value)))
        self._value = value
        if self.callback is not None:
            self.callback(self._value)

    _DATETIME_FORMATS = ['%a %b %d %H:%M:%S %Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M', '%Y-%m-%d', '%Y%m%d', '%H:%M:%S', '%H:%M']

    def _parse_datetime(self, value: str) -> datetime.datetime:
        for format in self._DATETIME_FORMATS:
            try:
                return datetime.datetime.strptime(value, format)
            except ValueError:
                pass
        raise Error('Unrecognized date/time format: %r' % value)

    _TIMEDELTA_ABBREV_DICT = {'h': 'hours', 'm': 'minutes', 'min': 'minutes', 's': 'seconds', 'sec': 'seconds', 'ms': 'milliseconds', 'us': 'microseconds', 'd': 'days', 'w': 'weeks'}
    _FLOAT_PATTERN = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'
    _TIMEDELTA_PATTERN = re.compile('\\s*(%s)\\s*(\\w*)\\s*' % _FLOAT_PATTERN, re.IGNORECASE)

    def _parse_timedelta(self, value: str) -> datetime.timedelta:
        try:
            sum = datetime.timedelta()
            start = 0
            while start < len(value):
                m = self._TIMEDELTA_PATTERN.match(value, start)
                if not m:
                    raise Exception()
                num = float(m.group(1))
                units = m.group(2) or 'seconds'
                units = self._TIMEDELTA_ABBREV_DICT.get(units, units)
                sum += datetime.timedelta(**{units: num})
                start = m.end()
            return sum
        except Exception:
            raise

    def _parse_bool(self, value: str) -> bool:
        return value.lower() not in ('false', '0', 'f')

    def _parse_string(self, value: str) -> str:
        return _unicode(value)

options: OptionParser = OptionParser()
'Global options object.\n\nAll defined options are available as attributes on this object.\n'

def define(name: str, default: Any = None, type: Optional[Type] = None, help: Optional[str] = None, 
           metavar: Optional[str] = None, multiple: bool = False, group: Optional[str] = None, 
           callback: Optional[Callable[[Any], None]] = None) -> None:
    """Defines an option in the global namespace."""
    return options.define(name, default=default, type=type, help=help, metavar=metavar, multiple=multiple, group=group, callback=callback)

def parse_command_line(args: Optional[List[str]] = None, final: bool = True) -> List[str]:
    """Parses global options from the command line."""
    return options.parse_command_line(args, final=final)

def parse_config_file(path: str, final: bool = True) -> None:
    """Parses global options from a config file."""
    return options.parse_config_file(path, final=final)

def print_help(file: Optional[TextIO] = None) -> None:
    """Prints all the command line options to stderr (or another file)."""
    return options.print_help(file)

def add_parse_callback(callback: Callable[[], None]) -> None:
    """Adds a parse callback, to be invoked when option parsing is done."""
    options.add_parse_callback(callback)

define_logging_options(options)
