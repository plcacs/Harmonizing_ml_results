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
        self.__dict__['_options'] = {}  # type: Dict[str, _Option]
        self.__dict__['_parse_callbacks'] = []  # type: List[Callable[[], None]]
        self.define('help', default=None, type=bool, help='show this help information', callback=self._help_callback)

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
        return {opt.group_name for opt in self._options.values() if opt.group_name}

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
        return {opt.name: opt.value() for opt in self._options.values()}

    def define(self, name: str, default: Any = None, type: Optional[type] = None, help: Optional[str] = None,
               metavar: Optional[str] = None, multiple: bool = False, group: Optional[str] = None,
               callback: Optional[Callable[[Any], None]] = None) -> None:
        """Defines a new command line option.

        ``type`` can be any of `str`, `int`, `float`, `bool`,
        `~datetime.datetime`, or `~datetime.timedelta`. If no ``type``
        is given but a ``default`` is, ``type`` is the type of
        ``default``. Otherwise, ``type`` defaults to `str`.

        If ``multiple`` is True, the option value is a list of ``type``
        instead of an instance of ``type``.

        ``help`` and ``metavar`` are used to construct the
        automatically generated command line help string. The help
        message is formatted like::

           --name=METAVAR      help string

        ``group`` is used to group the defined options in logical
        groups. By default, command line options are grouped by the
        file in which they are defined.

        Command line option names must be unique globally.

        If a ``callback`` is given, it will be run with the new value whenever
        the option is changed.  This can be used to combine command-line
        and file-based options::

            define("config", type=str, help="path to config file",
                   callback=lambda path: parse_config_file(path, final=False))

        With this definition, options in the file specified by ``--config`` will
        override options set earlier on the command line, but can be overridden
        by later flags.
        """
        normalized = self._normalize_name(name)
        if normalized in self._options:
            raise Error('Option %r already defined in %s' % (normalized, self._options[normalized].file_name))
        frame = sys._getframe(0)
        if frame is not None:
            options_file: str = frame.f_code.co_filename  # type: ignore
            if frame.f_back is not None and frame.f_back.f_code.co_filename == options_file and (frame.f_back.f_code.co_name == 'define'):
                frame = frame.f_back
            assert frame.f_back is not None
            file_name: str = frame.f_back.f_code.co_filename  # type: ignore
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
            group_name: str = group
        else:
            group_name = file_name
        option = _Option(name, file_name=file_name, default=default, type=type, help=help,
                           metavar=metavar, multiple=multiple, group_name=group_name, callback=callback)
        self._options[normalized] = option

    def parse_command_line(self, args: Optional[List[str]] = None, final: bool = True) -> List[str]:
        """Parses all options given on the command line (defaults to
        `sys.argv`).

        Options look like ``--option=value`` and are parsed according
        to their ``type``. For boolean options, ``--option`` is
        equivalent to ``--option=true``

        If the option has ``multiple=True``, comma-separated values
        are accepted. For multi-value integer options, the syntax
        ``x:y`` is also accepted and equivalent to ``range(x, y)``.

        Note that ``args[0]`` is ignored since it is the program name
        in `sys.argv`.

        We return a list of all arguments that are not parsed as options.

        If ``final`` is ``False``, parse callbacks will not be run.
        This is useful for applications that wish to combine configurations
        from multiple sources.
        """
        if args is None:
            args = sys.argv  # type: List[str]
        remaining: List[str] = []
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
        """Parses and loads the config file at the given path.

        The config file contains Python code that will be executed (so
        it is **not safe** to use untrusted config files). Anything in
        the global namespace that matches a defined option will be
        used to set that option's value.

        Options may either be the specified type for the option or
        strings (in which case they will be parsed the same way as in
        `.parse_command_line`)

        Example (using the options defined in the top-level docs of
        this module)::

            port = 80
            mysql_host = 'mydb.example.com:3306'
            # Both lists and comma-separated strings are allowed for
            # multiple=True.
            memcache_hosts = ['cache1.example.com:11011',
                              'cache2.example.com:11011']
            memcache_hosts = 'cache1.example.com:11011,cache2.example.com:11011'

        If ``final`` is ``False``, parse callbacks will not be run.
        This is useful for applications that wish to combine configurations
        from multiple sources.
        """
        config: Dict[str, Any] = {'__file__': os.path.abspath(path)}
        with open(path, 'rb') as f:
            exec_in(native_str(f.read()), config, config)
        for name in config:
            normalized = self._normalize_name(name)
            if normalized in self._options:
                option = self._options[normalized]
                if option.multiple:
                    if not isinstance(config[name], (list, str)):
                        raise Error('Option %r is required to be a list of %s or a comma-separated string' %
                                    (option.name, option.type.__name__))
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

    def _help_callback(self, value: Any) -> None:
        if value:
            self.print_help()
            sys.exit(0)

    def add_parse_callback(self, callback: Callable[[], None]) -> None:
        """Adds a parse callback, to be invoked when option parsing is done."""
        self._parse_callbacks.append(callback)

    def run_parse_callbacks(self) -> None:
        for callback in self._parse_callbacks:
            callback()

    def mockable(self) -> " _Mockable":
        """Returns a wrapper around self that is compatible with
        `unittest.mock.patch`.

        The `unittest.mock.patch` function is incompatible with objects like ``options`` that
        override ``__getattr__`` and ``__setattr__``.  This function returns an object that can be
        used with `mock.patch.object <unittest.mock.patch.object>` to modify option values::

            with mock.patch.object(options.mockable(), 'name', value):
                assert options.name == value
        """
        return _Mockable(self)

class _Mockable:
    """`mock.patch` compatible wrapper for `OptionParser`.

    As of ``mock`` version 1.0.1, when an object uses ``__getattr__``
    hooks instead of ``__dict__``, ``patch.__exit__`` tries to delete
    the attribute it set instead of setting a new one (assuming that
    the object does not capture ``__setattr__``, so the patch
    created a new attribute in ``__dict__``).

    _Mockable's getattr and setattr pass through to the underlying
    OptionParser, and delattr undoes the effect of a previous setattr.
    """

    def __init__(self, options: OptionParser) -> None:
        self.__dict__['_options'] = options
        self.__dict__['_originals'] = {}  # type: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._options, name)

    def __setattr__(self, name: str, value: Any) -> None:
        assert name not in self._originals, "don't reuse mockable objects"
        self._originals[name] = getattr(self._options, name)
        setattr(self._options, name, value)

    def __delattr__(self, name: str) -> None:
        setattr(self._options, name, self._originals.pop(name))

class _Option:
    UNSET: Any = object()

    def __init__(self, name: str, default: Any = None, type: Optional[type] = None, help: Optional[str] = None,
                 metavar: Optional[str] = None, multiple: bool = False, file_name: Optional[str] = None,
                 group_name: Optional[str] = None, callback: Optional[Callable[[Any], None]] = None) -> None:
        if default is None and multiple:
            default = []
        self.name: str = name
        if type is None:
            raise ValueError('type must not be None')
        self.type: type = type
        self.help: Optional[str] = help
        self.metavar: Optional[str] = metavar
        self.multiple: bool = multiple
        self.file_name: Optional[str] = file_name
        self.group_name: Optional[str] = group_name
        self.callback: Optional[Callable[[Any], None]] = callback
        self.default: Any = default
        self._value: Any = _Option.UNSET

    def value(self) -> Any:
        return self.default if self._value is _Option.UNSET else self._value

    def parse(self, value: str) -> Any:
        _parse: Callable[[str], Any]
        if self.type in (datetime.datetime, datetime.timedelta, bool, basestring_type):
            _parse = {datetime.datetime: self._parse_datetime,
                      datetime.timedelta: self._parse_timedelta,
                      bool: self._parse_bool,
                      basestring_type: self._parse_string}[self.type]
        else:
            _parse = self.type  # type: ignore
        if self.multiple:
            self._value = []  # type: List[Any]
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

    _DATETIME_FORMATS: List[str] = ['%a %b %d %H:%M:%S %Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                                    '%Y-%m-%dT%H:%M', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M', '%Y-%m-%d', '%Y%m%d',
                                    '%H:%M:%S', '%H:%M']

    def _parse_datetime(self, value: str) -> datetime.datetime:
        for fmt in self._DATETIME_FORMATS:
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                pass
        raise Error('Unrecognized date/time format: %r' % value)

    _TIMEDELTA_ABBREV_DICT: Dict[str, str] = {
        'h': 'hours', 'm': 'minutes', 'min': 'minutes',
        's': 'seconds', 'sec': 'seconds', 'ms': 'milliseconds',
        'us': 'microseconds', 'd': 'days', 'w': 'weeks'
    }
    _FLOAT_PATTERN: str = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'
    _TIMEDELTA_PATTERN: re.Pattern = re.compile('\\s*(%s)\\s*(\\w*)\\s*' % _FLOAT_PATTERN, re.IGNORECASE)

    def _parse_timedelta(self, value: str) -> datetime.timedelta:
        try:
            total = datetime.timedelta()
            start = 0
            while start < len(value):
                m = self._TIMEDELTA_PATTERN.match(value, start)
                if not m:
                    raise Exception()
                num = float(m.group(1))
                units = m.group(2) or 'seconds'
                units = self._TIMEDELTA_ABBREV_DICT.get(units, units)
                total += datetime.timedelta(**{units: num})
                start = m.end()
            return total
        except Exception:
            raise Error('Unrecognized time delta format: %r' % value)

    def _parse_bool(self, value: str) -> bool:
        return value.lower() not in ('false', '0', 'f')

    def _parse_string(self, value: str) -> str:
        return _unicode(value)

options: OptionParser = OptionParser()
'Global options object.\n\nAll defined options are available as attributes on this object.\n'

def define(name: str, default: Any = None, type: Optional[type] = None, help: Optional[str] = None,
           metavar: Optional[str] = None, multiple: bool = False, group: Optional[str] = None,
           callback: Optional[Callable[[Any], None]] = None) -> None:
    """Defines an option in the global namespace.

    See `OptionParser.define`.
    """
    return options.define(name, default=default, type=type, help=help, metavar=metavar,
                          multiple=multiple, group=group, callback=callback)

def parse_command_line(args: Optional[List[str]] = None, final: bool = True) -> List[str]:
    """Parses global options from the command line.

    See `OptionParser.parse_command_line`.
    """
    return options.parse_command_line(args, final=final)

def parse_config_file(path: str, final: bool = True) -> None:
    """Parses global options from a config file.

    See `OptionParser.parse_config_file`.
    """
    return options.parse_config_file(path, final=final)

def print_help(file: Optional[TextIO] = None) -> None:
    """Prints all the command line options to stderr (or another file).

    See `OptionParser.print_help`.
    """
    return options.print_help(file)

def add_parse_callback(callback: Callable[[], None]) -> None:
    """Adds a parse callback, to be invoked when option parsing is done.

    See `OptionParser.add_parse_callback`
    """
    options.add_parse_callback(callback)

define_logging_options(options)