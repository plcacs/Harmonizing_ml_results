"""A command line parsing module that lets modules define their own options.

This module is inspired by Google's `gflags
<https://github.com/google/python-gflags>`_. The primary difference
with libraries such as `argparse` is that a global registry is used so
that options may be defined in any module (it also enables
`tornado.log` by default). The rest of Tornado does not depend on this
module, so feel free to use `argparse` or other configuration
libraries if you prefer them.

Options must be defined with `tornado.options.define` before use,
generally at the top level of a module. The options are then
accessible as attributes of `tornado.options.options`::

    # myapp/db.py
    from tornado.options import define, options

    define("mysql_host", default="127.0.0.1:3306", help="Main user DB")
    define("memcache_hosts", default="127.0.0.1:11011", multiple=True,
           help="Main user memcache servers")

    def connect():
        db = database.Connection(options.mysql_host)
        ...

    # myapp/server.py
    from tornado.options import define, options

    define("port", default=8080, help="port to listen on")

    def start_server():
        app = make_app()
        app.listen(options.port)

The ``main()`` method of your application does not need to be aware of all of
the options used throughout your program; they are all automatically loaded
when the modules are loaded.  However, all modules that define options
must have been imported before the command line is parsed.

Your ``main()`` method can parse the command line or parse a config file with
either `parse_command_line` or `parse_config_file`::

    import myapp.db, myapp.server
    import tornado

    if __name__ == '__main__':
        tornado.options.parse_command_line()
        # or
        tornado.options.parse_config_file("/etc/server.conf")

.. note::

   When using multiple ``parse_*`` functions, pass ``final=False`` to all
   but the last one, or side effects may occur twice (in particular,
   this can result in log messages being doubled).

`tornado.options.options` is a singleton instance of `OptionParser`, and
the top-level functions in this module (`define`, `parse_command_line`, etc)
simply call methods on it.  You may create additional `OptionParser`
instances to define isolated sets of options, such as for subcommands.

.. note::

   By default, several options are defined that will configure the
   standard `logging` module when `parse_command_line` or `parse_config_file`
   are called.  If you want Tornado to leave the logging configuration
   alone so you can manage it yourself, either pass ``--logging=none``
   on the command line or do the following to disable it in code::

       from tornado.options import options, parse_command_line
       options.logging = None
       parse_command_line()

.. note::

   `parse_command_line` or `parse_config_file` function should called after
   logging configuration and user-defined command line flags using the
   ``callback`` option definition, or these configurations will not take effect.

.. versionchanged:: 4.3
   Dashes and underscores are fully interchangeable in option names;
   options can be defined, set, and read with any mix of the two.
   Dashes are typical for command-line usage while config files require
   underscores.
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
from typing import Any, Iterator, Iterable, Tuple, Set, Dict, Callable, List, TextIO, Optional, Union, Type, TypeVar, cast

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

    def group_dict(self, group: Optional[str] = None) -> Dict[str, Any]:
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

        .. note::

            `tornado.options` is primarily a command-line library.
            Config file support is provided for applications that wish
            to use it, but applications that prefer config files may
            wish to look at other libraries instead.

        .. versionchanged:: 4.1
           Config files are now always interpreted as utf-8 instead of
           the system default encoding.

        .. versionchanged:: 4.4
           The special variable ``__file__`` is available inside config
           files, specifying the absolute path to the config file itself.

        .. versionchanged:: 5.1
           Added the ability to set options via strings in config files.

        """
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
        by_group = {}
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

    def __init__(self, name: str