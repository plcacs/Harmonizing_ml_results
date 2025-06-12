"""Command-line programs using :pypi:`click`."""
import abc
import asyncio
import inspect
import io
import os
import sys
import typing
from functools import wraps
from pathlib import Path
from textwrap import wrap
from types import ModuleType
from typing import Any, Awaitable, Callable, ClassVar, Dict, IO, List, Mapping, MutableSequence, Optional, Sequence, Tuple, Type, Union, cast, no_type_check
import click
from click import echo
from colorclass import Color, disable_all_colors, enable_all_colors
from mode import Service, ServiceT, Worker
from mode.utils import text
from mode.utils.compat import want_bytes
from mode.utils.imports import import_from_cwd, symbol_by_name
from mode.utils.typing import NoReturn
from mode.worker import exiting
from faust.types._env import CONSOLE_PORT, DATADIR, DEBUG, WORKDIR
from faust.types import AppT, CodecArg, ModelT
from faust.utils import json
from faust.utils import terminal
from faust.utils.codegen import reprcall
from . import params
if typing.TYPE_CHECKING:
    from faust.app import App as _App
else:

    class _App:
        ...
try:
    import click_completion
except ImportError:
    click_completion = None
else:
    click_completion.init()
__all__ = ['AppCommand', 'Command', 'argument', 'cli', 'find_app', 'option']

class argument:
    """Create command-line argument.

    SeeAlso:
        :func:`click.argument`
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.argument = click.argument(*self.args, **self.kwargs)

    def __call__(self, fun):
        return self.argument(fun)

    def __repr__(self):
        return reprcall(type(self).__name__, self.args, self.kwargs)

class option:
    """Create command-line option.

    SeeAlso:
        :func:`click.option`
    """

    def __init__(self, *args, show_default=True, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.option = click.option(*args, show_default=show_default, **kwargs)

    def __call__(self, fun):
        return self.option(fun)

    def __repr__(self):
        return reprcall(type(self).__name__, self.args, self.kwargs)
OptionDecorator = Callable[[Any], Any]
OptionSequence = Sequence[OptionDecorator]
OptionList = MutableSequence[OptionDecorator]
LOOP_CHOICES = ('aio', 'eventlet', 'uvloop')
DEFAULT_LOOP = 'aio'
DEFAULT_LOGLEVEL = 'WARN'
LOGLEVELS = ('CRIT', 'ERROR', 'WARN', 'INFO', 'DEBUG')
faust_version = symbol_by_name('faust:__version__')

class State:
    app = None
    quiet = False
    debug = False
    workdir = None
    datadir = None
    json = False
    no_color = False
    loop = None
    logfile = None
    loglevel = None
    blocking_timeout = None
    console_port = None

def compat_option(*args, state_key, callback=None, expose_value=False, **kwargs):

    def _callback(ctx, param, value):
        state = ctx.ensure_object(State)
        prev_value = getattr(state, state_key, None)
        if prev_value is None and value != param.default:
            setattr(state, state_key, value)
        return callback(ctx, param, value) if callback else value
    return option(*args, callback=_callback, expose_value=expose_value, **kwargs)
now_builtin_worker_options = [compat_option('--logfile', '-f', state_key='logfile', default=None, type=params.WritableFilePath, help='Path to logfile (default is <stderr>).'), compat_option('--loglevel', '-l', state_key='loglevel', default=DEFAULT_LOGLEVEL, type=params.CaseInsensitiveChoice(LOGLEVELS), help='Logging level to use.'), compat_option('--blocking-timeout', state_key='blocking_timeout', default=None, type=float, help='when --debug: Blocking detector timeout.'), compat_option('--console-port', state_key='console_port', default=CONSOLE_PORT, type=params.TCPPort(), help='when --debug: Port to run debugger console on.')]
core_options = [click.version_option(version=f'Faust {faust_version}'), option('--app', '-A', help='Path of Faust application to use, or the name of a module.'), option('--quiet/--no-quiet', '-q', default=False, help='Silence output to <stdout>/<stderr>.'), option('--debug/--no-debug', default=DEBUG, help='Enable debugging output, and the blocking detector.'), option('--no-color/--color', '--no_color/--color', default=False, help='Enable colors in output.'), option('--workdir', '-W', default=WORKDIR, type=params.WritableDirectory, help='Working directory to change to after start.'), option('--datadir', '-D', default=DATADIR, type=params.WritableDirectory, help='Directory to keep application state.'), option('--json', default=False, is_flag=True, help='Return output in machine-readable JSON format'), option('--loop', '-L', default=DEFAULT_LOOP, type=click.Choice(LOOP_CHOICES), help='Event loop implementation to use.')]
builtin_options = cast(List, core_options) + cast(List, now_builtin_worker_options)

class _FaustRootContextT(click.Context):
    pass

def find_app(app, *, symbol_by_name=symbol_by_name, imp=import_from_cwd, attr_name='app'):
    """Find app by string like ``examples.simple``.

    Notes:
        This function uses ``import_from_cwd`` to temporarily
        add the current working directory to :envvar:`PYTHONPATH`,
        such that when importing the app it will search the current
        working directory last.

        You can think of it as temporarily
        running with the :envvar:`PYTHONPATH` set like this:

        .. sourcecode: console

            $ PYTHONPATH="${PYTHONPATH}:."

        You can disable this with the ``imp`` keyword argument,
        for example passing ``imp=importlib.import_module``.

    Examples:
        >>> # If providing the name of a module, it will attempt
        >>> # to find an attribute name (.app) in that module.
        >>> # Example below is the same as importing::
        >>> #    from examples.simple import app
        >>> find_app('examples.simple')

        >>> # If you want an attribute other than .app you can
        >>> # use : to separate module and attribute.
        >>> # Examples below is the same as importing::
        >>> #     from examples.simple import my_app
        >>> find_app('examples.simple:my_app')

        >>> # You can also use period for the module/attribute separator
        >>> find_app('examples.simple.my_app')
    """
    try:
        val = symbol_by_name(app, imp=imp)
    except AttributeError:
        val = imp(app)
    if isinstance(val, ModuleType) and ':' not in app:
        found = getattr(val, attr_name)
        if isinstance(found, ModuleType):
            raise AttributeError(f'Looks like module, not app: -A {app}')
        val = found
    return prepare_app(val, app)

def prepare_app(app, name):
    app.finalize()
    if app.conf._origin is None:
        app.conf._origin = name
    app.worker_init()
    if app.conf.autodiscover:
        app.discover()
    app.worker_init_post_autodiscover()
    if 1:
        main = sys.modules.get('__main__')
        if main is not None and 'cProfile.py' in getattr(main, '__file__', ''):
            from ..models import registry
            registry.update({(app.conf.origin or '') + k[8:]: v for k, v in registry.items() if k.startswith('cProfile.')})
        return app

def _apply_options(options):
    """Add list of ``click.option`` values to click command function."""

    def _inner(fun):
        for opt in options:
            fun = opt(fun)
        return fun
    return _inner

class _Group(click.Group):

    def get_help(self, ctx):
        self._maybe_import_app()
        return super().get_help(ctx)

    def get_usage(self, ctx):
        self._maybe_import_app()
        return super().get_usage(ctx)

    def _maybe_import_app(self, argv=sys.argv):
        workdir = self._extract_param(argv, '-W', '--workdir')
        if workdir:
            os.chdir(Path(workdir).absolute())
        appstr = self._extract_param(argv, '-A', '--app')
        if appstr is not None:
            find_app(appstr)

    def _extract_param(self, argv, shortopt, longopt):
        for i, arg in enumerate(argv):
            if arg == shortopt:
                try:
                    return argv[i + 1]
                except IndexError:
                    raise click.UsageError(f'Missing argument for {shortopt}')
            elif arg.startswith(longopt):
                if '=' in arg:
                    _, _, value = arg.partition('=')
                    return value
                else:
                    try:
                        return argv[i + 1]
                    except IndexError:
                        raise click.UsageError(f'Missing argument for {longopt}')
        return None

    @no_type_check
    def make_context(self, info_name, args, app=None, parent=None, stdout=None, stderr=None, side_effects=True, **extra):
        ctx = super().make_context(info_name, args, **extra)
        self._maybe_import_app()
        root = cast(_FaustRootContextT, ctx.find_root())
        root.app = app
        root.stdout = stdout
        root.stderr = stderr
        root.side_effects = side_effects
        return ctx

@click.group(cls=_Group)
@_apply_options(builtin_options)
@click.pass_context
def cli(*args, **kwargs):
    """Welcome, see list of commands and options below.

    Use --help for help, --version for version information.

    https://faust.readthedocs.io
    """
    return _prepare_cli(*args, **kwargs)

def _prepare_cli(ctx, app, quiet, debug, workdir, datadir, json, no_color, loop):
    """Faust command-line interface."""
    state = ctx.ensure_object(State)
    state.app = app
    state.quiet = quiet
    state.debug = debug
    state.workdir = workdir
    state.datadir = datadir
    state.json = json
    state.no_color = no_color
    state.loop = loop
    root = cast(_FaustRootContextT, ctx.find_root())
    if root.side_effects:
        if workdir:
            os.environ['F_WORKDIR'] = workdir
            os.chdir(Path(workdir).absolute())
        if datadir:
            os.environ['F_DATADIR'] = datadir
        if not no_color and terminal.isatty(sys.stdout):
            enable_all_colors()
        else:
            disable_all_colors()
        if json:
            disable_all_colors()

class Command(abc.ABC):
    """Base class for subcommands."""
    UsageError = click.UsageError
    abstract = True
    _click = None
    daemon = False
    redirect_stdouts = None
    redirect_stdouts_level = None
    builtin_options = builtin_options
    options = None
    prog_name = ''

    @classmethod
    def as_click_command(cls):
        """Convert command into :pypi:`click` command."""

        @click.pass_context
        @wraps(cls)
        def _inner(*args, **kwargs):
            cmd = cls(*args, **kwargs)
            with exiting(print_exception=True, file=sys.stderr):
                cmd()
        return _apply_options(cls.options or [])(cli.command(help=cls.__doc__)(_inner))

    def __init_subclass__(self, *args, **kwargs):
        if self.abstract:
            self.abstract = False
        else:
            self._click = self.as_click_command()
        _apply_options(self.builtin_options)(self._parse)
        _apply_options(self.options or [])(self._parse)

    @classmethod
    def parse(cls, argv):
        """Parse command-line arguments in ``argv`` and return mapping."""
        return cls._parse(argv, standalone_mode=False)

    @staticmethod
    @click.command()
    def _parse(**kwargs):
        return kwargs

    def __init__(self, ctx, *args, **kwargs):
        self.ctx = ctx
        root = cast(_FaustRootContextT, self.ctx.find_root())
        self.state = ctx.ensure_object(State)
        self.debug = self.state.debug
        self.quiet = self.state.quiet
        self.workdir = self.state.workdir
        self.datadir = self.state.datadir
        self.json = self.state.json
        self.no_color = self.state.no_color
        self.logfile = self.state.logfile
        self.stdout = root.stdout or sys.stdout
        self.stderr = root.stderr or sys.stderr
        self.args = args
        self.kwargs = kwargs
        self.prog_name = root.command_path
        self._loglevel = self.state.loglevel
        self._blocking_timeout = self.state.blocking_timeout
        self._console_port = self.state.console_port

    @no_type_check
    async def run(self, *args, **kwargs):
        """Override this method to define what your command does."""
        ...

    async def execute(self, *args, **kwargs):
        """Execute command."""
        try:
            await self.run(*args, **kwargs)
        finally:
            await self.on_stop()

    async def on_stop(self):
        """Call after command executed."""
        ...

    def __call__(self, *args, **kwargs):
        """Call command-line command.

        This will raise :exc:`SystemExit` before returning,
        and the exit code will be set accordingly.
        """
        self.run_using_worker(*args, **kwargs)

    def run_using_worker(self, *args, **kwargs):
        """Execute command using :class:`faust.Worker`."""
        loop = asyncio.get_event_loop()
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        service = self.as_service(loop, *args, **kwargs)
        worker = self.worker_for_service(service, loop)
        self.on_worker_created(worker)
        raise worker.execute_from_commandline()

    def on_worker_created(self, worker):
        """Call when creating :class:`faust.Worker` to execute this command."""
        ...

    def as_service(self, loop, *args, **kwargs):
        """Wrap command in a :class:`mode.Service` object."""
        return Service.from_awaitable(self.execute(*args, **kwargs), name=type(self).__name__, loop=loop or asyncio.get_event_loop())

    def worker_for_service(self, service, loop=None):
        """Create :class:`faust.Worker` instance for this command."""
        return self._Worker(service, debug=self.debug, quiet=self.quiet, stdout=self.stdout, stderr=self.stderr, loglevel=self.loglevel, logfile=self.logfile, blocking_timeout=self.blocking_timeout, console_port=self.console_port, redirect_stdouts=self.redirect_stdouts or False, redirect_stdouts_level=self.redirect_stdouts_level, loop=loop or asyncio.get_event_loop(), daemon=self.daemon)

    @property
    def _Worker(self):
        return Worker

    def tabulate(self, data, headers=None, wrap_last_row=True, title='', title_color='blue', **kwargs):
        """Create an ANSI representation of a table of two-row tuples.

        See Also:
            Keyword arguments are forwarded to
            :class:`terminaltables.SingleTable`

        Note:
            If the :option:`--json <faust --json>` option is enabled
            this returns json instead.
        """
        if self.json:
            return self._tabulate_json(data, headers=headers)
        if headers:
            data = [headers] + list(data)
        title = self.bold(self.color(title_color, title))
        table = self.table(data, title=title, **kwargs)
        if wrap_last_row:
            data = [list(item[:-1]) + [self._table_wrap(table, item[-1])] for item in data]
        return table.table

    def _tabulate_json(self, data, headers=None):
        if headers:
            return json.dumps([dict(zip(headers, row)) for row in data])
        return json.dumps(data)

    def table(self, data, title='', **kwargs):
        """Format table data as ANSI/ASCII table."""
        return terminal.table(data, title=title, target=sys.stdout, **kwargs)

    def color(self, name, text):
        """Return text having a certain color by name.

        Examples::
            >>> self.color('blue', 'text_to_color')
            >>> self.color('hiblue', text_to_color')

        See Also:
            :pypi:`colorclass`: for a list of available colors.
        """
        return Color(f'{{{name}}}{text}{{/{name}}}')

    def dark(self, text):
        """Return cursor text."""
        return self.color('autoblack', text)

    def bold(self, text):
        """Return text in bold."""
        return self.color('b', text)

    def bold_tail(self, text, *, sep='.'):
        """Put bold emphasis on the last part of a ``foo.bar.baz`` string."""
        head, fsep, tail = text.rpartition(sep)
        return fsep.join([head, self.bold(tail)])

    def _table_wrap(self, table, text):
        max_width = max(table.column_max_width(1), 10)
        return '\n'.join(wrap(text, max_width))

    def say(self, message, file=None, err=None, **kwargs):
        """Print something to stdout (or use ``file=stderr`` kwarg).

        Note:
            Does not do anything if the :option:`--quiet <faust --quiet>`
            option is enabled.
        """
        if not self.quiet:
            echo(message, file=file or self.stdout, err=cast(bool, err or self.stderr), **kwargs)

    def carp(self, s, **kwargs):
        """Print something to stdout (or use ``file=stderr`` kwargs).

        Note:
            Does not do anything if the :option:`--debug <faust --debug>`
            option is enabled.
        """
        if self.debug:
            self.say(f'#-- {s}', **kwargs)

    def dumps(self, obj):
        """Serialize object using JSON."""
        return json.dumps(obj)

    @property
    def loglevel(self):
        """Return the log level used for this command."""
        return self._loglevel or DEFAULT_LOGLEVEL

    @loglevel.setter
    def loglevel(self, level):
        self._loglevel = level

    @property
    def blocking_timeout(self):
        """Return the blocking timeout used for this command."""
        return self._blocking_timeout or 0.0

    @blocking_timeout.setter
    def blocking_timeout(self, timeout):
        self._blocking_timeout = timeout

    @property
    def console_port(self):
        """Return the :pypi:`aiomonitor` console port."""
        return self._console_port or CONSOLE_PORT

    @console_port.setter
    def console_port(self, port):
        self._console_port = port

class AppCommand(Command):
    """Command that takes ``-A app`` as argument."""
    abstract = True
    require_app = True

    @classmethod
    def from_handler(cls, *options, **kwargs):
        """Decorate ``async def`` command to create command class."""

        def _inner(fun):
            target = fun
            if not inspect.signature(fun).parameters:
                target = staticmethod(fun)
            fields = {'run': target, '__doc__': fun.__doc__, '__name__': fun.__name__, '__qualname__': fun.__qualname__, '__module__': fun.__module__, '__wrapped__': fun, 'options': options}
            return type(fun.__name__, (cls,), {**fields, **kwargs})
        return _inner

    def __init__(self, ctx, *args, key_serializer=None, value_serializer=None, **kwargs):
        super().__init__(ctx)
        self.app = self._finalize_app(getattr(ctx.find_root(), 'app', None))
        self.args = args
        self.kwargs = kwargs
        self.key_serializer = key_serializer or self.app.conf.key_serializer
        self.value_serializer = value_serializer or self.app.conf.value_serializer

    def _finalize_app(self, app):
        if app is not None:
            return self._finalize_concrete_app(app)
        else:
            return self._app_from_str(self.state.app)

    def _app_from_str(self, appstr=None):
        if appstr:
            return find_app(appstr)
        else:
            if self.require_app:
                raise self.UsageError('Need to specify app using -A parameter')
            return None

    def _finalize_concrete_app(self, app):
        app.finalize()
        origin = app.conf.origin
        if sys.argv:
            origin = self._detect_main_package(sys.argv)
        return prepare_app(app, origin)

    def _detect_main_package(self, argv):
        prog = Path(argv[0]).absolute()
        paths = []
        p = prog.parent
        while p:
            if not (p / '__init__.py').is_file():
                break
            paths.append(p)
            p = p.parent
        package = '.'.join([p.name for p in paths] + [prog.with_suffix('').name])
        if package.endswith('.__main__'):
            package = package[:-9]
        return package

    async def on_stop(self):
        """Call after command executed."""
        await super().on_stop()
        app = cast(_App, self.app)
        if app._producer is not None and app._producer.started:
            await app._producer.stop()
        if app.started:
            await app.stop()
        if app._http_client is not None:
            await app._maybe_close_http_client()

    def to_key(self, typ, key):
        """Convert command-line argument string to model (key).

        Arguments:
            typ: The name of the model to create.
            key: The string json of the data to populate it with.

        Notes:
            Uses :attr:`key_serializer` to set the :term:`codec`
            for the key (e.g. ``"json"``), as set by the
            :option:`--key-serializer <faust send --key-serializer>` option.
        """
        return self.to_model(typ, key, self.key_serializer)

    def to_value(self, typ, value):
        """Convert command-line argument string to model (value).

        Arguments:
            typ: The name of the model to create.
            key: The string json of the data to populate it with.

        Notes:
            Uses :attr:`value_serializer` to set the :term:`codec`
            for the value (e.g. ``"json"``), as set by the
            :option:`--value-serializer <faust send --value-serializer>`
            option.
        """
        return self.to_model(typ, value, self.value_serializer)

    def to_model(self, typ, value, serializer):
        """Convert command-line argument to model.

        Generic version of :meth:`to_key`/:meth:`to_value`.

        Arguments:
            typ: The name of the model to create.
            key: The string json of the data to populate it with.
            serializer: The argument setting it apart from to_key/to_value
                enables you to specify a custom serializer not mandated
                by :attr:`key_serializer`, and :attr:`value_serializer`.

        Notes:
            Uses :attr:`value_serializer` to set the :term:`codec`
            for the value (e.g. ``"json"``), as set by the
            :option:`--value-serializer <faust send --value-serializer>`
            option.
        """
        if typ:
            model = self.import_relative_to_app(typ)
            return model.loads(want_bytes(value), serializer=serializer)
        return want_bytes(value)

    def import_relative_to_app(self, attr):
        """Import string like "module.Model", or "Model" to model class."""
        try:
            return symbol_by_name(attr)
        except ImportError as original_exc:
            if not self.app.conf.origin:
                raise
            root, _, _ = self.app.conf.origin.partition(':')
            try:
                return symbol_by_name(f'{root}.models.{attr}')
            except ImportError:
                try:
                    return symbol_by_name(f'{root}.{attr}')
                except ImportError:
                    raise original_exc from original_exc

    def to_topic(self, entity):
        """Convert topic name given on command-line to ``app.topic()``."""
        if not entity:
            raise self.UsageError('Missing topic/@agent name')
        if entity.startswith('@'):
            return self.import_relative_to_app(entity[1:])
        return self.app.topic(entity)

    def abbreviate_fqdn(self, name, *, prefix=''):
        """Abbreviate fully-qualified Python name, by removing origin.

        ``app.conf.origin`` is the package where the app is defined,
        so if this is ``examples.simple`` it returns the truncated::

            >>> app.conf.origin
            'examples.simple'
            >>> abbr_fqdn(app.conf.origin,
            ...           'examples.simple.Withdrawal',
            ...           prefix='[...]')
            '[...]Withdrawal'

        but if the package is not part of origin it provides the full path::

            >>> abbr_fqdn(app.conf.origin,
            ...           'examples.other.Foo', prefix='[...]')
            'examples.other.foo'
        """
        if self.app.conf.origin:
            return text.abbr_fqdn(self.app.conf.origin, name, prefix=prefix)
        return ''

    @property
    def blocking_timeout(self):
        """Return the blocking timeout used for this command."""
        return self._blocking_timeout or self.app.conf.blocking_timeout

    @blocking_timeout.setter
    def blocking_timeout(self, timeout):
        self._blocking_timeout = timeout

def call_command(command, args=None, stdout=None, stderr=None, side_effects=False, **kwargs):
    exitcode = 0
    if stdout is None:
        stdout = io.StringIO()
    if stderr is None:
        stderr = io.StringIO()
    try:
        cli(args=[command] + (args or []), side_effects=side_effects, stdout=stdout, stderr=stderr, **kwargs)
    except SystemExit as exc:
        exitcode = exc.code
    return (exitcode, stdout, stderr)