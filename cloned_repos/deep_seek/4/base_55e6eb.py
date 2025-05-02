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
from typing import (
    Any, Awaitable, Callable, ClassVar, Dict, IO, List, Mapping, 
    MutableSequence, Optional, Sequence, Tuple, Type, Union, cast, 
    no_type_check, TypeVar, Generic, overload
)
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

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AppT_co = TypeVar('AppT_co', bound=AppT, covariant=True)

try:
    import click_completion
except ImportError:
    click_completion = None
else:
    click_completion.init()

__all__ = ['AppCommand', 'Command', 'argument', 'cli', 'find_app', 'option']

class argument:
    """Create command-line argument."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.argument = click.argument(*self.args, **self.kwargs)

    def __call__(self, fun: F) -> F:
        return self.argument(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)

class option:
    """Create command-line option."""
    def __init__(self, *args: Any, show_default: bool = True, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.option = click.option(*args, show_default=show_default, **kwargs)

    def __call__(self, fun: F) -> F:
        return self.option(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)

OptionDecorator = Callable[[F], F]
OptionSequence = Sequence[OptionDecorator]
OptionList = MutableSequence[OptionDecorator]
LOOP_CHOICES = ('aio', 'eventlet', 'uvloop')
DEFAULT_LOOP = 'aio'
DEFAULT_LOGLEVEL = 'WARN'
LOGLEVELS = ('CRIT', 'ERROR', 'WARN', 'INFO', 'DEBUG')
faust_version = symbol_by_name('faust:__version__')

class State:
    app: Optional[Union[str, _App]] = None
    quiet: bool = False
    debug: bool = False
    workdir: Optional[str] = None
    datadir: Optional[str] = None
    json: bool = False
    no_color: bool = False
    loop: Optional[str] = None
    logfile: Optional[str] = None
    loglevel: Optional[str] = None
    blocking_timeout: Optional[float] = None
    console_port: Optional[int] = None

def compat_option(
    *args: Any, 
    state_key: str, 
    callback: Optional[Callable[..., Any]] = None, 
    expose_value: bool = False, 
    **kwargs: Any
) -> option:
    def _callback(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        state = ctx.ensure_object(State)
        prev_value = getattr(state, state_key, None)
        if prev_value is None and value != param.default:
            setattr(state, state_key, value)
        return callback(ctx, param, value) if callback else value
    return option(*args, callback=_callback, expose_value=expose_value, **kwargs)

now_builtin_worker_options: List[option] = [
    compat_option('--logfile', '-f', state_key='logfile', default=None, 
                 type=params.WritableFilePath, help='Path to logfile (default is <stderr>).'),
    compat_option('--loglevel', '-l', state_key='loglevel', default=DEFAULT_LOGLEVEL, 
                 type=params.CaseInsensitiveChoice(LOGLEVELS), help='Logging level to use.'),
    compat_option('--blocking-timeout', state_key='blocking_timeout', default=None, 
                 type=float, help='when --debug: Blocking detector timeout.'),
    compat_option('--console-port', state_key='console_port', default=CONSOLE_PORT, 
                 type=params.TCPPort(), help='when --debug: Port to run debugger console on.')
]

core_options: List[option] = [
    click.version_option(version=f'Faust {faust_version}'),
    option('--app', '-A', help='Path of Faust application to use, or the name of a module.'),
    option('--quiet/--no-quiet', '-q', default=False, help='Silence output to <stdout>/<stderr>.'),
    option('--debug/--no-debug', default=DEBUG, help='Enable debugging output, and the blocking detector.'),
    option('--no-color/--color', '--no_color/--color', default=False, help='Enable colors in output.'),
    option('--workdir', '-W', default=WORKDIR, type=params.WritableDirectory, help='Working directory to change to after start.'),
    option('--datadir', '-D', default=DATADIR, type=params.WritableDirectory, help='Directory to keep application state.'),
    option('--json', default=False, is_flag=True, help='Return output in machine-readable JSON format'),
    option('--loop', '-L', default=DEFAULT_LOOP, type=click.Choice(LOOP_CHOICES), help='Event loop implementation to use.')
]

builtin_options: List[option] = cast(List[option], core_options) + cast(List[option], now_builtin_worker_options)

class _FaustRootContextT(click.Context):
    app: Optional[_App]
    stdout: IO[str]
    stderr: IO[str]
    side_effects: bool

def find_app(
    app: Union[str, _App], 
    *, 
    symbol_by_name: Callable[..., Any] = symbol_by_name,
    imp: Callable[..., Any] = import_from_cwd,
    attr_name: str = 'app'
) -> _App:
    """Find app by string like ``examples.simple``."""
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

def prepare_app(app: _App, name: str) -> _App:
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

def _apply_options(options: Sequence[OptionDecorator]) -> Callable[[F], F]:
    """Add list of ``click.option`` values to click command function."""
    def _inner(fun: F) -> F:
        for opt in options:
            fun = opt(fun)
        return fun
    return _inner

class _Group(click.Group):
    def get_help(self, ctx: click.Context) -> str:
        self._maybe_import_app()
        return super().get_help(ctx)

    def get_usage(self, ctx: click.Context) -> str:
        self._maybe_import_app()
        return super().get_usage(ctx)

    def _maybe_import_app(self, argv: List[str] = sys.argv) -> None:
        workdir = self._extract_param(argv, '-W', '--workdir')
        if workdir:
            os.chdir(Path(workdir).absolute())
        appstr = self._extract_param(argv, '-A', '--app')
        if appstr is not None:
            find_app(appstr)

    def _extract_param(self, argv: List[str], shortopt: str, longopt: str) -> Optional[str]:
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
    def make_context(
        self, 
        info_name: str, 
        args: List[str], 
        app: Optional[_App] = None, 
        parent: Optional[click.Context] = None, 
        stdout: Optional[IO[str]] = None, 
        stderr: Optional[IO[str]] = None, 
        side_effects: bool = True, 
        **extra: Any
    ) -> click.Context:
        ctx = super().make_context(info_name, args, **extra)
        self._maybe_import_app()
        root = cast(_FaustRootContextT, ctx.find_root())
        root.app = app
        root.stdout = stdout or sys.stdout
        root.stderr = stderr or sys.stderr
        root.side_effects = side_effects
        return ctx

@click.group(cls=_Group)
@_apply_options(builtin_options)
@click.pass_context
def cli(
    ctx: click.Context,
    app: Optional[str],
    quiet: bool,
    debug: bool,
    workdir: str,
    datadir: str,
    json: bool,
    no_color: bool,
    loop: str
) -> None:
    """Welcome, see list of commands and options below."""
    return _prepare_cli(ctx, app, quiet, debug, workdir, datadir, json, no_color, loop)

def _prepare_cli(
    ctx: click.Context,
    app: Optional[str],
    quiet: bool,
    debug: bool,
    workdir: str,
    datadir: str,
    json: bool,
    no_color: bool,
    loop: str
) -> None:
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
    abstract: ClassVar[bool] = True
    _click: Optional[click.Command] = None
    daemon: bool = False
    redirect_stdouts: Optional[bool] = None
    redirect_stdouts_level: Optional[int] = None
    builtin_options: ClassVar[List[option]] = builtin_options
    options: Optional[List[option]] = None
    prog_name: str = ''

    @classmethod
    def as_click_command(cls) -> click.Command:
        """Convert command into :pypi:`click` command."""
        @click.pass_context
        @wraps(cls)
        def _inner(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
            cmd = cls(ctx, *args, **kwargs)
            with exiting(print_exception=True, file=sys.stderr):
                cmd()
        return _apply_options(cls.options or [])(cli.command(help=cls.__doc__)(_inner)

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        if cls.abstract:
            cls.abstract = False
        else:
            cls._click = cls.as_click_command()
        _apply_options(cls.builtin_options)(cls._parse)
        _apply_options(cls.options or [])(cls._parse)

    @classmethod
    def parse(cls, argv: List[str]) -> Dict[str, Any]:
        """Parse command-line arguments in ``argv`` and return mapping."""
        return cls._parse(argv, standalone_mode=False)

    @staticmethod
    @click.command()
    def _parse(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    def __init__(
        self, 
        ctx: click.Context, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
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

    async def run(self, *args: Any, **kwargs: Any) -> None:
        """Override this method to define what your command does."""
        ...

    async def execute(self, *args: Any, **kwargs: Any) -> None:
        """Execute command."""
        try:
            await self.run(*args, **kwargs)
        finally:
            await self.on_stop()

    async def on_stop(self) -> None:
        """Call after command executed."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Call command-line command."""
        self.run_using_worker(*args, **kwargs)

    def run_using_worker(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Execute command using :class:`faust.Worker`."""
        loop = asyncio.get_event_loop()
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        service = self.as_service(loop, *args, **kwargs)
        worker = self.worker_for_service(service, loop)
        self.on_worker_created(worker)
        raise worker.execute_from_commandline()

    def on_worker_created(self, worker: Worker) -> None:
        """Call when creating :class:`faust.Worker` to execute this command."""
        ...

    def as_service(
        self, 
        loop: asyncio.AbstractEventLoop, 
        *args: Any, 
        **kwargs: Any
    ) -> ServiceT:
        """Wrap command in a :class:`mode.Service` object."""
        return Service.from_awaitable(
            self.execute(*args, **kwargs), 
            name=type(self).__name__, 
            loop=loop or asyncio.get_event_loop()
        )

    def worker_for_service(
        self, 
        service: ServiceT, 
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Worker:
        """Create :class:`faust.Worker` instance for this command."""
        return self._Worker(
            service, 
            debug=self.debug, 
            quiet=self.quiet, 
            stdout=self.stdout, 
            stderr=self.stderr, 
            loglevel=self.loglevel, 
            logfile=self.logfile, 
            blocking_timeout=self.blocking_timeout, 
            console_port=self.console_port, 
            redirect_stdouts=self.redirect_stdouts or False, 
            redirect_stdouts_level=self.redirect_stdouts_level, 
            loop=loop or asyncio.get_event_loop(), 
            daemon=self.daemon
        )

    @property
    def _Worker(self) -> Type[Worker]:
        return Worker

    def tabulate(
        self, 
        data: Sequence[Sequence[Any]], 
        headers: Optional[Sequence[str]] = None, 
        wrap_last_row: bool = True, 
        title: str = '', 
        title_color: str = 'blue', 
        **kwargs: Any
    ) -> str:
        """Create an ANSI representation of a table of two-row tuples."""
        if self.json:
            return self._tabulate_json(data, headers=headers)
        if headers:
            data = [headers] + list(data)
        title =