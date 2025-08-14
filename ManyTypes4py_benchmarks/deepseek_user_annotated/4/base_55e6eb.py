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
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    IO,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
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

from faust.types._env import (
    CONSOLE_PORT,
    DATADIR,
    DEBUG,
    WORKDIR,
)
from faust.types import AppT, CodecArg, ModelT
from faust.utils import json
from faust.utils import terminal
from faust.utils.codegen import reprcall

from . import params

if typing.TYPE_CHECKING:
    from faust.app import App as _App
else:
    class _App: ...  # noqa

try:
    import click_completion
except ImportError:
    click_completion = None
else:  # pragma: no cover
    click_completion.init()

__all__ = [
    'AppCommand',
    'Command',
    'argument',
    'cli',
    'find_app',
    'option',
]


class argument:
    """Create command-line argument."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs
        self.argument: Callable[[Any], Any] = click.argument(
            *self.args, **self.kwargs)

    def __call__(self, fun: Any) -> Any:
        return self.argument(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)


class option:
    """Create command-line option."""

    def __init__(self, *args: Any,
                 show_default: bool = True,
                 **kwargs: Any) -> None:
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs
        self.option: Callable[[Any], Any] = click.option(
            *args, show_default=show_default, **kwargs)

    def __call__(self, fun: Any) -> Any:
        return self.option(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)


OptionDecorator = Callable[[Any], Any]
OptionSequence = Sequence[OptionDecorator]
OptionList = MutableSequence[OptionDecorator]

LOOP_CHOICES: Sequence[str] = ('aio', 'eventlet', 'uvloop')
DEFAULT_LOOP: str = 'aio'
DEFAULT_LOGLEVEL: str = 'WARN'
LOGLEVELS: Sequence[str] = (
    'CRIT',
    'ERROR',
    'WARN',
    'INFO',
    'DEBUG',
)

faust_version: str = symbol_by_name('faust:__version__')


class State:
    app: Optional[AppT] = None
    quiet: bool = False
    debug: bool = False
    workdir: Optional[str] = None
    datadir: Optional[str] = None
    json: bool = False
    no_color: bool = False
    loop: Optional[str] = None
    logfile: Optional[str] = None
    loglevel: Optional[int] = None
    blocking_timeout: Optional[float] = None
    console_port: Optional[int] = None


def compat_option(
        *args: Any,
        state_key: str,
        callback: Optional[Callable[[click.Context, click.Parameter, Any], Any]] = None,
        expose_value: bool = False,
        **kwargs: Any) -> Callable[[Any], click.Parameter]:

    def _callback(ctx: click.Context,
                  param: click.Parameter,
                  value: Any) -> Any:
        state = ctx.ensure_object(State)
        prev_value = getattr(state, state_key, None)
        if prev_value is None and value != param.default:
            setattr(state, state_key, value)
        return callback(ctx, param, value) if callback else value

    return option(
        *args, callback=_callback, expose_value=expose_value, **kwargs)


now_builtin_worker_options: OptionSequence = [
    compat_option(
        '--logfile', '-f',
        state_key='logfile',
        default=None,
        type=params.WritableFilePath,
        help='Path to logfile (default is <stderr>).',
    ),
    compat_option(
        '--loglevel', '-l',
        state_key='loglevel',
        default=DEFAULT_LOGLEVEL,
        type=params.CaseInsensitiveChoice(LOGLEVELS),
        help='Logging level to use.',
    ),
    compat_option(
        '--blocking-timeout',
        state_key='blocking_timeout',
        default=None,
        type=float,
        help='when --debug: Blocking detector timeout.',
    ),
    compat_option(
        '--console-port',
        state_key='console_port',
        default=CONSOLE_PORT,
        type=params.TCPPort(),
        help='when --debug: Port to run debugger console on.',
    ),
]

core_options: OptionSequence = [
    click.version_option(version=f'Faust {faust_version}'),
    option('--app', '-A',
           help='Path of Faust application to use, or the name of a module.'),
    option('--quiet/--no-quiet', '-q', default=False,
           help='Silence output to <stdout>/<stderr>.'),
    option('--debug/--no-debug', default=DEBUG,
           help='Enable debugging output, and the blocking detector.'),
    option('--no-color/--color', '--no_color/--color', default=False,
           help='Enable colors in output.'),
    option('--workdir', '-W',
           default=WORKDIR,
           type=params.WritableDirectory,
           help='Working directory to change to after start.'),
    option('--datadir', '-D',
           default=DATADIR,
           type=params.WritableDirectory,
           help='Directory to keep application state.'),
    option('--json', default=False, is_flag=True,
           help='Return output in machine-readable JSON format'),
    option('--loop', '-L',
           default=DEFAULT_LOOP,
           type=click.Choice(LOOP_CHOICES),
           help='Event loop implementation to use.'),
]

builtin_options: OptionSequence = cast(List[OptionDecorator], core_options) + cast(List[OptionDecorator], now_builtin_worker_options)


class _FaustRootContextT(click.Context):
    app: Optional[AppT]
    stdout: Optional[IO]
    stderr: Optional[IO]
    side_effects: bool


def find_app(app: str,
             *,
             symbol_by_name: Callable = symbol_by_name,
             imp: Callable = import_from_cwd,
             attr_name: str = 'app') -> AppT:
    """Find app by string."""
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


def prepare_app(app: AppT, name: Optional[str]) -> AppT:
    app.finalize()
    if app.conf._origin is None:
        app.conf._origin = name
    app.worker_init()
    if app.conf.autodiscover:
        app.discover()
    app.worker_init_post_autodiscover()

    if 1:  # pragma: no cover
        main = sys.modules.get('__main__')
        if main is not None and 'cProfile.py' in getattr(main, '__file__', ''):
            from ..models import registry
            registry.update({
                (app.conf.origin or '') + k[8:]: v
                for k, v in registry.items()
                if k.startswith('cProfile.')
            })
        return app


def _apply_options(options: OptionSequence) -> OptionDecorator:
    def _inner(fun: OptionDecorator) -> OptionDecorator:
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

    def _maybe_import_app(self, argv: Sequence[str] = sys.argv) -> None:
        workdir = self._extract_param(argv, '-W', '--workdir')
        if workdir:
            os.chdir(Path(workdir).absolute())
        appstr = self._extract_param(argv, '-A', '--app')
        if appstr is not None:
            find_app(appstr)

    def _extract_param(self,
                       argv: Sequence[str],
                       shortopt: str,
                       longopt: str) -> Optional[str]:
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
                        raise click.UsageError(
                            f'Missing argument for {longopt}')
        return None

    @no_type_check
    def make_context(self,
                     info_name: str,
                     args: str,
                     app: AppT = None,
                     parent: click.Context = None,
                     stdout: IO = None,
                     stderr: IO = None,
                     side_effects: bool = True,
                     **extra: Any) -> click.Context:
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
def cli(*args: Any, **kwargs: Any) -> None:
    """Welcome, see list of commands and options below."""
    return _prepare_cli(*args, **kwargs)


def _prepare_cli(ctx: click.Context,
                 app: Union[AppT, str],
                 quiet: bool,
                 debug: bool,
                 workdir: str,
                 datadir: str,
                 json: bool,
                 no_color: bool,
                 loop: str) -> None:
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

    UsageError: Type[Exception] = click.UsageError

    abstract: ClassVar[bool] = True
    _click: Any = None

    debug: bool
    quiet: bool
    workdir: str
    datadir: str
    json: bool
    no_color: bool
    logfile: str
    _loglevel: Optional[str]
    _blocking_timeout: Optional[float]
    _console_port: Optional[int]

    stdout: IO
    stderr: IO

    daemon: bool = False
    redirect_stdouts: Optional[bool] = None
    redirect_stdouts_level: Optional[int] = None

    builtin_options: OptionSequence = builtin_options
    options: Optional[OptionList] = None

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    prog_name: str = ''

    @classmethod
    def as_click_command(cls) -> Callable[..., Any]:
        @click.pass_context
        @wraps(cls)
        def _inner(*args: Any, **kwargs: Any) -> NoReturn:
            cmd = cls(*args, **kwargs)
            with exiting(print_exception=True, file=sys.stderr):
                cmd()

        return _apply_options(cls.options or [])(
            cli.command(help=cls.__doc__)(_inner))

    def __init_subclass__(self, *args: Any, **kwargs: Any) -> None:
        if self.abstract:
            self.abstract = False
        else:
            self._click = self.as_click_command()

        _apply_options(self.builtin_options)(self._parse)
        _apply_options(self.options or [])(self._parse)

    @classmethod
    def parse(cls, argv: Sequence[str]) -> Mapping[str, Any]:
        return cls._parse(argv, standalone_mode=False)

    @staticmethod
    @click.command()
    def _parse(**kwargs: Any) -> Mapping[str, Any]:
        return kwargs

    def __init__(self, ctx: click.Context, *args: Any, **kwargs: Any) -> None:
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

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        try:
            await self.run(*args, **kwargs)
        finally:
            await self.on_stop()

    async def on_stop(self) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        self.run_using_worker(*args, **kwargs)

    def run_using_worker(self, *args: Any, **kwargs: Any) -> NoReturn:
        loop = asyncio.get_event_loop()
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        service = self.as_service(loop, *args, **kwargs)
        worker = self.worker_for_service(service, loop)
        self.on_worker_created(worker)
        raise worker.execute_from_commandline()

    def on_worker_created(self, worker: Worker) -> None:
        ...

    def as_service(self,
                   loop: asyncio.AbstractEventLoop,
                   *args: Any, **kwargs: Any) -> ServiceT:
        return Service.from_awaitable(
            self.execute(*args, **kwargs),
            name=type(self).__name__,
            loop=loop or asyncio.get_event_loop())

    def worker_for_service(self,
                           service: ServiceT,
                           loop: asyncio.AbstractEventLoop = None) -> Worker:
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
            daemon=self.daemon,
        )

    @property
    def _Worker(self) -> Type[Worker]:
        return Worker

    def tabulate(self,
                 data: terminal.TableDataT,
                 headers: Sequence[str] = None,
                 wrap_last_row: bool = True,
                 title: str = '',
                 title_color: str = 'blue',
                 **kwargs: Any) -> str:
        if self.json:
            return self._tabulate_json(data, headers=headers)
        if headers:
            data = [headers] + list(data)
        title = self.bold(self.color(title_color, title))
        table = self.table(data, title=title, **kwargs)
        if wrap_last_row:
            data = [
                list(item[:-1]) + [self._table_wrap(table, item[-1])]
                for item in data
            ]
        return table.table

    def _tabulate_json(self,
                       data: terminal.TableDataT,
                       headers: Sequence[str] = None) -> str:
        if headers:
            return json.dumps([dict(zip(headers, row