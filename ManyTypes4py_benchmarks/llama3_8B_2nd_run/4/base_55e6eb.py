from typing import Any, Callable, Dict, List, Optional, Tuple
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
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs
        self.argument: Callable[[Any], Any] = click.argument(*self.args, **self.kwargs)

    def __call__(self, fun: Callable[[Any], Any]) -> Any:
        return self.argument(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)

class option:
    """Create command-line option.

    SeeAlso:
        :func:`click.option`
    """

    def __init__(self, *args, show_default: bool = True, **kwargs):
        self.args: Tuple[Any, ...] = args
        self.kwargs: Dict[str, Any] = kwargs
        self.option: Callable[[Any], Any] = click.option(*self.args, show_default=show_default, **self.kwargs)

    def __call__(self, fun: Callable[[Any], Any]) -> Any:
        return self.option(fun)

    def __repr__(self) -> str:
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
    app: Optional[_App]
    quiet: bool
    debug: bool
    workdir: Optional[Path]
    datadir: Optional[Path]
    json: bool
    no_color: bool
    loop: Optional[asyncio.BaseEventLoop]
    logfile: Optional[IO[str]]
    loglevel: Optional[str]
    blocking_timeout: Optional[float]
    console_port: Optional[int]

def compat_option(*args, state_key: str, callback: Optional[Callable[[Any, Any, Any], Any]] = None, expose_value: bool = False, **kwargs) -> Callable[[Any], Any]:
    ...

def find_app(app: str, *, symbol_by_name=symbol_by_name, imp=import_from_cwd, attr_name='app') -> _App:
    ...

class _FaustRootContextT(click.Context):
    ...

class Command(abc.ABC):
    """Base class for subcommands."""
    UsageError = click.UsageError
    abstract: bool = True
    _click: Optional[Callable[[Any], Any]]
    daemon: bool
    redirect_stdouts: Optional[bool]
    redirect_stdouts_level: Optional[str]
    builtin_options: List[OptionDecorator]
    options: Optional[List[OptionDecorator]]
    prog_name: str

    @classmethod
    def as_click_command(cls) -> Callable[[Any], Any]:
        ...

    def __init_subclass__(cls, *args, **kwargs):
        ...

    @classmethod
    def parse(cls, argv: List[str]) -> Dict[str, Any]:
        ...

    def __init__(self, ctx: click.Context, *args, **kwargs):
        ...

    async def run(self, *args, **kwargs) -> Any:
        ...

    async def execute(self, *args, **kwargs) -> Any:
        ...

    def __call__(self, *args, **kwargs) -> Any:
        ...

    def run_using_worker(self, *args, **kwargs) -> Any:
        ...

    def on_worker_created(self, worker: Worker) -> Any:
        ...

    def as_service(self, loop: asyncio.BaseEventLoop, *args, **kwargs) -> ServiceT:
        ...

    def worker_for_service(self, service: ServiceT, loop: asyncio.BaseEventLoop) -> Worker:
        ...

    def tabulate(self, data: List[Tuple[Any, ...]], headers: Optional[List[str]] = None, wrap_last_row: bool = True, title: str = '', title_color: str = 'blue', **kwargs) -> str:
        ...

    def carp(self, s: str, **kwargs) -> Any:
        ...

    def dumps(self, obj: Any) -> str:
        ...

    @property
    def loglevel(self) -> str:
        ...

    @loglevel.setter
    def loglevel(self, level: str) -> None:
        ...

    @property
    def blocking_timeout(self) -> float:
        ...

    @blocking_timeout.setter
    def blocking_timeout(self, timeout: float) -> None:
        ...

    @property
    def console_port(self) -> int:
        ...

    @console_port.setter
    def console_port(self, port: int) -> None:
        ...

class AppCommand(Command):
    """Command that takes ``-A app`` as argument."""
    abstract: bool = True
    require_app: bool = True

    @classmethod
    def from_handler(cls, *options, **kwargs) -> Callable[[Any], Any]:
        ...

    def __init__(self, ctx: click.Context, *args, key_serializer: Optional[str] = None, value_serializer: Optional[str] = None, **kwargs):
        ...

    def _finalize_app(self, app: Optional[_App]) -> _App:
        ...

    def to_key(self, typ: str, key: str) -> Any:
        ...

    def to_value(self, typ: str, value: str) -> Any:
        ...

    def to_model(self, typ: str, value: str, serializer: str) -> Any:
        ...

    def import_relative_to_app(self, attr: str) -> Any:
        ...

    def to_topic(self, entity: str) -> Any:
        ...

    def abbreviate_fqdn(self, name: str, *, prefix: str = '') -> str:
        ...

    @property
    def blocking_timeout(self) -> float:
        ...

    @blocking_timeout.setter
    def blocking_timeout(self, timeout: float) -> None:
        ...
