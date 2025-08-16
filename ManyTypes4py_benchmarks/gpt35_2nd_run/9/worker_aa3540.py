from faust.types import AppT
from faust.worker import Worker as FaustWorker
from mode import Worker
from mode.utils.logging import level_name
from typing import Any, List, Optional, Tuple, Type
import asyncio
import os
import platform
import socket
from mode.utils.imports import symbol_by_name
from yarl import URL
from . import params
from .base import AppCommand, now_builtin_worker_options, option

class worker(AppCommand):
    daemon: bool = True
    redirect_stdouts: bool = True
    worker_options: List = [option('--with-web/--without-web', default=True, help='Enable/disable web server and related components.'), option('--web-port', '-p', default=None, type=params.TCPPort(), help=f'Port to run web server on (default: {WEB_PORT})'), option('--web-transport', default=None, type=params.URLParam(), help=f'Web server transport (default: {WEB_TRANSPORT})'), option('--web-bind', '-b', type=str), option('--web-host', '-h', default=socket.gethostname(), type=str, help=f'Canonical host name for the web server (default: {WEB_BIND})')]
    options: List = worker_options + now_builtin_worker_options

    def on_worker_created(self, worker: Worker) -> None:
        ...

    def as_service(self, loop: asyncio.AbstractEventLoop, *args, **kwargs) -> AppT:
        ...

    def _init_worker_options(self, *args, with_web: bool, web_port: Optional[int], web_bind: str, web_host: str, web_transport: URL, **kwargs) -> None:
        ...

    @property
    def _Worker(self) -> Type[Worker]:
        ...

    def banner(self, worker: Worker) -> str:
        ...

    def _format_banner_table(self, data: List[Tuple[str, Any]]) -> str:
        ...

    def _banner_title(self) -> str:
        ...

    def _banner_data(self, worker: Worker) -> List[Tuple[str, str]]:
        ...

    def _human_cython_info(self) -> Optional[Tuple[str, str]]:
        ...

    def _human_transport_info(self, loop: asyncio.AbstractEventLoop) -> str:
        ...

    def _driver_versions(self, app: AppT) -> List[str]:
        ...

    def faust_ident(self) -> str:
        ...

    def platform(self) -> str:
        ...
