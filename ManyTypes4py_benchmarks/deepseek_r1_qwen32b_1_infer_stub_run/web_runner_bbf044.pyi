import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from ssl import SSLContext
from yarl import URL
from .abc import AbstractAccessLogger, AbstractStreamWriter
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .typedefs import PathLike
from .web_app import Application
from .web_log import AccessLogger
from .web_protocol import RequestHandler
from .web_request import BaseRequest, Request
from .web_server import Server

__all__ = ('BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit')

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int

def _raise_graceful_exit() -> None: ...

class BaseSite(ABC):
    __slots__ = ('_runner', '_ssl_context', '_backlog', '_server')
    _runner: BaseRunner
    _ssl_context: Optional[SSLContext]
    _backlog: int
    _server: Optional[Any]

    def __init__(self, runner: BaseRunner, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None: ...

    @property
    def name(self) -> str: ...

    @abstractmethod
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

class TCPSite(BaseSite):
    __slots__ = ('_host', '_port', '_reuse_address', '_reuse_port')
    _host: Optional[str]
    _port: int
    _reuse_address: Optional[bool]
    _reuse_port: Optional[bool]

    def __init__(self, runner: BaseRunner, host: Optional[str] = None, port: Optional[int] = None, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128, reuse_address: Optional[bool] = None, reuse_port: Optional[bool] = None) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class UnixSite(BaseSite):
    __slots__ = ('_path',)
    _path: PathLike

    def __init__(self, runner: BaseRunner, path: PathLike, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class NamedPipeSite(BaseSite):
    __slots__ = ('_path',)
    _path: str

    def __init__(self, runner: BaseRunner, path: str) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class SockSite(BaseSite):
    __slots__ = ('_sock', '_name')
    _sock: socket.socket
    _name: str

    def __init__(self, runner: BaseRunner, sock: socket.socket, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class BaseRunner(ABC, Generic[_Request]):
    __slots__ = ('_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')
    _handle_signals: bool
    _kwargs: Dict[str, Any]
    _server: Optional[Any]
    _sites: List[BaseSite]
    _shutdown_timeout: float

    def __init__(self, *, handle_signals: bool = False, shutdown_timeout: float = 60.0, **kwargs: Any) -> None: ...

    @property
    def server(self) -> Optional[Any]: ...

    @property
    def addresses(self) -> List[Any]: ...

    @property
    def sites(self) -> Set[BaseSite]: ...

    async def setup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    async def cleanup(self) -> None: ...

    @abstractmethod
    async def _make_server(self) -> Any: ...

    @abstractmethod
    async def _cleanup_server(self) -> None: ...

    def _reg_site(self, site: BaseSite) -> None: ...

    def _check_site(self, site: BaseSite) -> None: ...

    def _unreg_site(self, site: BaseSite) -> None: ...

class ServerRunner(BaseRunner[BaseRequest]):
    __slots__ = ('_web_server',)
    _web_server: Server

    def __init__(self, web_server: Server, *, handle_signals: bool = False, **kwargs: Any) -> None: ...

    async def shutdown(self) -> None: ...

    async def _make_server(self) -> Server: ...

    async def _cleanup_server(self) -> None: ...

class AppRunner(BaseRunner[Request]):
    __slots__ = ('_app',)
    _app: Application

    def __init__(self, app: Application, *, handle_signals: bool = False, access_log_class: Type[AbstractAccessLogger] = AccessLogger, **kwargs: Any) -> None: ...

    @property
    def app(self) -> Application: ...

    async def shutdown(self) -> None: ...

    async def _make_server(self) -> Server: ...

    async def _cleanup_server(self) -> None: ...

    def _make_request(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: RequestHandler,
        writer: AbstractStreamWriter,
        task: asyncio.Task,
        _cls: Type[Request] = Request,
    ) -> Request: ...