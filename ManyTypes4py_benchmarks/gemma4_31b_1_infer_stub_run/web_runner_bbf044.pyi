import asyncio
import socket
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Set, Type, TypeVar, Union, Any
from yarl import URL
from .abc import AbstractAccessLogger
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .typedefs import PathLike
from .web_app import Application
from .web_log import AccessLogger
from .web_protocol import RequestHandler
from .web_request import BaseRequest, Request
from .web_server import Server

if typing.TYPE_CHECKING:
    from ssl import SSLContext

__all__ = ('BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit')

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int

def _raise_graceful_exit() -> None: ...

class BaseSite(ABC):
    _runner: 'BaseRunner[Any]'
    _ssl_context: Optional['SSLContext']
    _backlog: int
    _server: Optional[asyncio.AbstractServer]

    def __init__(self, runner: 'BaseRunner[Any]', *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

class TCPSite(BaseSite):
    _host: Optional[str]
    _port: int
    _reuse_address: Optional[bool]
    _reuse_port: Optional[bool]

    def __init__(
        self, 
        runner: 'BaseRunner[Any]', 
        host: Optional[str] = None, 
        port: Optional[int] = None, 
        *, 
        ssl_context: Optional['SSLContext'] = None, 
        backlog: int = 128, 
        reuse_address: Optional[bool] = None, 
        reuse_port: Optional[bool] = None
    ) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class UnixSite(BaseSite):
    _path: str

    def __init__(self, runner: 'BaseRunner[Any]', path: str, *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class NamedPipeSite(BaseSite):
    _path: str

    def __init__(self, runner: 'BaseRunner[Any]', path: str) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class SockSite(BaseSite):
    _sock: socket.socket
    _name: str

    def __init__(self, runner: 'BaseRunner[Any]', sock: socket.socket, *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

class BaseRunner(ABC, Generic[_Request]):
    _handle_signals: bool
    _kwargs: dict[str, Any]
    _server: Optional[Server]
    _sites: List[BaseSite]
    _shutdown_timeout: float

    def __init__(self, *, handle_signals: bool = False, shutdown_timeout: float = 60.0, **kwargs: Any) -> None: ...

    @property
    def server(self) -> Optional[Server]: ...

    @property
    def addresses(self) -> List[Union[tuple[str, int], str]]: ...

    @property
    def sites(self) -> Set[BaseSite]: ...

    async def setup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    async def cleanup(self) -> None: ...

    @abstractmethod
    async def _make_server(self) -> Server: ...

    @abstractmethod
    async def _cleanup_server(self) -> None: ...

    def _reg_site(self, site: BaseSite) -> None: ...

    def _check_site(self, site: BaseSite) -> None: ...

    def _unreg_site(self, site: BaseSite) -> None: ...

class ServerRunner(BaseRunner[BaseRequest]):
    _web_server: Server

    def __init__(self, web_server: Server, *, handle_signals: bool = False, **kwargs: Any) -> None: ...

    async def shutdown(self) -> None: ...

    async def _make_server(self) -> Server: ...

    async def _cleanup_server(self) -> None: ...

class AppRunner(BaseRunner[Request]):
    _app: Application

    def __init__(
        self, 
        app: Application, 
        *, 
        handle_signals: bool = False, 
        access_log_class: Type[AbstractAccessLogger] = AccessLogger, 
        **kwargs: Any
    ) -> None: ...

    @property
    def app(self) -> Application: ...

    async def shutdown(self) -> None: ...

    async def _make_server(self) -> Server: ...

    def _make_request(
        self, 
        message: RawRequestMessage, 
        payload: StreamReader, 
        protocol: RequestHandler, 
        writer: Any, 
        task: asyncio.Task, 
        _cls: Type[Request] = Request
    ) -> Request: ...

    async def _cleanup_server(self) -> None: ...