import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union
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

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    try:
        from ssl import SSLContext
    except ImportError:
        SSLContext = object

__all__: Tuple[str, ...] = ('BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit')

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int = 1

def _raise_graceful_exit() -> None: ...

class BaseSite(ABC):
    __slots__: Tuple[str, ...] = ('_runner', '_ssl_context', '_backlog', '_server')
    
    def __init__(self, runner: 'BaseRunner[Any]', *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    async def start(self) -> None: ...
    
    async def stop(self) -> None: ...

class TCPSite(BaseSite):
    __slots__: Tuple[str, ...] = ('_host', '_port', '_reuse_address', '_reuse_port')
    
    def __init__(self, runner: 'BaseRunner[Any]', host: Optional[str] = None, port: Optional[int] = None, *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128, reuse_address: Optional[bool] = None, reuse_port: Optional[bool] = None) -> None: ...
    
    @property
    def name(self) -> str: ...
    
    async def start(self) -> None: ...

class UnixSite(BaseSite):
    __slots__: Tuple[str, ...] = ('_path',)
    
    def __init__(self, runner: 'BaseRunner[Any]', path: PathLike, *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...
    
    @property
    def name(self) -> str: ...
    
    async def start(self) -> None: ...

class NamedPipeSite(BaseSite):
    __slots__: Tuple[str, ...] = ('_path',)
    
    def __init__(self, runner: 'BaseRunner[Any]', path: PathLike) -> None: ...
    
    @property
    def name(self) -> str: ...
    
    async def start(self) -> None: ...

class SockSite(BaseSite):
    __slots__: Tuple[str, ...] = ('_sock', '_name')
    
    def __init__(self, runner: 'BaseRunner[Any]', sock: socket.socket, *, ssl_context: Optional['SSLContext'] = None, backlog: int = 128) -> None: ...
    
    @property
    def name(self) -> str: ...
    
    async def start(self) -> None: ...

class BaseRunner(ABC, Generic[_Request]):
    __slots__: Tuple[str, ...] = ('_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')
    
    def __init__(self, *, handle_signals: bool = False, shutdown_timeout: float = 60.0, **kwargs: Any) -> None: ...
    
    @property
    def server(self) -> Optional[Server]: ...
    
    @property
    def addresses(self) -> List[Tuple[Any, ...]]: ...
    
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
    __slots__: Tuple[str, ...] = ('_web_server',)
    
    def __init__(self, web_server: Server, *, handle_signals: bool = False, **kwargs: Any) -> None: ...
    
    async def shutdown(self) -> None: ...
    
    async def _make_server(self) -> Server: ...
    
    async def _cleanup_server(self) -> None: ...

class AppRunner(BaseRunner[Request]):
    __slots__: Tuple[str, ...] = ('_app',)
    
    def __init__(self, app: Application, *, handle_signals: bool = False, access_log_class: Type[AbstractAccessLogger] = AccessLogger, **kwargs: Any) -> None: ...
    
    @property
    def app(self) -> Application: ...
    
    async def shutdown(self) -> None: ...
    
    async def _make_server(self) -> Server: ...
    
    def _make_request(self, message: RawRequestMessage, payload: StreamReader, protocol: RequestHandler, writer: AbstractStreamWriter, task: 'asyncio.Task[Any]', _cls: Type[Request] = Request) -> Request: ...
    
    async def _cleanup_server(self) -> None: ...