import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
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
import ssl

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    SSLContext = object

__all__: List[str] = ['BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit']

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int

def _raise_graceful_exit() -> None:
    ...

class BaseSite(ABC):
    __slots__: tuple = ('_runner', '_ssl_context', '_backlog', '_server')
    
    def __init__(self, runner: 'BaseRunner', *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None:
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    @abstractmethod
    async def start(self) -> Coroutine[None, None, None]:
        ...
    
    async def stop(self) -> Coroutine[None, None, None]:
        ...

class TCPSite(BaseSite):
    __slots__: tuple = ('_host', '_port', '_reuse_address', '_reuse_port')
    
    def __init__(self, runner: 'BaseRunner', host: Optional[str] = None, port: Optional[int] = None, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128, reuse_address: Optional[bool] = None, reuse_port: Optional[bool] = None) -> None:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    async def start(self) -> Coroutine[None, None, None]:
        ...

class UnixSite(BaseSite):
    __slots__: tuple = ('_path',)
    
    def __init__(self, runner: 'BaseRunner', path: PathLike, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    async def start(self) -> Coroutine[None, None, None]:
        ...

class NamedPipeSite(BaseSite):
    __slots__: tuple = ('_path',)
    
    def __init__(self, runner: 'BaseRunner', path: str) -> None:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    async def start(self) -> Coroutine[None, None, None]:
        ...

class SockSite(BaseSite):
    __slots__: tuple = ('_sock', '_name')
    
    def __init__(self, runner: 'BaseRunner', sock: socket.socket, *, ssl_context: Optional[SSLContext] = None, backlog: int = 128) -> None:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    async def start(self) -> Coroutine[None, None, None]:
        ...

class BaseRunner(ABC, Generic[_Request]):
    __slots__: tuple = ('_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')
    
    def __init__(self, *, handle_signals: bool = False, shutdown_timeout: float = 60.0, **kwargs: Any) -> None:
        ...
    
    @property
    def server(self) -> Optional[Server]:
        ...
    
    @property
    def addresses(self) -> List[Any]:
        ...
    
    @property
    def sites(self) -> Set['BaseSite']:
        ...
    
    async def setup(self) -> Coroutine[None, None, None]:
        ...
    
    @abstractmethod
    async def shutdown(self) -> Coroutine[None, None, None]:
        ...
    
    async def cleanup(self) -> Coroutine[None, None, None]:
        ...
    
    @abstractmethod
    async def _make_server(self) -> Coroutine[Any, Any, Server]:
        ...
    
    @abstractmethod
    async def _cleanup_server(self) -> Coroutine[None, None, None]:
        ...
    
    def _reg_site(self, site: 'BaseSite') -> None:
        ...
    
    def _check_site(self, site: 'BaseSite') -> None:
        ...
    
    def _unreg_site(self, site: 'BaseSite') -> None:
        ...

class ServerRunner(BaseRunner[BaseRequest]):
    __slots__: tuple = ('_web_server',)
    
    def __init__(self, web_server: Server, *, handle_signals: bool = False, **kwargs: Any) -> None:
        ...
    
    async def shutdown(self) -> Coroutine[None, None, None]:
        ...
    
    async def _make_server(self) -> Coroutine[Any, Any, Server]:
        ...
    
    async def _cleanup_server(self) -> Coroutine[None, None, None]:
        ...

class AppRunner(BaseRunner[Request]):
    __slots__: tuple = ('_app',)
    
    def __init__(self, app: Application, *, handle_signals: bool = False, access_log_class: Type[AbstractAccessLogger] = AccessLogger, **kwargs: Any) -> None:
        ...
    
    @property
    def app(self) -> Application:
        ...
    
    async def shutdown(self) -> Coroutine[None, None, None]:
        ...
    
    async def _make_server(self) -> Coroutine[Any, Any, Server]:
        ...
    
    def _make_request(self, message: RawRequestMessage, payload: StreamReader, protocol: RequestHandler, writer: AbstractStreamWriter, task: asyncio.Task, _cls: Type[Request] = Request) -> Request:
        ...
    
    async def _cleanup_server(self) -> Coroutine[None, None, None]:
        ...