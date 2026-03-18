```pyi
import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Set, Type, TypeVar
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

__all__: List[str]

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int

def _raise_graceful_exit() -> None: ...

class BaseSite(ABC):
    _runner: Any
    _ssl_context: Optional[SSLContext]
    _backlog: int
    _server: Any
    
    def __init__(self, runner: Any, *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...
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
        runner: Any,
        host: Optional[str] = ...,
        port: Optional[int] = ...,
        *,
        ssl_context: Optional[SSLContext] = ...,
        backlog: int = ...,
        reuse_address: Optional[bool] = ...,
        reuse_port: Optional[bool] = ...,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    async def start(self) -> None: ...

class UnixSite(BaseSite):
    _path: PathLike
    
    def __init__(self, runner: Any, path: PathLike, *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...
    @property
    def name(self) -> str: ...
    async def start(self) -> None: ...

class NamedPipeSite(BaseSite):
    _path: PathLike
    
    def __init__(self, runner: Any, path: PathLike) -> None: ...
    @property
    def name(self) -> str: ...
    async def start(self) -> None: ...

class SockSite(BaseSite):
    _sock: socket.socket
    _name: str
    
    def __init__(self, runner: Any, sock: socket.socket, *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...
    @property
    def name(self) -> str: ...
    async def start(self) -> None: ...

class BaseRunner(ABC, Generic[_Request]):
    _handle_signals: bool
    _kwargs: Any
    _server: Optional[Any]
    _sites: List[BaseSite]
    _shutdown_timeout: float
    
    def __init__(self, *, handle_signals: bool = ..., shutdown_timeout: float = ..., **kwargs: Any) -> None: ...
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
    _web_server: Server
    
    def __init__(self, web_server: Server, *, handle_signals: bool = ..., **kwargs: Any) -> None: ...
    async def shutdown(self) -> None: ...
    async def _make_server(self) -> Server: ...
    async def _cleanup_server(self) -> None: ...

class AppRunner(BaseRunner[Request]):
    _app: Application
    
    def __init__(
        self,
        app: Application,
        *,
        handle_signals: bool = ...,
        access_log_class: Type[AbstractAccessLogger] = ...,
        **kwargs: Any,
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
        writer: AbstractStreamWriter,
        task: asyncio.Task[Any],
        _cls: Type[Request] = ...,
    ) -> Request: ...
    async def _cleanup_server(self) -> None: ...
```