from typing import Any, Generic, List, Optional, Set, Tuple, Type, TypeVar
from abc import ABC, abstractmethod
import asyncio
import socket
from ssl import SSLContext
from .abc import AbstractAccessLogger, AbstractStreamWriter
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .typedefs import PathLike
from .web_app import Application
from .web_log import AccessLogger
from .web_protocol import RequestHandler
from .web_request import BaseRequest, Request
from .web_server import Server

__all__: Tuple[str, ...] = ...

_Request = TypeVar("_Request", bound=BaseRequest)


class GracefulExit(SystemExit):
    code: int


def _raise_graceful_exit() -> None: ...


class BaseSite(ABC):
    def __init__(self, runner: "BaseRunner[Any]", *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class TCPSite(BaseSite):
    def __init__(
        self,
        runner: "BaseRunner[Any]",
        host: Any = ...,
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
    def __init__(self, runner: "BaseRunner[Any]", path: PathLike, *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...


class NamedPipeSite(BaseSite):
    def __init__(self, runner: "BaseRunner[Any]", path: str) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...


class SockSite(BaseSite):
    def __init__(self, runner: "BaseRunner[Any]", sock: socket.socket, *, ssl_context: Optional[SSLContext] = ..., backlog: int = ...) -> None: ...

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...


class BaseRunner(ABC, Generic[_Request]):
    def __init__(self, *, handle_signals: bool = ..., shutdown_timeout: float = ..., **kwargs: Any) -> None: ...

    @property
    def server(self) -> Optional[Server]: ...

    @property
    def addresses(self) -> List[Any]: ...

    @property
    def sites(self) -> Set["BaseSite"]: ...

    async def setup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    async def cleanup(self) -> None: ...

    @abstractmethod
    async def _make_server(self) -> Server: ...

    @abstractmethod
    async def _cleanup_server(self) -> None: ...

    def _reg_site(self, site: "BaseSite") -> None: ...

    def _check_site(self, site: "BaseSite") -> None: ...

    def _unreg_site(self, site: "BaseSite") -> None: ...


class ServerRunner(BaseRunner[BaseRequest]):
    def __init__(self, web_server: Server, *, handle_signals: bool = ..., **kwargs: Any) -> None: ...

    async def shutdown(self) -> None: ...

    async def _make_server(self) -> Server: ...

    async def _cleanup_server(self) -> None: ...


class AppRunner(BaseRunner[Request]):
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