import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Dict,
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

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    try:
        from ssl import SSLContext
    except ImportError:
        SSLContext = object

__all__ = (
    'BaseSite',
    'TCPSite',
    'UnixSite',
    'NamedPipeSite',
    'SockSite',
    'BaseRunner',
    'AppRunner',
    'ServerRunner',
    'GracefulExit',
)

_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code: int = 1

def _raise_graceful_exit() -> None:
    raise GracefulExit()

class BaseSite(ABC):
    __slots__ = ('_runner', '_ssl_context', '_backlog', '_server')

    def __init__(
        self,
        runner: "BaseRunner[Any]",
        *,
        ssl_context: Optional[SSLContext] = None,
        backlog: int = 128,
    ) -> None:
        if runner.server is None:
            raise RuntimeError('Call runner.setup() before making a site')
        self._runner: BaseRunner[Any] = runner
        self._ssl_context: Optional[SSLContext] = ssl_context
        self._backlog: int = backlog
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the site (e.g. a URL)."""

    @abstractmethod
    async def start(self) -> None:
        self._runner._reg_site(self)

    async def stop(self) -> None:
        self._runner._check_site(self)
        if self._server is not None:
            self._server.close()
        self._runner._unreg_site(self)

class TCPSite(BaseSite):
    __slots__ = ('_host', '_port', '_reuse_address', '_reuse_port')

    def __init__(
        self,
        runner: "BaseRunner[Any]",
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        ssl_context: Optional[SSLContext] = None,
        backlog: int = 128,
        reuse_address: Optional[bool] = None,
        reuse_port: Optional[bool] = None,
    ) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._host: Optional[str] = host
        if port is None:
            port = 8443 if self._ssl_context else 8080
        self._port: int = port
        self._reuse_address: Optional[bool] = reuse_address
        self._reuse_port: Optional[bool] = reuse_port

    @property
    def name(self) -> str:
        scheme: str = 'https' if self._ssl_context else 'http'
        host: str = '0.0.0.0' if not self._host else self._host
        return str(URL.build(scheme=scheme, host=host, port=self._port))

    async def start(self) -> None:
        await super().start()
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        server: Any = self._runner.server
        assert server is not None
        self._server = await loop.create_server(
            server,
            self._host,
            self._port,
            ssl=self._ssl_context,
            backlog=self._backlog,
            reuse_address=self._reuse_address,
            reuse_port=self._reuse_port,
        )

class UnixSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(
        self,
        runner: "BaseRunner[Any]",
        path: PathLike,
        *,
        ssl_context: Optional[SSLContext] = None,
        backlog: int = 128,
    ) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._path: PathLike = path

    @property
    def name(self) -> str:
        scheme: str = 'https' if self._ssl_context else 'http'
        return f'{scheme}://unix:{self._path}:'

    async def start(self) -> None:
        await super().start()
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        server: Any = self._runner.server
        assert server is not None
        self._server = await loop.create_unix_server(
            server,
            self._path,
            ssl=self._ssl_context,
            backlog=self._backlog,
        )

class NamedPipeSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: "BaseRunner[Any]", path: str) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        if not isinstance(loop, asyncio.ProactorEventLoop):
            raise RuntimeError('Named Pipes only available in proactor loop under windows')
        super().__init__(runner)
        self._path: str = path

    @property
    def name(self) -> str:
        return self._path

    async def start(self) -> None:
        await super().start()
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        server: Any = self._runner.server
        assert server is not None
        _server = await loop.start_serving_pipe(server, self._path)
        self._server = _server[0]

class SockSite(BaseSite):
    __slots__ = ('_sock', '_name')

    def __init__(
        self,
        runner: "BaseRunner[Any]",
        sock: socket.socket,
        *,
        ssl_context: Optional[SSLContext] = None,
        backlog: int = 128,
    ) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._sock: socket.socket = sock
        scheme: str = 'https' if self._ssl_context else 'http'
        if hasattr(socket, 'AF_UNIX') and sock.family == socket.AF_UNIX:
            name: str = f'{scheme}://unix:{sock.getsockname()}:'
        else:
            host, port = sock.getsockname()[:2]
            name = str(URL.build(scheme=scheme, host=host, port=port))
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        await super().start()
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        server: Any = self._runner.server
        assert server is not None
        self._server = await loop.create_server(
            server,
            sock=self._sock,
            ssl=self._ssl_context,
            backlog=self._backlog,
        )

_RequestT = TypeVar("_RequestT", bound=BaseRequest)

class BaseRunner(ABC, Generic[_RequestT]):
    __slots__ = ('_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')

    def __init__(self, *, handle_signals: bool = False, shutdown_timeout: float = 60.0, **kwargs: Any) -> None:
        self._handle_signals: bool = handle_signals
        self._kwargs: Dict[str, Any] = kwargs
        self._server: Optional[Any] = None
        self._sites: List[BaseSite] = []
        self._shutdown_timeout: float = shutdown_timeout

    @property
    def server(self) -> Optional[Any]:
        return self._server

    @property
    def addresses(self) -> List[Any]:
        ret: List[Any] = []
        for site in self._sites:
            server = site._server
            if server is not None:
                sockets = server.sockets
                if sockets is not None:
                    for sock in sockets:
                        ret.append(sock.getsockname())
        return ret

    @property
    def sites(self) -> Set[BaseSite]:
        return set(self._sites)

    async def setup(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        if self._handle_signals:
            try:
                loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)
                loop.add_signal_handler(signal.SIGTERM, _raise_graceful_exit)
            except NotImplementedError:
                pass
        self._server = await self._make_server()

    @abstractmethod
    async def shutdown(self) -> None:
        """Call any shutdown hooks to help server close gracefully."""

    async def cleanup(self) -> None:
        for site in list(self._sites):
            await site.stop()
        if self._server:
            await asyncio.sleep(0)
            self._server.pre_shutdown()
            await self.shutdown()
            await self._server.shutdown(self._shutdown_timeout)
        await self._cleanup_server()
        self._server = None
        if self._handle_signals:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except NotImplementedError:
                pass

    @abstractmethod
    async def _make_server(self) -> Any:
        """Return a new server for the runner to serve requests."""

    @abstractmethod
    async def _cleanup_server(self) -> None:
        """Run any cleanup steps after the server is shutdown."""

    def _reg_site(self, site: BaseSite) -> None:
        if site in self._sites:
            raise RuntimeError(f'Site {site} is already registered in runner {self}')
        self._sites.append(site)

    def _check_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')

    def _unreg_site(self, site: BaseSite) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')
        self._sites.remove(site)

class ServerRunner(BaseRunner[BaseRequest]):
    """Low-level web server runner"""
    __slots__ = ('_web_server',)

    def __init__(self, web_server: Any, *, handle_signals: bool = False, **kwargs: Any) -> None:
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._web_server: Any = web_server

    async def shutdown(self) -> None:
        pass

    async def _make_server(self) -> Any:
        return self._web_server

    async def _cleanup_server(self) -> None:
        pass

class AppRunner(BaseRunner[Request]):
    """Web Application runner"""
    __slots__ = ('_app',)

    def __init__(
        self,
        app: Application,
        *,
        handle_signals: bool = False,
        access_log_class: Type[AbstractAccessLogger] = AccessLogger,
        **kwargs: Any,
    ) -> None:
        if not isinstance(app, Application):
            raise TypeError('The first argument should be web.Application instance, got {!r}'.format(app))
        kwargs['access_log_class'] = access_log_class
        if app._handler_args:
            for k, v in app._handler_args.items():
                kwargs[k] = v
        if not issubclass(kwargs['access_log_class'], AbstractAccessLogger):
            raise TypeError('access_log_class must be subclass of aiohttp.abc.AbstractAccessLogger, got {}'.format(kwargs['access_log_class']))
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._app: Application = app

    @property
    def app(self) -> Application:
        return self._app

    async def shutdown(self) -> None:
        await self._app.shutdown()

    async def _make_server(self) -> Any:
        self._app.on_startup.freeze()
        await self._app.startup()
        self._app.freeze()
        return Server(self._app._handle, request_factory=self._make_request, **self._kwargs)

    def _make_request(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: Any,
        writer: AbstractStreamWriter,
        task: asyncio.Task,
        _cls: Type[Request] = Request,
    ) -> Request:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return _cls(
            message,
            payload,
            protocol,
            writer,
            task,
            loop,
            client_max_size=self.app._client_max_size,
        )

    async def _cleanup_server(self) -> None:
        await self._app.cleanup()