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
__all__ = ('BaseSite', 'TCPSite', 'UnixSite', 'NamedPipeSite', 'SockSite', 'BaseRunner', 'AppRunner', 'ServerRunner', 'GracefulExit')
_Request = TypeVar('_Request', bound=BaseRequest)

class GracefulExit(SystemExit):
    code = 1

def _raise_graceful_exit() -> None:
    raise GracefulExit()

class BaseSite(ABC):
    __slots__ = ('_runner', '_ssl_context', '_backlog', '_server')

    def __init__(self, runner: typing.Callable, *, ssl_context: Union[None, typing.Callable]=None, backlog: int=128) -> None:
        if runner.server is None:
            raise RuntimeError('Call runner.setup() before making a site')
        self._runner = runner
        self._ssl_context = ssl_context
        self._backlog = backlog
        self._server = None

    @property
    @abstractmethod
    def name(self) -> typing.Text:
        """Return the name of the site (e.g. a URL)."""

    @abstractmethod
    async def start(self):
        self._runner._reg_site(self)

    async def stop(self):
        self._runner._check_site(self)
        if self._server is not None:
            self._server.close()
        self._runner._unreg_site(self)

class TCPSite(BaseSite):
    __slots__ = ('_host', '_port', '_reuse_address', '_reuse_port')

    def __init__(self, runner: typing.Callable, host=None, port=None, *, ssl_context: Union[None, typing.Callable]=None, backlog: int=128, reuse_address=None, reuse_port=None) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._host = host
        if port is None:
            port = 8443 if self._ssl_context else 8080
        self._port = port
        self._reuse_address = reuse_address
        self._reuse_port = reuse_port

    @property
    def name(self) -> typing.Text:
        scheme = 'https' if self._ssl_context else 'http'
        host = '0.0.0.0' if not self._host else self._host
        return str(URL.build(scheme=scheme, host=host, port=self._port))

    async def start(self):
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, self._host, self._port, ssl=self._ssl_context, backlog=self._backlog, reuse_address=self._reuse_address, reuse_port=self._reuse_port)

class UnixSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: typing.Callable, path: Union[str, aiohttp.web.Application, int], *, ssl_context: Union[None, typing.Callable]=None, backlog: int=128) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._path = path

    @property
    def name(self) -> typing.Text:
        scheme = 'https' if self._ssl_context else 'http'
        return f'{scheme}://unix:{self._path}:'

    async def start(self):
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_unix_server(server, self._path, ssl=self._ssl_context, backlog=self._backlog)

class NamedPipeSite(BaseSite):
    __slots__ = ('_path',)

    def __init__(self, runner: typing.Callable, path: Union[str, aiohttp.web.Application, int]) -> None:
        loop = asyncio.get_event_loop()
        if not isinstance(loop, asyncio.ProactorEventLoop):
            raise RuntimeError('Named Pipes only available in proactor loop under windows')
        super().__init__(runner)
        self._path = path

    @property
    def name(self) -> typing.Text:
        return self._path

    async def start(self):
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        _server = await loop.start_serving_pipe(server, self._path)
        self._server = _server[0]

class SockSite(BaseSite):
    __slots__ = ('_sock', '_name')

    def __init__(self, runner: typing.Callable, sock, *, ssl_context: Union[None, typing.Callable]=None, backlog: int=128) -> None:
        super().__init__(runner, ssl_context=ssl_context, backlog=backlog)
        self._sock = sock
        scheme = 'https' if self._ssl_context else 'http'
        if hasattr(socket, 'AF_UNIX') and sock.family == socket.AF_UNIX:
            name = f'{scheme}://unix:{sock.getsockname()}:'
        else:
            host, port = sock.getsockname()[:2]
            name = str(URL.build(scheme=scheme, host=host, port=port))
        self._name = name

    @property
    def name(self) -> typing.Text:
        return self._name

    async def start(self):
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, sock=self._sock, ssl=self._ssl_context, backlog=self._backlog)

class BaseRunner(ABC, Generic[_Request]):
    __slots__ = ('_handle_signals', '_kwargs', '_server', '_sites', '_shutdown_timeout')

    def __init__(self, *, handle_signals=False, shutdown_timeout=60.0, **kwargs) -> None:
        self._handle_signals = handle_signals
        self._kwargs = kwargs
        self._server = None
        self._sites = []
        self._shutdown_timeout = shutdown_timeout

    @property
    def server(self):
        return self._server

    @property
    def addresses(self) -> list:
        ret = []
        for site in self._sites:
            server = site._server
            if server is not None:
                sockets = server.sockets
                if sockets is not None:
                    for sock in sockets:
                        ret.append(sock.getsockname())
        return ret

    @property
    def sites(self) -> set:
        return set(self._sites)

    async def setup(self):
        loop = asyncio.get_event_loop()
        if self._handle_signals:
            try:
                loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)
                loop.add_signal_handler(signal.SIGTERM, _raise_graceful_exit)
            except NotImplementedError:
                pass
        self._server = await self._make_server()

    @abstractmethod
    async def shutdown(self):
        """Call any shutdown hooks to help server close gracefully."""

    async def cleanup(self):
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
            loop = asyncio.get_running_loop()
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except NotImplementedError:
                pass

    @abstractmethod
    async def _make_server(self):
        """Return a new server for the runner to serve requests."""

    @abstractmethod
    async def _cleanup_server(self):
        """Run any cleanup steps after the server is shutdown."""

    def _reg_site(self, site: Union[dict[str, typing.Any], dict]) -> None:
        if site in self._sites:
            raise RuntimeError(f'Site {site} is already registered in runner {self}')
        self._sites.append(site)

    def _check_site(self, site: str) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')

    def _unreg_site(self, site: Any) -> None:
        if site not in self._sites:
            raise RuntimeError(f'Site {site} is not registered in runner {self}')
        self._sites.remove(site)

class ServerRunner(BaseRunner[BaseRequest]):
    """Low-level web server runner"""
    __slots__ = ('_web_server',)

    def __init__(self, web_server, *, handle_signals=False, **kwargs) -> None:
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._web_server = web_server

    async def shutdown(self):
        pass

    async def _make_server(self):
        return self._web_server

    async def _cleanup_server(self):
        pass

class AppRunner(BaseRunner[Request]):
    """Web Application runner"""
    __slots__ = ('_app',)

    def __init__(self, app, *, handle_signals=False, access_log_class=AccessLogger, **kwargs) -> None:
        if not isinstance(app, Application):
            raise TypeError('The first argument should be web.Application instance, got {!r}'.format(app))
        kwargs['access_log_class'] = access_log_class
        if app._handler_args:
            for k, v in app._handler_args.items():
                kwargs[k] = v
        if not issubclass(kwargs['access_log_class'], AbstractAccessLogger):
            raise TypeError('access_log_class must be subclass of aiohttp.abc.AbstractAccessLogger, got {}'.format(kwargs['access_log_class']))
        super().__init__(handle_signals=handle_signals, **kwargs)
        self._app = app

    @property
    def app(self):
        return self._app

    async def shutdown(self):
        await self._app.shutdown()

    async def _make_server(self):
        self._app.on_startup.freeze()
        await self._app.startup()
        self._app.freeze()
        return Server(self._app._handle, request_factory=self._make_request, **self._kwargs)

    def _make_request(self, message: Union[int, str], payload: Union[int, str], protocol: Union[int, str], writer: Union[int, str], task: Union[int, str], _cls: Any=Request):
        loop = asyncio.get_running_loop()
        return _cls(message, payload, protocol, writer, task, loop, client_max_size=self.app._client_max_size)

    async def _cleanup_server(self):
        await self._app.cleanup()