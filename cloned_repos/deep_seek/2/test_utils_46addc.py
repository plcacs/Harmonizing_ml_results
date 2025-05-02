"""Utilities shared by tests."""
import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Generic, Iterator, List, Optional, 
    Type, TypeVar, Union, cast, overload, Awaitable, Coroutine, Mapping,
    MutableMapping, Sequence, Tuple, Set, FrozenSet, Iterable, AsyncIterator,
    AsyncIterable, ContextManager, AsyncContextManager, NoReturn, Text, BinaryIO
)
from unittest import IsolatedAsyncioTestCase, mock
from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
import aiohttp
from aiohttp.client import _RequestContextManager, _RequestOptions, _WSRequestContextManager
from . import ClientSession, hdrs
from .abc import AbstractCookieJar, AbstractStreamWriter
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import LooseHeaders, StrOrURL
from .web import (
    Application, AppRunner, BaseRequest, BaseRunner, Request, RequestHandler,
    Server, ServerRunner, SockSite, UrlMappingMatchInfo
)
from .web_protocol import _RequestHandler

if TYPE_CHECKING:
    from ssl import SSLContext
else:
    SSLContext = None

if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

_ApplicationNone = TypeVar('_ApplicationNone', Application, None)
_Request = TypeVar('_Request', bound=BaseRequest)
REUSE_ADDRESS: bool = os.name == 'posix' and sys.platform != 'cygwin'

def get_unused_port_socket(host: str, family: int = socket.AF_INET) -> socket.socket:
    return get_port_socket(host, 0, family)

def get_port_socket(host: str, port: int, family: int = socket.AF_INET) -> socket.socket:
    s = socket.socket(family, socket.SOCK_STREAM)
    if REUSE_ADDRESS:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    return s

def unused_port() -> int:
    """Return a port that is unused on the current host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return cast(int, s.getsockname()[1])

class BaseTestServer(ABC, Generic[_Request]):
    __test__: bool = False

    def __init__(
        self, 
        *, 
        scheme: str = '', 
        host: str = '127.0.0.1', 
        port: Optional[int] = None, 
        skip_url_asserts: bool = False, 
        socket_factory: Callable[..., socket.socket] = get_port_socket, 
        **kwargs: Any
    ) -> None:
        self.runner: Optional[BaseRunner] = None
        self._root: Optional[URL] = None
        self.host: str = host
        self.port: int = port or 0
        self._closed: bool = False
        self.scheme: str = scheme
        self.skip_url_asserts: bool = skip_url_asserts
        self.socket_factory: Callable[..., socket.socket] = socket_factory
        self._ssl: Optional[SSLContext] = None

    async def start_server(self, **kwargs: Any) -> None:
        if self.runner:
            return
        self._ssl = kwargs.pop('ssl', None)
        self.runner = await self._make_runner(handler_cancellation=True, **kwargs)
        await self.runner.setup()
        absolute_host = self.host
        try:
            version = ipaddress.ip_address(self.host).version
        except ValueError:
            version = 4
        if version == 6:
            absolute_host = f'[{self.host}]'
        family = socket.AF_INET6 if version == 6 else socket.AF_INET
        _sock = self.socket_factory(self.host, self.port, family)
        self.host, self.port = _sock.getsockname()[:2]
        site = SockSite(self.runner, sock=_sock, ssl_context=self._ssl)
        await site.start()
        server = site._server
        assert server is not None
        sockets = server.sockets
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]
        if not self.scheme:
            self.scheme = 'https' if self._ssl else 'http'
        self._root = URL(f'{self.scheme}://{absolute_host}:{self.port}')

    @abstractmethod
    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        """Return a new runner for the server."""

    def make_url(self, path: str) -> URL:
        assert self._root is not None
        url = URL(path)
        if not self.skip_url_asserts:
            assert not url.absolute
            return self._root.join(url)
        else:
            return URL(str(self._root) + str(path))

    @property
    def started(self) -> bool:
        return self.runner is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def handler(self) -> Any:
        runner = self.runner
        assert runner is not None
        assert runner.server is not None
        return runner.server

    async def close(self) -> None:
        """Close all fixtures created by the test client."""
        if self.started and (not self.closed):
            assert self.runner is not None
            await self.runner.cleanup()
            self._root = None
            self.port = 0
            self._closed = True

    async def __aenter__(self) -> Self:
        await self.start_server()
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_value: Optional[BaseException]], 
        traceback: Optional[TracebackType]
    ) -> None:
        await self.close()

class TestServer(BaseTestServer[Request]):
    def __init__(
        self, 
        app: Application, 
        *, 
        scheme: str = '', 
        host: str = '127.0.0.1', 
        port: Optional[int] = None, 
        **kwargs: Any
    ) -> None:
        self.app: Application = app
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs: Any) -> AppRunner:
        return AppRunner(self.app, **kwargs)

class RawTestServer(BaseTestServer[BaseRequest]):
    def __init__(
        self, 
        handler: Callable[..., Any], 
        *, 
        scheme: str = '', 
        host: str = '127.0.0.1', 
        port: Optional[int] = None, 
        **kwargs: Any
    ) -> None:
        self._handler: Callable[..., Any] = handler
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs: Any) -> ServerRunner:
        srv = Server(self._handler, **kwargs)
        return ServerRunner(srv, **kwargs)

class TestClient(Generic[_Request, _ApplicationNone]):
    __test__: bool = False

    @overload
    def __init__(self, server: TestServer, *, cookie_jar: Optional[AbstractCookieJar] = None, **kwargs: Any) -> None:
        ...

    @overload
    def __init__(self, server: RawTestServer, *, cookie_jar: Optional[AbstractCookieJar] = None, **kwargs: Any) -> None:
        ...

    def __init__(
        self, 
        server: Union[TestServer, RawTestServer], 
        *, 
        cookie_jar: Optional[AbstractCookieJar] = None, 
        **kwargs: Any
    ) -> None:
        if not isinstance(server, BaseTestServer):
            raise TypeError('server must be TestServer instance, found type: %r' % type(server))
        self._server: Union[TestServer, RawTestServer] = server
        if cookie_jar is None:
            cookie_jar = aiohttp.CookieJar(unsafe=True)
        self._session: ClientSession = ClientSession(cookie_jar=cookie_jar, **kwargs)
        self._session._retry_connection = False
        self._closed: bool = False
        self._responses: List[ClientResponse] = []
        self._websockets: List[ClientWebSocketResponse] = []

    async def start_server(self) -> None:
        await self._server.start_server()

    @property
    def scheme(self) -> str:
        return self._server.scheme

    @property
    def host(self) -> str:
        return self._server.host

    @property
    def port(self) -> int:
        return self._server.port

    @property
    def server(self) -> Union[TestServer, RawTestServer]:
        return self._server

    @property
    def app(self) -> Optional[Application]:
        return getattr(self._server, 'app', None)

    @property
    def session(self) -> ClientSession:
        return self._session

    def make_url(self, path: str) -> URL:
        return self._server.make_url(path)

    async def _request(self, method: str, path: str, **kwargs: Any) -> ClientResponse:
        resp = await self._session.request(method, self.make_url(path), **kwargs)
        self._responses.append(resp)
        return resp

    if sys.version_info >= (3, 11) and TYPE_CHECKING:
        def request(self, method: str, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def get(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def options(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def head(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def post(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def put(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def patch(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...

        def delete(self, path: str, **kwargs: Any) -> _RequestContextManager:
            ...
    else:
        def request(self, method: str, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(method, path, **kwargs))

        def get(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_GET, path, **kwargs))

        def post(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_POST, path, **kwargs))

        def options(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_OPTIONS, path, **kwargs))

        def head(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_HEAD, path, **kwargs))

        def put(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_PUT, path, **kwargs))

        def patch(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_PATCH, path, **kwargs))

        def delete(self, path: str, **kwargs: Any) -> _RequestContextManager:
            return _RequestContextManager(self._request(hdrs.METH_DELETE, path, **kwargs))

    def ws_connect(self, path: str, **kwargs: Any) -> _WSRequestContextManager:
        return _WSRequestContextManager(self._ws_connect(path, **kwargs))

    async def _ws_connect(self, path: str, **kwargs: Any) -> ClientWebSocketResponse:
        ws = await self._session.ws_connect(self.make_url(path), **kwargs)
        self._websockets.append(ws)
        return ws

    async def close(self) -> None:
        if not self._closed:
            for resp in self._responses:
                resp.close()
            for ws in self._websockets:
                await ws.close()
            await self._session.close()
            await self._server.close()
            self._closed = True

    async def __aenter__(self) -> Self:
        await self.start_server()
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc: Optional[BaseException], 
        tb: Optional[TracebackType]
    ) -> None:
        await self.close()

class AioHTTPTestCase(IsolatedAsyncioTestCase, ABC):
    @abstractmethod
    async def get_application(self) -> Application:
        ...

    async def asyncSetUp(self) -> None:
        self.app: Application = await self.get_application()
        self.server: TestServer = await self.get_server(self.app)
        self.client: TestClient[Request, Application] = await self.get_client(self.server)
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def get_server(self, app: Application) -> TestServer:
        return TestServer(app)

    async def get_client(self, server: TestServer) -> TestClient[Request, Application]:
        return TestClient(server)

_LOOP_FACTORY = Callable[[], asyncio.AbstractEventLoop]

@contextlib.contextmanager
def loop_context(
    loop_factory: _LOOP_FACTORY = asyncio.new_event_loop, 
    fast: bool = False
) -> Iterator[asyncio.AbstractEventLoop]:
    loop = setup_test_loop(loop_factory)
    yield loop
    teardown_test_loop(loop, fast=fast)

def setup_test_loop(loop_factory: _LOOP_FACTORY = asyncio.new_event_loop) -> asyncio.AbstractEventLoop:
    loop = loop_factory()
    asyncio.set_event_loop(loop)
    return loop

def teardown_test_loop(loop: asyncio.AbstractEventLoop, fast: bool = False) -> None:
    closed = loop.is_closed()
    if not closed:
        loop.call_soon(loop.stop)
        loop.run_forever()
        loop.close()
    if not fast:
        gc.collect()
    asyncio.set_event_loop(None)

def _create_app_mock() -> mock.MagicMock:
    def get_dict(app: mock.MagicMock, key: str) -> Any:
        return app.__app_dict[key]

    def set_dict(app: mock.MagicMock, key: str, value: Any) -> None:
        app.__app_dict[key] = value

    app = mock.MagicMock(spec=Application)
    app.__app_dict: Dict[str, Any] = {}
    app.__getitem__ = get_dict
    app.__setitem__ = set_dict
    app.on_response_prepare = Signal(app)
    app.on_response_prepare.freeze()
    return app

def _create_transport(sslcontext: Optional[SSLContext] = None) -> mock.Mock:
    transport = mock.Mock()

    def get_extra_info(key: str) -> Optional[SSLContext]:
        if key == 'sslcontext':
            return sslcontext
        else:
            return None
    transport.get_extra_info.side_effect = get_extra_info
    return transport

def make_mocked_request(
    method: str, 
    path: str, 
    headers: Optional[LooseHeaders] = None, 
    *, 
    match_info: Optional[Dict[str, Any]] = None, 
    version: HttpVersion = HttpVersion(1, 1), 
    closing: bool = False, 
    app: Optional[Application] = None, 
    writer: Optional[AbstractStreamWriter] = None, 
    protocol: Optional[Any] = None, 
    transport: Optional[Any] = None, 
    payload: Any = EMPTY_PAYLOAD, 
    sslcontext: Optional[SSLContext] = None, 
    client_max_size: int = 1024 ** 2, 
    loop: Any = ...
) -> Request:
    task = mock.Mock()
    if loop is ...:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = mock.Mock()
            loop.create_future.return_value = ()
    if version < HttpVersion(1, 1):
        closing = True
    if headers:
        headers = CIMultiDictProxy(CIMultiDict(headers))
        raw_hdrs = tuple((k.encode('utf-8'), v.encode('utf-8')) for k, v in headers.items())
    else:
        headers = CIMultiDictProxy(CIMultiDict())
        raw_hdrs = ()
    chunked = 'chunked' in headers.get(hdrs.TRANSFER_ENCODING, '').lower()
    message = RawRequestMessage(
        method, path, version, headers, raw_hdrs, closing, None, False, chunked, URL(path)
   