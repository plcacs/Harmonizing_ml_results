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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union, cast, overload
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
from .web import Application, AppRunner, BaseRequest, BaseRunner, Request, RequestHandler, Server, ServerRunner, SockSite, UrlMappingMatchInfo
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
REUSE_ADDRESS = os.name == 'posix' and sys.platform != 'cygwin'

def get_unused_port_socket(host, family=socket.AF_INET):
    return get_port_socket(host, 0, family)

def get_port_socket(host, port, family=socket.AF_INET):
    s = socket.socket(family, socket.SOCK_STREAM)
    if REUSE_ADDRESS:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    return s

def unused_port():
    """Return a port that is unused on the current host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return cast(int, s.getsockname()[1])

class BaseTestServer(ABC, Generic[_Request]):
    __test__ = False

    def __init__(self, *, scheme='', host='127.0.0.1', port=None, skip_url_asserts=False, socket_factory=get_port_socket, **kwargs):
        self.runner = None
        self._root = None
        self.host = host
        self.port = port or 0
        self._closed = False
        self.scheme = scheme
        self.skip_url_asserts = skip_url_asserts
        self.socket_factory = socket_factory

    async def start_server(self, **kwargs):
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
    async def _make_runner(self, **kwargs):
        """Return a new runner for the server."""

    def make_url(self, path):
        assert self._root is not None
        url = URL(path)
        if not self.skip_url_asserts:
            assert not url.absolute
            return self._root.join(url)
        else:
            return URL(str(self._root) + str(path))

    @property
    def started(self):
        return self.runner is not None

    @property
    def closed(self):
        return self._closed

    @property
    def handler(self):
        runner = self.runner
        assert runner is not None
        assert runner.server is not None
        return runner.server

    async def close(self):
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run when the object is garbage collected, and on
        exit when used as a context manager.

        """
        if self.started and (not self.closed):
            assert self.runner is not None
            await self.runner.cleanup()
            self._root = None
            self.port = 0
            self._closed = True

    async def __aenter__(self):
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

class TestServer(BaseTestServer[Request]):

    def __init__(self, app, *, scheme='', host='127.0.0.1', port=None, **kwargs):
        self.app = app
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs):
        return AppRunner(self.app, **kwargs)

class RawTestServer(BaseTestServer[BaseRequest]):

    def __init__(self, handler, *, scheme='', host='127.0.0.1', port=None, **kwargs):
        self._handler = handler
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs):
        srv = Server(self._handler, **kwargs)
        return ServerRunner(srv, **kwargs)

class TestClient(Generic[_Request, _ApplicationNone]):
    """
    A test client implementation.

    To write functional tests for aiohttp based servers.

    """
    __test__ = False

    @overload
    def __init__(self, server, *, cookie_jar=None, **kwargs):
        ...

    @overload
    def __init__(self, server, *, cookie_jar=None, **kwargs):
        ...

    def __init__(self, server, *, cookie_jar=None, **kwargs):
        if not isinstance(server, BaseTestServer):
            raise TypeError('server must be TestServer instance, found type: %r' % type(server))
        self._server = server
        if cookie_jar is None:
            cookie_jar = aiohttp.CookieJar(unsafe=True)
        self._session = ClientSession(cookie_jar=cookie_jar, **kwargs)
        self._session._retry_connection = False
        self._closed = False
        self._responses = []
        self._websockets = []

    async def start_server(self):
        await self._server.start_server()

    @property
    def scheme(self):
        return self._server.scheme

    @property
    def host(self):
        return self._server.host

    @property
    def port(self):
        return self._server.port

    @property
    def server(self):
        return self._server

    @property
    def app(self):
        return getattr(self._server, 'app', None)

    @property
    def session(self):
        """An internal aiohttp.ClientSession.

        Unlike the methods on the TestClient, client session requests
        do not automatically include the host in the url queried, and
        will require an absolute path to the resource.

        """
        return self._session

    def make_url(self, path):
        return self._server.make_url(path)

    async def _request(self, method, path, **kwargs):
        resp = await self._session.request(method, self.make_url(path), **kwargs)
        self._responses.append(resp)
        return resp
    if sys.version_info >= (3, 11) and TYPE_CHECKING:

        def request(self, method, path, **kwargs):
            ...

        def get(self, path, **kwargs):
            ...

        def options(self, path, **kwargs):
            ...

        def head(self, path, **kwargs):
            ...

        def post(self, path, **kwargs):
            ...

        def put(self, path, **kwargs):
            ...

        def patch(self, path, **kwargs):
            ...

        def delete(self, path, **kwargs):
            ...
    else:

        def request(self, method, path, **kwargs):
            """Routes a request to tested http server.

            The interface is identical to aiohttp.ClientSession.request,
            except the loop kwarg is overridden by the instance used by the
            test server.

            """
            return _RequestContextManager(self._request(method, path, **kwargs))

        def get(self, path, **kwargs):
            """Perform an HTTP GET request."""
            return _RequestContextManager(self._request(hdrs.METH_GET, path, **kwargs))

        def post(self, path, **kwargs):
            """Perform an HTTP POST request."""
            return _RequestContextManager(self._request(hdrs.METH_POST, path, **kwargs))

        def options(self, path, **kwargs):
            """Perform an HTTP OPTIONS request."""
            return _RequestContextManager(self._request(hdrs.METH_OPTIONS, path, **kwargs))

        def head(self, path, **kwargs):
            """Perform an HTTP HEAD request."""
            return _RequestContextManager(self._request(hdrs.METH_HEAD, path, **kwargs))

        def put(self, path, **kwargs):
            """Perform an HTTP PUT request."""
            return _RequestContextManager(self._request(hdrs.METH_PUT, path, **kwargs))

        def patch(self, path, **kwargs):
            """Perform an HTTP PATCH request."""
            return _RequestContextManager(self._request(hdrs.METH_PATCH, path, **kwargs))

        def delete(self, path, **kwargs):
            """Perform an HTTP PATCH request."""
            return _RequestContextManager(self._request(hdrs.METH_DELETE, path, **kwargs))

    def ws_connect(self, path, **kwargs):
        """Initiate websocket connection.

        The api corresponds to aiohttp.ClientSession.ws_connect.

        """
        return _WSRequestContextManager(self._ws_connect(path, **kwargs))

    async def _ws_connect(self, path, **kwargs):
        ws = await self._session.ws_connect(self.make_url(path), **kwargs)
        self._websockets.append(ws)
        return ws

    async def close(self):
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run on exit when used as a(n) (asynchronous)
        context manager.

        """
        if not self._closed:
            for resp in self._responses:
                resp.close()
            for ws in self._websockets:
                await ws.close()
            await self._session.close()
            await self._server.close()
            self._closed = True

    async def __aenter__(self):
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

class AioHTTPTestCase(IsolatedAsyncioTestCase, ABC):
    """A base class to allow for unittest web applications using aiohttp.

    Provides the following:

    * self.client (aiohttp.test_utils.TestClient): an aiohttp test client.
    * self.app (aiohttp.web.Application): the application returned by
        self.get_application()

    Note that the TestClient's methods are asynchronous: you have to
    execute function on the test client using asynchronous methods.
    """

    @abstractmethod
    async def get_application(self):
        """Get application.

        This method should be overridden to return the aiohttp.web.Application
        object to test.
        """

    async def asyncSetUp(self):
        self.app = await self.get_application()
        self.server = await self.get_server(self.app)
        self.client = await self.get_client(self.server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def get_server(self, app):
        """Return a TestServer instance."""
        return TestServer(app)

    async def get_client(self, server):
        """Return a TestClient instance."""
        return TestClient(server)
_LOOP_FACTORY = Callable[[], asyncio.AbstractEventLoop]

@contextlib.contextmanager
def loop_context(loop_factory=asyncio.new_event_loop, fast=False):
    """A contextmanager that creates an event_loop, for test purposes.

    Handles the creation and cleanup of a test loop.
    """
    loop = setup_test_loop(loop_factory)
    yield loop
    teardown_test_loop(loop, fast=fast)

def setup_test_loop(loop_factory=asyncio.new_event_loop):
    """Create and return an asyncio.BaseEventLoop instance.

    The caller should also call teardown_test_loop,
    once they are done with the loop.
    """
    loop = loop_factory()
    asyncio.set_event_loop(loop)
    return loop

def teardown_test_loop(loop, fast=False):
    """Teardown and cleanup an event_loop created by setup_test_loop."""
    closed = loop.is_closed()
    if not closed:
        loop.call_soon(loop.stop)
        loop.run_forever()
        loop.close()
    if not fast:
        gc.collect()
    asyncio.set_event_loop(None)

def _create_app_mock():

    def get_dict(app, key):
        return app.__app_dict[key]

    def set_dict(app, key, value):
        app.__app_dict[key] = value
    app = mock.MagicMock(spec=Application)
    app.__app_dict = {}
    app.__getitem__ = get_dict
    app.__setitem__ = set_dict
    app.on_response_prepare = Signal(app)
    app.on_response_prepare.freeze()
    return app

def _create_transport(sslcontext=None):
    transport = mock.Mock()

    def get_extra_info(key):
        if key == 'sslcontext':
            return sslcontext
        else:
            return None
    transport.get_extra_info.side_effect = get_extra_info
    return transport

def make_mocked_request(method, path, headers=None, *, match_info=None, version=HttpVersion(1, 1), closing=False, app=None, writer=None, protocol=None, transport=None, payload=EMPTY_PAYLOAD, sslcontext=None, client_max_size=1024 ** 2, loop=...):
    """Creates mocked web.Request testing purposes.

    Useful in unit tests, when spinning full web server is overkill or
    specific conditions and errors are hard to trigger.
    """
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
        raw_hdrs = tuple(((k.encode('utf-8'), v.encode('utf-8')) for k, v in headers.items()))
    else:
        headers = CIMultiDictProxy(CIMultiDict())
        raw_hdrs = ()
    chunked = 'chunked' in headers.get(hdrs.TRANSFER_ENCODING, '').lower()
    message = RawRequestMessage(method, path, version, headers, raw_hdrs, closing, None, False, chunked, URL(path))
    if app is None:
        app = _create_app_mock()
    if transport is None:
        transport = _create_transport(sslcontext)
    if protocol is None:
        protocol = mock.Mock()
        protocol.transport = transport
    if writer is None:
        writer = mock.Mock()
        writer.write_headers = make_mocked_coro(None)
        writer.write = make_mocked_coro(None)
        writer.write_eof = make_mocked_coro(None)
        writer.drain = make_mocked_coro(None)
        writer.transport = transport
    protocol.transport = transport
    req = Request(message, payload, protocol, writer, task, loop, client_max_size=client_max_size)
    match_info = UrlMappingMatchInfo({} if match_info is None else match_info, mock.Mock())
    match_info.add_app(app)
    req._match_info = match_info
    return req

def make_mocked_coro(return_value=sentinel, raise_exception=sentinel):
    """Creates a coroutine mock."""

    async def mock_coro(*args, **kwargs):
        if raise_exception is not sentinel:
            raise raise_exception
        if not inspect.isawaitable(return_value):
            return return_value
        await return_value
    return mock.Mock(wraps=mock_coro)