from typing import Deque, Tuple, List, Dict, Any, Callable, Optional, Type, Union
from tornado.escape import _unicode
from tornado import gen, version
from tornado.httpclient import HTTPResponse, HTTPError, AsyncHTTPClient, main, _RequestProxy, HTTPRequest
from tornado import httputil
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError, IOStream
from tornado.netutil import Resolver, OverrideResolver, _client_ssl_defaults, is_valid_ip
from tornado.log import gen_log
from tornado.tcpclient import TCPClient
import base64
import collections
import copy
import functools
import re
import socket
import ssl
import sys
import time
from io import BytesIO
import urllib.parse
from typing import Dict, Any, Callable, Optional, Type, Union
from types import TracebackType
if typing.TYPE_CHECKING:
    from typing import Deque, Tuple, List

class HTTPTimeoutError(HTTPError):
    """Error raised by SimpleAsyncHTTPClient on timeout.

    For historical reasons, this is a subclass of `.HTTPClientError`
    which simulates a response code of 599.

    .. versionadded:: 5.1
    """

    def __init__(self, message: str):
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or 'Timeout'

class HTTPStreamClosedError(HTTPError):
    """Error raised by SimpleAsyncHTTPClient when the underlying stream is closed.

    When a more specific exception is available (such as `ConnectionResetError`),
    it may be raised instead of this one.

    For historical reasons, this is a subclass of `.HTTPClientError`
    which simulates a response code of 599.

    .. versionadded:: 5.1
    """

    def __init__(self, message: str):
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or 'Stream closed'

class SimpleAsyncHTTPClient(AsyncHTTPClient):
    """Non-blocking HTTP client with no external dependencies.

    This class implements an HTTP 1.1 client on top of Tornado's IOStreams.
    Some features found in the curl-based AsyncHTTPClient are not yet
    supported.  In particular, proxies are not supported, connections
    are not reused, and callers cannot select the network interface to be
    used.

    This implementation supports the following arguments, which can be passed
    to ``configure()`` to control the global singleton, or to the constructor
    when ``force_instance=True``.

    ``max_clients`` is the number of concurrent requests that can be
    in progress; when this limit is reached additional requests will be
    queued. Note that time spent waiting in this queue still counts
    against the ``request_timeout``.

    ``defaults`` is a dict of parameters that will be used as defaults on all
    `.HTTPRequest` objects submitted to this client.

    ``hostname_mapping`` is a dictionary mapping hostnames to IP addresses.
    It can be used to make local DNS changes when modifying system-wide
    settings like ``/etc/hosts`` is not possible or desirable (e.g. in
    unittests). ``resolver`` is similar, but using the `.Resolver` interface
    instead of a simple mapping.

    ``max_buffer_size`` (default 100MB) is the number of bytes
    that can be read into memory at once. ``max_body_size``
    (defaults to ``max_buffer_size``) is the largest response body
    that the client will accept.  Without a
    ``streaming_callback``, the smaller of these two limits
    applies; with a ``streaming_callback`` only ``max_body_size``
    does.

    .. versionchanged:: 4.2
        Added the ``max_body_size`` argument.
    """

    def __init__(self, 
                 max_clients: int = 10, 
                 hostname_mapping: Optional[Dict[str, str]] = None, 
                 max_buffer_size: int = 104857600, 
                 resolver: Optional[Resolver] = None, 
                 defaults: Optional[Dict[str, str]] = None, 
                 max_header_size: Optional[int] = None, 
                 max_body_size: Optional[int] = None) -> None:
        super().__init__(defaults=defaults)
        self.max_clients = max_clients
        self.queue: Deque[Tuple[Any, HTTPRequest, Callable[[HTTPResponse], None]]] = collections.deque()
        self.active: Dict[Any, Tuple[HTTPRequest, Callable[[HTTPResponse], None], Optional[Callable[[], None]]]] = {}
        self.waiting: Dict[Any, Tuple[HTTPRequest, Callable[[HTTPResponse], None], Optional[Callable[[], None]]]] = {}
        self.max_buffer_size = max_buffer_size
        self.max_header_size = max_header_size
        self.max_body_size = max_body_size
        if resolver:
            self.resolver = resolver
            self.own_resolver = False
        else:
            self.resolver = Resolver()
            self.own_resolver = True
        if hostname_mapping is not None:
            self.resolver = OverrideResolver(resolver=self.resolver, mapping=hostname_mapping)
        self.tcp_client = TCPClient(resolver=self.resolver)

    def close(self) -> None:
        super().close()
        if self.own_resolver:
            self.resolver.close()
        self.tcp_client.close()

    def fetch_impl(self, 
                   request: HTTPRequest, 
                   callback: Callable[[HTTPResponse], None]) -> None:
        key = object()
        self.queue.append((key, request, callback))
        assert request.connect_timeout is not None
        assert request.request_timeout is not None
        timeout_handle = None
        if len(self.active) >= self.max_clients:
            timeout = min(request.connect_timeout, request.request_timeout) or request.connect_timeout or request.request_timeout
            if timeout:
                timeout_handle = self.io_loop.add_timeout(self.io_loop.time() + timeout, functools.partial(self._on_timeout, key, 'in request queue'))
        self.waiting[key] = (request, callback, timeout_handle)
        self._process_queue()
        if self.queue:
            gen_log.debug('max_clients limit reached, request queued. %d active, %d queued requests.' % (len(self.active), len(self.queue)))

    def _process_queue(self) -> None:
        while self.queue and len(self.active) < self.max_clients:
            key, request, callback = self.queue.popleft()
            if key not in self.waiting:
                continue
            self._remove_timeout(key)
            self.active[key] = (request, callback)
            release_callback = functools.partial(self._release_fetch, key)
            self._handle_request(request, release_callback, callback)

    def _connection_class(self) -> Type[IOStream]:
        return _HTTPConnection

    def _handle_request(self, 
                       request: HTTPRequest, 
                       release_callback: Callable[[], None], 
                       final_callback: Optional[Callable[[HTTPResponse], None]], 
                       max_buffer_size: int, 
                       tcp_client: TCPClient, 
                       max_header_size: Optional[int], 
                       max_body_size: Optional[int]) -> None:
        self._connection_class()(self, request, release_callback, final_callback, max_buffer_size, tcp_client, max_header_size, max_body_size)

    def _release_fetch(self, key: Any) -> None:
        del self.active[key]
        self._process_queue()

    def _remove_timeout(self, key: Any) -> None:
        if key in self.waiting:
            request, callback, timeout_handle = self.waiting[key]
            if timeout_handle is not None:
                self.io_loop.remove_timeout(timeout_handle)
            del self.waiting[key]

    def _on_timeout(self, key: Any, info: Optional[str] = None) -> None:
        """Timeout callback of request.

        Construct a timeout HTTPResponse when a timeout occurs.

        :arg object key: A simple object to mark the request.
        :info string key: More detailed timeout information.
        """
        request, callback, timeout_handle = self.waiting[key]
        self.queue.remove((key, request, callback))
        error_message = f'Timeout {info}' if info else 'Timeout'
        timeout_response = HTTPResponse(request, 599, error=HTTPTimeoutError(error_message), request_time=self.io_loop.time() - request.start_time)
        self.io_loop.add_callback(callback, timeout_response)
        del self.waiting[key]

class _HTTPConnection(httputil.HTTPMessageDelegate):
    _SUPPORTED_METHODS: Tuple[str, ...] = {'GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'}

    def __init__(self, 
                 client: SimpleAsyncHTTPClient, 
                 request: HTTPRequest, 
                 release_callback: Callable[[], None], 
                 final_callback: Optional[Callable[[HTTPResponse], None]], 
                 max_buffer_size: int, 
                 tcp_client: TCPClient, 
                 max_header_size: Optional[int], 
                 max_body_size: Optional[int]) -> None:
        self.io_loop = IOLoop.current()
        self.start_time = self.io_loop.time()
        self.start_wall_time = time.time()
        self.client = client
        self.request = request
        self.release_callback = release_callback
        self.final_callback = final_callback
        self.max_buffer_size = max_buffer_size
        self.tcp_client = tcp_client
        self.max_header_size = max_header_size
        self.max_body_size = max_body_size
        self.code: Optional[int] = None
        self.headers: Optional[Dict[str, str]] = None
        self.chunks: List[bytes] = []
        self._decompressor: Optional[Callable[[bytes], bytes]] = None
        self._timeout: Optional[Callable[[], None]] = None
        self._sockaddr: Optional[Tuple[str, int]] = None
        IOLoop.current().add_future(gen.convert_yielded(self.run()), lambda f: f.result())

    def _get_ssl_options(self, scheme: str) -> Optional[ssl.SSLContext]:
        if scheme == 'https':
            if self.request.ssl_options is not None:
                return self.request.ssl_options
            if self.request.validate_cert and self.request.ca_certs is None and (self.request.client_cert is None) and (self.request.client_key is None):
                return _client_ssl_defaults
            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.request.ca_certs)
            if not self.request.validate_cert:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
            if self.request.client_cert is not None:
                ssl_ctx.load_cert_chain(self.request.client_cert, self.request.client_key)
            if hasattr(ssl, 'OP_NO_COMPRESSION'):
                ssl_ctx.options |= ssl.OP_NO_COMPRESSION
            return ssl_ctx
        return None

    def _on_timeout(self, info: Optional[str] = None) -> None:
        """Timeout callback of _HTTPConnection instance.

        Raise a `HTTPTimeoutError` when a timeout occurs.

        :info string key: More detailed timeout information.
        """
        self._timeout = None
        error_message = f'Timeout {info}' if info else 'Timeout'
        if self.final_callback is not None:
            self._handle_exception(HTTPTimeoutError, HTTPTimeoutError(error_message), None)

    def _remove_timeout(self) -> None:
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def _create_connection(self, stream: IOStream) -> HTTP1Connection:
        stream.set_nodelay(True)
        connection = HTTP1Connection(stream, True, HTTP1ConnectionParameters(no_keep_alive=True, max_header_size=self.max_header_size, max_body_size=self.max_body_size, decompress=bool(self.request.decompress_response)), self._sockaddr)
        return connection

    async def _write_body(self, start_read: bool) -> None:
        if self.request.body is not None:
            self.connection.write(self.request.body)
        elif self.request.body_producer is not None:
            fut = self.request.body_producer(self.connection.write)
            if fut is not None:
                await fut
        self.connection.finish()
        if start_read:
            try:
                await self.connection.read_response(self)
            except StreamClosedError:
                if not self._handle_exception(*sys.exc_info()):
                    raise

    def _release(self) -> None:
        if self.release_callback is not None:
            release_callback = self.release_callback
            self.release_callback = None
            release_callback()

    def _run_callback(self, response: HTTPResponse) -> None:
        self._release()
        if self.final_callback is not None:
            final_callback = self.final_callback
            self.final_callback = None
            self.io_loop.add_callback(final_callback, response)

    def _handle_exception(self, typ: Type[Exception], value: Exception, tb: TracebackType) -> bool:
        if self.final_callback is not None:
            self._remove_timeout()
            if isinstance(value, StreamClosedError):
                if value.real_error is None:
                    value = HTTPStreamClosedError('Stream closed')
                else:
                    value = value.real_error
            self._run_callback(HTTPResponse(self.request, 599, error=value, request_time=self.io_loop.time() - self.start_time, start_time=self.start_wall_time))
            if hasattr(self, 'stream'):
                self.stream.close()
            return True
        else:
            return isinstance(value, StreamClosedError)

    def on_connection_close(self) -> None:
        if self.final_callback is not None:
            message = 'Connection closed'
            if self.stream.error:
                raise self.stream.error
            try:
                raise HTTPStreamClosedError(message)
            except HTTPStreamClosedError:
                self._handle_exception(*sys.exc_info())

    async def headers_received(self, first_line: httputil.ResponseStartLine, headers: Dict[str, str]) -> None:
        assert isinstance(first_line, httputil.ResponseStartLine)
        if self.request.expect_100_continue and first_line.code == 100:
            await self._write_body(False)
            return
        self.code = first_line.code
        self.reason = first_line.reason
        self.headers = headers
        if self._should_follow_redirect():
            return
        if self.request.header_callback is not None:
            self.request.header_callback('%s %s %s\r\n' % first_line)
            for k, v in self.headers.get_all():
                self.request.header_callback(f'{k}: {v}\r\n')
            self.request.header_callback('\r\n')

    def _should_follow_redirect(self) -> bool:
        if self.request.follow_redirects:
            assert self.request.max_redirects is not None
            return self.code in (301, 302, 303, 307, 308) and self.request.max_redirects > 0 and (self.headers is not None) and (self.headers.get('Location') is not None)
        return False

    def finish(self) -> None:
        assert self.code is not None
        data = b''.join(self.chunks)
        self._remove_timeout()
        original_request = getattr(self.request, 'original_request', self.request)
        if self._should_follow_redirect():
            assert isinstance(self.request, _RequestProxy)
            assert self.headers is not None
            new_request = copy.copy(self.request.request)
            new_request.url = urllib.parse.urljoin(self.request.url, self.headers['Location'])
            assert self.request.max_redirects is not None
            new_request.max_redirects = self.request.max_redirects - 1
            del new_request.headers['Host']
            if self.code == 303 and self.request.method != 'HEAD' or (self.code in (301, 302) and self.request.method == 'POST'):
                new_request.method = 'GET'
                new_request.body = None
                for h in ['Content-Length', 'Content-Type', 'Content-Encoding', 'Transfer-Encoding']:
                    try:
                        del self.request.headers[h]
                    except KeyError:
                        pass
            new_request.original_request = original_request
            final_callback = self.final_callback
            self.final_callback = None
            self._release()
            assert self.client is not None
            fut = self.client.fetch(new_request, raise_error=False)
            fut.add_done_callback(lambda f: final_callback(f.result()))
            self._on_end_request()
            return
        if self.request.streaming_callback:
            buffer = BytesIO()
        else:
            buffer = BytesIO(data)
        response = HTTPResponse(original_request, self.code, reason=getattr(self, 'reason', None), headers=self.headers, request_time=self.io_loop.time() - self.start_time, start_time=self.start_wall_time, buffer=buffer, effective_url=self.request.url)
        self._run_callback(response)
        self._on_end_request()

    def _on_end_request(self) -> None:
        self.stream.close()
if __name__ == '__main__':
    AsyncHTTPClient.configure(SimpleAsyncHTTPClient)
    main()
