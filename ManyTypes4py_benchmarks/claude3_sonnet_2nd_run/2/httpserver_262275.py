"""A non-blocking, single-threaded HTTP server.

Typical applications have little direct interaction with the `HTTPServer`
class except to start a server at the beginning of the process
(and even that is often done indirectly via `tornado.web.Application.listen`).

.. versionchanged:: 4.0

   The ``HTTPRequest`` class that used to live in this module has been moved
   to `tornado.httputil.HTTPServerRequest`.  The old name remains as an alias.
"""
import socket
import ssl
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
import typing
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable
if typing.TYPE_CHECKING:
    from typing import Set

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    """A non-blocking, single-threaded HTTP server.

    A server is defined by a subclass of `.HTTPServerConnectionDelegate`,
    or, for backwards compatibility, a callback that takes an
    `.HTTPServerRequest` as an argument. The delegate is usually a
    `tornado.web.Application`.

    `HTTPServer` supports keep-alive connections by default
    (automatically for HTTP/1.1, or for HTTP/1.0 when the client
    requests ``Connection: keep-alive``).

    If ``xheaders`` is ``True``, we support the
    ``X-Real-Ip``/``X-Forwarded-For`` and
    ``X-Scheme``/``X-Forwarded-Proto`` headers, which override the
    remote IP and URI scheme/protocol for all requests.  These headers
    are useful when running Tornado behind a reverse proxy or load
    balancer.  The ``protocol`` argument can also be set to ``https``
    if Tornado is run behind an SSL-decoding proxy that does not set one of
    the supported ``xheaders``.

    By default, when parsing the ``X-Forwarded-For`` header, Tornado will
    select the last (i.e., the closest) address on the list of hosts as the
    remote host IP address.  To select the next server in the chain, a list of
    trusted downstream hosts may be passed as the ``trusted_downstream``
    argument.  These hosts will be skipped when parsing the ``X-Forwarded-For``
    header.

    To make this server serve SSL traffic, send the ``ssl_options`` keyword
    argument with an `ssl.SSLContext` object. For compatibility with older
    versions of Python ``ssl_options`` may also be a dictionary of keyword
    arguments for the `ssl.SSLContext.wrap_socket` method.::

       ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
       ssl_ctx.load_cert_chain(os.path.join(data_dir, "mydomain.crt"),
                               os.path.join(data_dir, "mydomain.key"))
       HTTPServer(application, ssl_options=ssl_ctx)

    `HTTPServer` initialization follows one of three patterns (the
    initialization methods are defined on `tornado.tcpserver.TCPServer`):

    1. `~tornado.tcpserver.TCPServer.listen`: single-process::

            async def main():
                server = HTTPServer()
                server.listen(8888)
                await asyncio.Event().wait()

            asyncio.run(main())

       In many cases, `tornado.web.Application.listen` can be used to avoid
       the need to explicitly create the `HTTPServer`.

       While this example does not create multiple processes on its own, when
       the ``reuse_port=True`` argument is passed to ``listen()`` you can run
       the program multiple times to create a multi-process service.

    2. `~tornado.tcpserver.TCPServer.add_sockets`: multi-process::

            sockets = bind_sockets(8888)
            tornado.process.fork_processes(0)
            async def post_fork_main():
                server = HTTPServer()
                server.add_sockets(sockets)
                await asyncio.Event().wait()
            asyncio.run(post_fork_main())

       The ``add_sockets`` interface is more complicated, but it can be used with
       `tornado.process.fork_processes` to run a multi-process service with all
       worker processes forked from a single parent.  ``add_sockets`` can also be
       used in single-process servers if you want to create your listening
       sockets in some way other than `~tornado.netutil.bind_sockets`.

       Note that when using this pattern, nothing that touches the event loop
       can be run before ``fork_processes``.

    3. `~tornado.tcpserver.TCPServer.bind`/`~tornado.tcpserver.TCPServer.start`:
       simple **deprecated** multi-process::

            server = HTTPServer()
            server.bind(8888)
            server.start(0)  # Forks multiple sub-processes
            IOLoop.current().start()

       This pattern is deprecated because it requires interfaces in the
       `asyncio` module that have been deprecated since Python 3.10. Support for
       creating multiple processes in the ``start`` method will be removed in a
       future version of Tornado.

    .. versionchanged:: 4.0
       Added ``decompress_request``, ``chunk_size``, ``max_header_size``,
       ``idle_connection_timeout``, ``body_timeout``, ``max_body_size``
       arguments.  Added support for `.HTTPServerConnectionDelegate`
       instances as ``request_callback``.

    .. versionchanged:: 4.1
       `.HTTPServerConnectionDelegate.start_request` is now called with
       two arguments ``(server_conn, request_conn)`` (in accordance with the
       documentation) instead of one ``(request_conn)``.

    .. versionchanged:: 4.2
       `HTTPServer` is now a subclass of `tornado.util.Configurable`.

    .. versionchanged:: 4.5
       Added the ``trusted_downstream`` argument.

    .. versionchanged:: 5.0
       The ``io_loop`` argument has been removed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def initialize(self, 
                  request_callback: Union[httputil.HTTPServerConnectionDelegate, Callable[[httputil.HTTPServerRequest], None]], 
                  no_keep_alive: bool = False, 
                  xheaders: bool = False, 
                  ssl_options: Union[Dict[str, Any], ssl.SSLContext, None] = None, 
                  protocol: Optional[str] = None, 
                  decompress_request: bool = False, 
                  chunk_size: Optional[int] = None, 
                  max_header_size: Optional[int] = None, 
                  idle_connection_timeout: Optional[float] = None, 
                  body_timeout: Optional[float] = None, 
                  max_body_size: Optional[int] = None, 
                  max_buffer_size: Optional[int] = None, 
                  trusted_downstream: Optional[List[str]] = None) -> None:
        self.request_callback = request_callback
        self.xheaders = xheaders
        self.protocol = protocol
        self.conn_params = HTTP1ConnectionParameters(
            decompress=decompress_request,
            chunk_size=chunk_size,
            max_header_size=max_header_size,
            header_timeout=idle_connection_timeout or 3600,
            max_body_size=max_body_size,
            body_timeout=body_timeout,
            no_keep_alive=no_keep_alive
        )
        TCPServer.__init__(self, ssl_options=ssl_options, max_buffer_size=max_buffer_size, read_chunk_size=chunk_size)
        self._connections: "Set[HTTP1ServerConnection]" = set()
        self.trusted_downstream = trusted_downstream

    @classmethod
    def configurable_base(cls) -> Type[Configurable]:
        return HTTPServer

    @classmethod
    def configurable_default(cls) -> Type[Configurable]:
        return HTTPServer

    async def close_all_connections(self) -> None:
        """Close all open connections and asynchronously wait for them to finish.

        This method is used in combination with `~.TCPServer.stop` to
        support clean shutdowns (especially for unittests). Typical
        usage would call ``stop()`` first to stop accepting new
        connections, then ``await close_all_connections()`` to wait for
        existing connections to finish.

        This method does not currently close open websocket connections.

        Note that this method is a coroutine and must be called with ``await``.

        """
        while self._connections:
            conn = next(iter(self._connections))
            await conn.close()

    def handle_stream(self, stream: iostream.IOStream, address: Tuple[str, int]) -> None:
        context = _HTTPRequestContext(stream, address, self.protocol, self.trusted_downstream)
        conn = HTTP1ServerConnection(stream, self.conn_params, context)
        self._connections.add(conn)
        conn.start_serving(self)

    def start_request(self, 
                     server_conn: HTTP1ServerConnection, 
                     request_conn: httputil.HTTPConnection) -> httputil.HTTPMessageDelegate:
        if isinstance(self.request_callback, httputil.HTTPServerConnectionDelegate):
            delegate = self.request_callback.start_request(server_conn, request_conn)
        else:
            delegate = _CallableAdapter(self.request_callback, request_conn)
        if self.xheaders:
            delegate = _ProxyAdapter(delegate, request_conn)
        return delegate

    def on_close(self, server_conn: HTTP1ServerConnection) -> None:
        self._connections.remove(typing.cast(HTTP1ServerConnection, server_conn))

class _CallableAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, 
                request_callback: Callable[[httputil.HTTPServerRequest], None], 
                request_conn: httputil.HTTPConnection) -> None:
        self.connection = request_conn
        self.request_callback = request_callback
        self.request: Optional[httputil.HTTPServerRequest] = None
        self.delegate: Optional[httputil.HTTPMessageDelegate] = None
        self._chunks: List[bytes] = []

    def headers_received(self, 
                        start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], 
                        headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        self.request = httputil.HTTPServerRequest(
            connection=self.connection,
            start_line=typing.cast(httputil.RequestStartLine, start_line),
            headers=headers
        )
        return None

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        self._chunks.append(chunk)
        return None

    def finish(self) -> None:
        assert self.request is not None
        self.request.body = b''.join(self._chunks)
        self.request._parse_body()
        self.request_callback(self.request)

    def on_connection_close(self) -> None:
        del self._chunks

class _HTTPRequestContext:

    def __init__(self, 
                stream: iostream.IOStream, 
                address: Union[Tuple[str, int], str], 
                protocol: Optional[str], 
                trusted_downstream: Optional[List[str]] = None) -> None:
        self.address = address
        if stream.socket is not None:
            self.address_family = stream.socket.family
        else:
            self.address_family = None
        if self.address_family in (socket.AF_INET, socket.AF_INET6) and address is not None:
            self.remote_ip = address[0]
        else:
            self.remote_ip = '0.0.0.0'
        if protocol:
            self.protocol = protocol
        elif isinstance(stream, iostream.SSLIOStream):
            self.protocol = 'https'
        else:
            self.protocol = 'http'
        self._orig_remote_ip = self.remote_ip
        self._orig_protocol = self.protocol
        self.trusted_downstream: "Set[str]" = set(trusted_downstream or [])

    def __str__(self) -> str:
        if self.address_family in (socket.AF_INET, socket.AF_INET6):
            return self.remote_ip
        elif isinstance(self.address, bytes):
            return native_str(self.address)
        else:
            return str(self.address)

    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None:
        """Rewrite the ``remote_ip`` and ``protocol`` fields."""
        ip = headers.get('X-Forwarded-For', self.remote_ip)
        for ip in (cand.strip() for cand in reversed(ip.split(','))):
            if ip not in self.trusted_downstream:
                break
        ip = headers.get('X-Real-Ip', ip)
        if netutil.is_valid_ip(ip):
            self.remote_ip = ip
        proto_header = headers.get('X-Scheme', headers.get('X-Forwarded-Proto', self.protocol))
        if proto_header:
            proto_header = proto_header.split(',')[-1].strip()
        if proto_header in ('http', 'https'):
            self.protocol = proto_header

    def _unapply_xheaders(self) -> None:
        """Undo changes from `_apply_xheaders`.

        Xheaders are per-request so they should not leak to the next
        request on the same connection.
        """
        self.remote_ip = self._orig_remote_ip
        self.protocol = self._orig_protocol

class _ProxyAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, 
                delegate: httputil.HTTPMessageDelegate, 
                request_conn: httputil.HTTPConnection) -> None:
        self.connection = request_conn
        self.delegate = delegate

    def headers_received(self, 
                        start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], 
                        headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        self.connection.context._apply_xheaders(headers)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        return self.delegate.data_received(chunk)

    def finish(self) -> None:
        self.delegate.finish()
        self._cleanup()

    def on_connection_close(self) -> None:
        self.delegate.on_connection_close()
        self._cleanup()

    def _cleanup(self) -> None:
        self.connection.context._unapply_xheaders()

HTTPRequest = httputil.HTTPServerRequest
