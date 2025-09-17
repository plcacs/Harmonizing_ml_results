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
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable, Set

if typing.TYPE_CHECKING:
    from tornado.iostream import IOStream
    from tornado.http1connection import HTTP1ServerConnection

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def initialize(
        self,
        request_callback: Union[Callable[[httputil.HTTPServerRequest], Any], httputil.HTTPServerConnectionDelegate],
        no_keep_alive: bool = False,
        xheaders: bool = False,
        ssl_options: Optional[Union[ssl.SSLContext, Dict[str, Any]]] = None,
        protocol: Optional[str] = None,
        decompress_request: bool = False,
        chunk_size: Optional[int] = None,
        max_header_size: Optional[int] = None,
        idle_connection_timeout: Optional[float] = None,
        body_timeout: Optional[float] = None,
        max_body_size: Optional[int] = None,
        max_buffer_size: Optional[int] = None,
        trusted_downstream: Optional[Union[Set[str], List[str]]] = None
    ) -> None:
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
        self._connections: Set[HTTP1ServerConnection] = set()
        self.trusted_downstream = trusted_downstream

    @classmethod
    def configurable_base(cls) -> Type["HTTPServer"]:
        return HTTPServer

    @classmethod
    def configurable_default(cls) -> Type["HTTPServer"]:
        return HTTPServer

    async def close_all_connections(self) -> None:
        while self._connections:
            conn = next(iter(self._connections))
            await conn.close()

    def handle_stream(self, stream: "iostream.IOStream", address: Any) -> None:
        context = _HTTPRequestContext(stream, address, self.protocol, self.trusted_downstream)
        conn = HTTP1ServerConnection(stream, self.conn_params, context)
        self._connections.add(conn)
        conn.start_serving(self)

    def start_request(self, server_conn: HTTP1ServerConnection, request_conn: Any) -> httputil.HTTPMessageDelegate:
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
    def __init__(
        self,
        request_callback: Callable[[httputil.HTTPServerRequest], Any],
        request_conn: Any
    ) -> None:
        self.connection = request_conn
        self.request_callback = request_callback
        self.request: Optional[httputil.HTTPServerRequest] = None
        self.delegate: Optional[Any] = None
        self._chunks: List[bytes] = []

    def headers_received(self, start_line: Any, headers: Dict[str, str]) -> Optional[Any]:
        self.request = httputil.HTTPServerRequest(
            connection=self.connection,
            start_line=typing.cast(httputil.RequestStartLine, start_line),
            headers=headers
        )
        return None

    def data_received(self, chunk: bytes) -> Optional[Any]:
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
    def __init__(
        self,
        stream: "iostream.IOStream",
        address: Any,
        protocol: Optional[str],
        trusted_downstream: Optional[Union[Set[str], List[str]]] = None
    ) -> None:
        self.address = address
        if stream.socket is not None:
            self.address_family: Optional[int] = stream.socket.family
        else:
            self.address_family = None
        if self.address_family in (socket.AF_INET, socket.AF_INET6) and address is not None:
            self.remote_ip: str = address[0]
        else:
            self.remote_ip = '0.0.0.0'
        if protocol:
            self.protocol = protocol
        elif isinstance(stream, iostream.SSLIOStream):
            self.protocol = 'https'
        else:
            self.protocol = 'http'
        self._orig_remote_ip: str = self.remote_ip
        self._orig_protocol: str = self.protocol
        self.trusted_downstream: Set[str] = set(trusted_downstream or [])

    def __str__(self) -> str:
        if self.address_family in (socket.AF_INET, socket.AF_INET6):
            return self.remote_ip
        elif isinstance(self.address, bytes):
            return native_str(self.address)
        else:
            return str(self.address)

    def _apply_xheaders(self, headers: Dict[str, Any]) -> None:
        ip = headers.get('X-Forwarded-For', self.remote_ip)
        for ip in (cand.strip() for cand in ip.split(',')[::-1]):
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
        self.remote_ip = self._orig_remote_ip
        self.protocol = self._orig_protocol


class _ProxyAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: Any) -> None:
        self.connection = request_conn
        self.delegate = delegate

    def headers_received(self, start_line: Any, headers: Dict[str, str]) -> Optional[Any]:
        self.connection.context._apply_xheaders(headers)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk: bytes) -> Optional[Any]:
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