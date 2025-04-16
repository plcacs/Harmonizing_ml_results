import socket
import ssl
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable, Set, cast
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
if typing.TYPE_CHECKING:
    from typing import Set

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def initialize(self, request_callback, no_keep_alive=False, xheaders=False, ssl_options=None, protocol=None, decompress_request=False, chunk_size=None, max_header_size=None, idle_connection_timeout=None, body_timeout=None, max_body_size=None, max_buffer_size=None, trusted_downstream=None):
        self.request_callback = request_callback
        self.xheaders = xheaders
        self.protocol = protocol
        self.conn_params = HTTP1ConnectionParameters(decompress=decompress_request, chunk_size=chunk_size, max_header_size=max_header_size, header_timeout=idle_connection_timeout or 3600, max_body_size=max_body_size, body_timeout=body_timeout, no_keep_alive=no_keep_alive)
        TCPServer.__init__(self, ssl_options=ssl_options, max_buffer_size=max_buffer_size, read_chunk_size=chunk_size)
        self._connections: Set[HTTP1ServerConnection] = set()
        self.trusted_downstream = trusted_downstream

    @classmethod
    def configurable_base(cls):
        return HTTPServer

    @classmethod
    def configurable_default(cls):
        return HTTPServer

    async def close_all_connections(self) -> None:
        while self._connections:
            conn = next(iter(self._connections))
            await conn.close()

    def handle_stream(self, stream, address):
        context = _HTTPRequestContext(stream, address, self.protocol, self.trusted_downstream)
        conn = HTTP1ServerConnection(stream, self.conn_params, context)
        self._connections.add(conn)
        conn.start_serving(self)

    def start_request(self, server_conn, request_conn):
        if isinstance(self.request_callback, httputil.HTTPServerConnectionDelegate):
            delegate = self.request_callback.start_request(server_conn, request_conn)
        else:
            delegate = _CallableAdapter(self.request_callback, request_conn)
        if self.xheaders:
            delegate = _ProxyAdapter(delegate, request_conn)
        return delegate

    def on_close(self, server_conn):
        self._connections.remove(cast(HTTP1ServerConnection, server_conn))

class _CallableAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, request_callback, request_conn):
        self.connection = request_conn
        self.request_callback = request_callback
        self.request: Optional[httputil.HTTPServerRequest] = None
        self.delegate = None
        self._chunks: List[bytes] = []

    def headers_received(self, start_line, headers):
        self.request = httputil.HTTPServerRequest(connection=self.connection, start_line=cast(httputil.RequestStartLine, start_line), headers=headers)
        return None

    def data_received(self, chunk):
        self._chunks.append(chunk)
        return None

    def finish(self):
        assert self.request is not None
        self.request.body = b''.join(self._chunks)
        self.request._parse_body()
        self.request_callback(self.request)

    def on_connection_close(self):
        del self._chunks

class _HTTPRequestContext:

    def __init__(self, stream, address, protocol, trusted_downstream=None):
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
        self.trusted_downstream = set(trusted_downstream or [])

    def __str__(self):
        if self.address_family in (socket.AF_INET, socket.AF_INET6):
            return self.remote_ip
        elif isinstance(self.address, bytes):
            return native_str(self.address)
        else:
            return str(self.address)

    def _apply_xheaders(self, headers):
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

    def _unapply_xheaders(self):
        self.remote_ip = self._orig_remote_ip
        self.protocol = self._orig_protocol

class _ProxyAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, delegate, request_conn):
        self.connection = request_conn
        self.delegate = delegate

    def headers_received(self, start_line, headers):
        self.connection.context._apply_xheaders(headers)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk):
        return self.delegate.data_received(chunk)

    def finish(self):
        self.delegate.finish()
        self._cleanup()

    def on_connection_close(self):
        self.delegate.on_connection_close()
        self._cleanup()

    def _cleanup(self):
        self.connection.context._unapply_xheaders()
HTTPRequest = httputil.HTTPServerRequest