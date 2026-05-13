import socket
import ssl
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable, Set
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Union[Callable[[httputil.HTTPServerRequest], Any], httputil.HTTPServerConnectionDelegate],
        no_keep_alive: bool = ...,
        xheaders: bool = ...,
        ssl_options: Optional[Union[ssl.SSLContext, Dict[str, Any]]] = ...,
        protocol: Optional[str] = ...,
        decompress_request: bool = ...,
        chunk_size: Optional[int] = ...,
        max_header_size: Optional[int] = ...,
        idle_connection_timeout: Optional[float] = ...,
        body_timeout: Optional[float] = ...,
        max_body_size: Optional[int] = ...,
        max_buffer_size: Optional[int] = ...,
        trusted_downstream: Optional[List[str]] = ...,
    ) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type['HTTPServer']: ...
    @classmethod
    def configurable_default(cls) -> Type['HTTPServer']: ...
    async def close_all_connections(self) -> None: ...
    def handle_stream(self, stream: iostream.IOStream, address: Tuple[str, int]) -> None: ...
    def start_request(self, server_conn: httputil.HTTPServerConnection, request_conn: httputil.HTTPConnection) -> httputil.HTTPMessageDelegate: ...
    def on_close(self, server_conn: httputil.HTTPServerConnection) -> None: ...
    request_callback: Union[Callable[[httputil.HTTPServerRequest], Any], httputil.HTTPServerConnectionDelegate]
    xheaders: bool
    protocol: Optional[str]
    conn_params: HTTP1ConnectionParameters
    trusted_downstream: Optional[List[str]]
    _connections: Set[HTTP1ServerConnection]

class _CallableAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, request_callback: Callable[[httputil.HTTPServerRequest], Any], request_conn: httputil.HTTPConnection) -> None: ...
    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[httputil.HTTPMessageDelegate]: ...
    def data_received(self, chunk: bytes) -> Optional[httputil.HTTPMessageDelegate]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...
    connection: httputil.HTTPConnection
    request_callback: Callable[[httputil.HTTPServerRequest], Any]
    request: Optional[httputil.HTTPServerRequest]
    delegate: Optional[httputil.HTTPMessageDelegate]
    _chunks: List[bytes]

class _HTTPRequestContext:
    def __init__(self, stream: iostream.IOStream, address: Tuple[str, int], protocol: Optional[str], trusted_downstream: Optional[List[str]] = ...) -> None: ...
    def __str__(self) -> str: ...
    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None: ...
    def _unapply_xheaders(self) -> None: ...
    address: Tuple[str, int]
    address_family: Optional[int]
    remote_ip: str
    protocol: str
    _orig_remote_ip: str
    _orig_protocol: str
    trusted_downstream: Set[str]

class _ProxyAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: httputil.HTTPConnection) -> None: ...
    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[httputil.HTTPMessageDelegate]: ...
    def data_received(self, chunk: bytes) -> Optional[httputil.HTTPMessageDelegate]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...
    def _cleanup(self) -> None: ...
    connection: httputil.HTTPConnection
    delegate: httputil.HTTPMessageDelegate

HTTPRequest = httputil.HTTPServerRequest