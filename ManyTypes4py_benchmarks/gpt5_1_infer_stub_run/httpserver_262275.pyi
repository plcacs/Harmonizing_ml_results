from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Set, Tuple, Type, Union
import ssl
from tornado import httputil, iostream
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters, HTTP1ServerConnection
from tornado.tcpserver import TCPServer
from tornado.util import Configurable

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    request_callback: Union[httputil.HTTPServerConnectionDelegate, Callable[[httputil.HTTPServerRequest], None]]
    xheaders: bool
    protocol: Optional[str]
    conn_params: HTTP1ConnectionParameters
    _connections: Set[HTTP1ServerConnection]
    trusted_downstream: Optional[Iterable[str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Union[httputil.HTTPServerConnectionDelegate, Callable[[httputil.HTTPServerRequest], None]],
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
        trusted_downstream: Optional[Iterable[str]] = ...,
    ) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type["HTTPServer"]: ...
    @classmethod
    def configurable_default(cls) -> Type["HTTPServer"]: ...
    async def close_all_connections(self) -> None: ...
    def handle_stream(
        self,
        stream: iostream.BaseIOStream,
        address: Optional[Union[Tuple[str, int], Tuple[str, int, int, int], str, bytes]],
    ) -> None: ...
    def start_request(self, server_conn: HTTP1ServerConnection, request_conn: HTTP1Connection) -> httputil.HTTPMessageDelegate: ...
    def on_close(self, server_conn: HTTP1ServerConnection) -> None: ...

class _CallableAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, request_callback: Callable[[httputil.HTTPServerRequest], None], request_conn: HTTP1Connection) -> None: ...
    def headers_received(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]: ...
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...

class _HTTPRequestContext:
    address: Optional[Union[Tuple[str, int], Tuple[str, int, int, int], str, bytes]]
    address_family: Optional[int]
    remote_ip: str
    protocol: str
    trusted_downstream: Set[str]

    def __init__(
        self,
        stream: iostream.BaseIOStream,
        address: Optional[Union[Tuple[str, int], Tuple[str, int, int, int], str, bytes]],
        protocol: Optional[str],
        trusted_downstream: Optional[Iterable[str]] = ...,
    ) -> None: ...
    def __str__(self) -> str: ...
    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None: ...
    def _unapply_xheaders(self) -> None: ...

class _ProxyAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: HTTP1Connection) -> None: ...
    def headers_received(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]: ...
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...
    def _cleanup(self) -> None: ...

HTTPRequest = httputil.HTTPServerRequest