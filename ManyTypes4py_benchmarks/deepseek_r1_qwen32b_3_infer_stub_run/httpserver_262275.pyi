"""A non-blocking, single-threaded HTTP server."""

import socket
import ssl
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Callable,
    Awaitable,
)
from tornado import httputil, iostream, netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
import ssl

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    """A non-blocking, single-threaded HTTP server."""
    request_callback: Callable[[httputil.HTTPServerRequest], None]
    xheaders: bool
    protocol: Optional[str]
    conn_params: HTTP1ConnectionParameters
    _connections: Set[HTTP1ServerConnection]
    trusted_downstream: Optional[Set[str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def initialize(
        self,
        request_callback: Callable[[httputil.HTTPServerRequest], None],
        no_keep_alive: bool = False,
        xheaders: bool = False,
        ssl_options: Union[ssl.SSLContext, Dict[str, Any]] = None,
        protocol: Optional[str] = None,
        decompress_request: bool = False,
        chunk_size: Optional[int] = None,
        max_header_size: Optional[int] = None,
        idle_connection_timeout: Optional[int] = None,
        body_timeout: Optional[int] = None,
        max_body_size: Optional[int] = None,
        max_buffer_size: Optional[int] = None,
        trusted_downstream: Optional[Set[str]] = None,
    ) -> None:
        ...

    @classmethod
    def configurable_base(cls) -> type:
        ...

    @classmethod
    def configurable_default(cls) -> Any:
        ...

    async def close_all_connections(self) -> None:
        ...

    def handle_stream(self, stream: iostream.IOStream, address: Tuple[str, int]) -> None:
        ...

    def start_request(
        self,
        server_conn: Any,
        request_conn: Any,
    ) -> httputil.HTTPMessageDelegate:
        ...

    def on_close(self, server_conn: HTTP1ServerConnection) -> None:
        ...

class _CallableAdapter(httputil.HTTPMessageDelegate):
    connection: Any
    request_callback: Callable[[httputil.HTTPServerRequest], None]
    request: Optional[httputil.HTTPServerRequest]
    delegate: Any
    _chunks: List[bytes]

    def __init__(
        self,
        request_callback: Callable[[httputil.HTTPServerRequest], None],
        request_conn: Any,
    ) -> None:
        ...

    def headers_received(
        self,
        start_line: httputil.RequestStartLine,
        headers: httputil.HTTPHeaders,
    ) -> None:
        ...

    def data_received(self, chunk: bytes) -> None:
        ...

    def finish(self) -> None:
        ...

    def on_connection_close(self) -> None:
        ...

class _HTTPRequestContext:
    address: Tuple[str, int]
    address_family: Optional[int]
    remote_ip: str
    protocol: str
    _orig_remote_ip: str
    _orig_protocol: str
    trusted_downstream: Set[str]

    def __init__(
        self,
        stream: iostream.IOStream,
        address: Tuple[str, int],
        protocol: Optional[str],
        trusted_downstream: Optional[Set[str]] = None,
    ) -> None:
        ...

    def __str__(self) -> str:
        ...

    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None:
        ...

    def _unapply_xheaders(self) -> None:
        ...

class _ProxyAdapter(httputil.HTTPMessageDelegate):
    connection: Any
    delegate: httputil.HTTPMessageDelegate

    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: Any) -> None:
        ...

    def headers_received(
        self,
        start_line: httputil.RequestStartLine,
        headers: httputil.HTTPHeaders,
    ) -> None:
        ...

    def data_received(self, chunk: bytes) -> None:
        ...

    def finish(self) -> None:
        ...

    def on_connection_close(self) -> None:
        ...

    def _cleanup(self) -> None:
        ...

HTTPRequest = httputil.HTTPServerRequest