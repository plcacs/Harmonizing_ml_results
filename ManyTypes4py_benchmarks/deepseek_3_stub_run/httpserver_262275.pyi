import socket
import ssl
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from tornado import httputil, iostream, netutil
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado.tcpserver import TCPServer
from tornado.util import Configurable


class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Union[
            Callable[[httputil.HTTPServerRequest], Any],
            httputil.HTTPServerConnectionDelegate,
        ],
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
        trusted_downstream: Optional[List[str]] = None,
    ) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type["HTTPServer"]: ...
    @classmethod
    def configurable_default(cls) -> Type["HTTPServer"]: ...
    async def close_all_connections(self) -> None: ...
    def handle_stream(
        self, stream: iostream.IOStream, address: Tuple[str, int]
    ) -> None: ...
    def start_request(
        self,
        server_conn: httputil.HTTPServerConnection,
        request_conn: httputil.HTTPConnection,
    ) -> httputil.HTTPMessageDelegate: ...
    def on_close(self, server_conn: httputil.HTTPServerConnection) -> None: ...


class _CallableAdapter(httputil.HTTPMessageDelegate):
    def __init__(
        self,
        request_callback: Callable[[httputil.HTTPServerRequest], Any],
        request_conn: httputil.HTTPConnection,
    ) -> None: ...
    def headers_received(
        self,
        start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine],
        headers: httputil.HTTPHeaders,
    ) -> Optional[httputil.HTTPMessageDelegate]: ...
    def data_received(self, chunk: bytes) -> Optional[httputil.HTTPMessageDelegate]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...


class _HTTPRequestContext:
    def __init__(
        self,
        stream: iostream.IOStream,
        address: Tuple[str, int],
        protocol: Optional[str],
        trusted_downstream: Optional[List[str]] = None,
    ) -> None: ...
    def __str__(self) -> str: ...
    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None: ...
    def _unapply_xheaders(self) -> None: ...


class _ProxyAdapter(httputil.HTTPMessageDelegate):
    def __init__(
        self,
        delegate: httputil.HTTPMessageDelegate,
        request_conn: httputil.HTTPConnection,
    ) -> None: ...
    def headers_received(
        self,
        start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine],
        headers: httputil.HTTPHeaders,
    ) -> Optional[httputil.HTTPMessageDelegate]: ...
    def data_received(self, chunk: bytes) -> Optional[httputil.HTTPMessageDelegate]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...
    def _cleanup(self) -> None: ...


HTTPRequest: Type[httputil.HTTPServerRequest] = ...