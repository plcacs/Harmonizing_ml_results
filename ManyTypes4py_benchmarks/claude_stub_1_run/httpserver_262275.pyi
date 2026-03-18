```python
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

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    request_callback: Any
    xheaders: bool
    protocol: Optional[str]
    conn_params: HTTP1ConnectionParameters
    _connections: Set[HTTP1ServerConnection]
    trusted_downstream: Optional[Set[str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Any,
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
    def configurable_base(cls) -> Type[HTTPServer]: ...
    @classmethod
    def configurable_default(cls) -> Type[HTTPServer]: ...
    async def close_all_connections(self) -> None: ...
    def handle_stream(self, stream: Any, address: Any) -> None: ...
    def start_request(
        self, server_conn: Any, request_conn: Any
    ) -> httputil.HTTPMessageDelegate: ...
    def on_close(self, server_conn: Any) -> None: ...

class _CallableAdapter(httputil.HTTPMessageDelegate):
    connection: Any
    request_callback: Callable[[httputil.HTTPServerRequest], None]
    request: Optional[httputil.HTTPServerRequest]
    delegate: Optional[httputil.HTTPMessageDelegate]
    _chunks: List[bytes]

    def __init__(
        self, request_callback: Callable[[httputil.HTTPServerRequest], None], request_conn: Any
    ) -> None: ...
    def headers_received(
        self, start_line: Any, headers: httputil.HTTPHeaders
    ) -> Optional[int]: ...
    def data_received(self, chunk: bytes) -> Optional[int]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...

class _HTTPRequestContext:
    address: Any
    address_family: Optional[int]
    remote_ip: str
    protocol: str
    _orig_remote_ip: str
    _orig_protocol: str
    trusted_downstream: Set[str]

    def __init__(
        self,
        stream: Any,
        address: Any,
        protocol: Optional[str] = ...,
        trusted_downstream: Optional[List[str]] = ...,
    ) -> None: ...
    def __str__(self) -> str: ...
    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None: ...
    def _unapply_xheaders(self) -> None: ...

class _ProxyAdapter(httputil.HTTPMessageDelegate):
    connection: Any
    delegate: httputil.HTTPMessageDelegate

    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: Any) -> None: ...
    def headers_received(
        self, start_line: Any, headers: httputil.HTTPHeaders
    ) -> Optional[int]: ...
    def data_received(self, chunk: bytes) -> Optional[int]: ...
    def finish(self) -> None: ...
    def on_connection_close(self) -> None: ...
    def _cleanup(self) -> None: ...

HTTPRequest: Type[httputil.HTTPServerRequest]
```