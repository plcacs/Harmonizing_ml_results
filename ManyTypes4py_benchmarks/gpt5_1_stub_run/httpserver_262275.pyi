from typing import Any, Awaitable, Type
from tornado import httputil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
from tornado.httputil import HTTPServerRequest as HTTPRequest

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Any,
        no_keep_alive: Any = ...,
        xheaders: Any = ...,
        ssl_options: Any = ...,
        protocol: Any = ...,
        decompress_request: Any = ...,
        chunk_size: Any = ...,
        max_header_size: Any = ...,
        idle_connection_timeout: Any = ...,
        body_timeout: Any = ...,
        max_body_size: Any = ...,
        max_buffer_size: Any = ...,
        trusted_downstream: Any = ...,
    ) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type["HTTPServer"]: ...
    @classmethod
    def configurable_default(cls) -> Type["HTTPServer"]: ...
    async def close_all_connections(self) -> Awaitable[None]: ...
    def handle_stream(self, stream: Any, address: Any) -> None: ...
    def start_request(self, server_conn: Any, request_conn: Any) -> httputil.HTTPMessageDelegate: ...
    def on_close(self, server_conn: Any) -> None: ...