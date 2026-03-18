from typing import Any, Type
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
from tornado import httputil

class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize(
        self,
        request_callback: Any,
        no_keep_alive: bool = ...,
        xheaders: bool = ...,
        ssl_options: Any = ...,
        protocol: Any = ...,
        decompress_request: bool = ...,
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
    async def close_all_connections(self) -> None: ...
    def handle_stream(self, stream: Any, address: Any) -> None: ...
    def start_request(self, server_conn: Any, request_conn: Any) -> Any: ...
    def on_close(self, server_conn: Any) -> None: ...

HTTPRequest = httputil.HTTPServerRequest