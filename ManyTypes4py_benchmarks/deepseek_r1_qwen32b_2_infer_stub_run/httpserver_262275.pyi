"""Stub file for 'httpserver_262275' module."""

from __future__ import annotations
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
    Awaitable,
)
from tornado import httputil, iostream
from tornado.tcpserver import TCPServer
from tornado.util import Configurable

HTTPRequest = httputil.HTTPServerRequest


class HTTPServer(TCPServer, Configurable, httputil.HTTPServerConnectionDelegate):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def initialize(
        self,
        request_callback: Union[Callable[[HTTPServerRequest], Any], httputil.HTTPServerConnectionDelegate],
        no_keep_alive: bool = ...,
        xheaders: bool = ...,
        ssl_options: Optional[ssl.SSLContext] = ...,
        protocol: Optional[str] = ...,
        decompress_request: bool = ...,
        chunk_size: Optional[int] = ...,
        max_header_size: Optional[int] = ...,
        idle_connection_timeout: Optional[float] = ...,
        body_timeout: Optional[float] = ...,
        max_body_size: Optional[int] = ...,
        max_buffer_size: Optional[int] = ...,
        trusted_downstream: Optional[Set[str]] = ...,
    ) -> None:
        ...

    @classmethod
    def configurable_base(cls) -> Type[HTTPServer]:
        ...

    @classmethod
    def configurable_default(cls) -> HTTPServer:
        ...

    async def close_all_connections(self) -> None:
        ...

    def handle_stream(self, stream: iostream.IOStream, address: Tuple[str, int]) -> None:
        ...

    def start_request(
        self,
        server_conn: httputil.HTTP1ServerConnection,
        request_conn: httputil.HTTP1Connection,
    ) -> httputil.HTTPMessageDelegate:
        ...

    def on_close(self, server_conn: httputil.HTTP1ServerConnection) -> None:
        ...


class _CallableAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, request_callback: Callable[[HTTPServerRequest], Any], request_conn: httputil.HTTP1Connection) -> None:
        ...

    def headers_received(
        self,
        start_line: httputil.RequestStartLine,
        headers: Dict[str, str],
    ) -> None:
        ...

    def data_received(self, chunk: bytes) -> None:
        ...

    def finish(self) -> None:
        ...

    def on_connection_close(self) -> None:
        ...


class _HTTPRequestContext:
    def __init__(
        self,
        stream: iostream.IOStream,
        address: Tuple[str, int],
        protocol: str,
        trusted_downstream: Optional[Set[str]] = ...,
    ) -> None:
        ...

    def __str__(self) -> str:
        ...

    def _apply_xheaders(self, headers: Dict[str, str]) -> None:
        ...

    def _unapply_xheaders(self) -> None:
        ...


class _ProxyAdapter(httputil.HTTPMessageDelegate):
    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: httputil.HTTP1Connection) -> None:
        ...

    def headers_received(
        self,
        start_line: httputil.RequestStartLine,
        headers: Dict[str, str],
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