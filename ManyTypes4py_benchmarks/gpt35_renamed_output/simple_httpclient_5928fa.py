from tornado.escape import _unicode
from tornado import gen, version
from tornado.httpclient import HTTPResponse, HTTPError, AsyncHTTPClient, main, _RequestProxy, HTTPRequest
from tornado import httputil
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError, IOStream
from tornado.netutil import Resolver, OverrideResolver, _client_ssl_defaults, is_valid_ip
from tornado.log import gen_log
from tornado.tcpclient import TCPClient
import base64
import collections
import copy
import functools
import re
import socket
import ssl
import sys
import time
from io import BytesIO
import urllib.parse
from typing import Dict, Any, Callable, Optional, Type, Union
from types import TracebackType
import typing
if typing.TYPE_CHECKING:
    from typing import Deque, Tuple, List


class HTTPTimeoutError(HTTPError):
    def __init__(self, message: str):
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or 'Timeout'


class HTTPStreamClosedError(HTTPError):
    def __init__(self, message: str):
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or 'Stream closed'


class SimpleAsyncHTTPClient(AsyncHTTPClient):
    def func_x06utst0(self, max_clients: int = 10, hostname_mapping: Optional[Dict[str, str]] = None,
                      max_buffer_size: int = 104857600, resolver: Optional[Resolver] = None,
                      defaults: Optional[Dict[str, Any]] = None, max_header_size: Optional[int] = None,
                      max_body_size: Optional[int] = None) -> None:
        ...

    def func_swp5jinx(self) -> None:
        ...

    def func_tmt4th5a(self, request: HTTPRequest, callback: Callable) -> None:
        ...

    def func_1drxodvj(self) -> None:
        ...

    def func_8m0rm0ww(self) -> Type[HTTP1Connection]:
        ...

    def func_hepzmc7y(self, request: HTTPRequest, release_callback: Callable, final_callback: Callable) -> None:
        ...

    def func_i6qdxtg5(self, key: object) -> None:
        ...

    def func_wdph9s63(self, key: object) -> None:
        ...

    def func_pumid49l(self, key: object, info: Optional[str] = None) -> None:
        ...


class _HTTPConnection(httputil.HTTPMessageDelegate):
    def __init__(self, client: SimpleAsyncHTTPClient, request: HTTPRequest, release_callback: Callable,
                 final_callback: Callable, max_buffer_size: int, tcp_client: TCPClient, max_header_size: int,
                 max_body_size: int) -> None:
        ...

    async def func_a1rdrscx(self) -> None:
        ...

    def func_s6l34e51(self, scheme: str) -> Optional[ssl.SSLContext]:
        ...

    def func_pumid49l(self, info: Optional[str] = None) -> None:
        ...

    def func_wdph9s63(self) -> None:
        ...

    def func_o9t0deyr(self, stream: IOStream) -> HTTP1Connection:
        ...

    async def func_4p3rx0dv(self, start_read: bool) -> None:
        ...

    def func_4dtk83z2(self) -> None:
        ...

    def func_mxtasp9n(self, response: HTTPResponse) -> None:
        ...

    def func_fez80zj8(self, typ: Type, value: Any, tb: TracebackType) -> bool:
        ...

    def func_a4rx8kj8(self) -> None:
        ...

    async def func_0yf0hrbg(self, first_line: httputil.ResponseStartLine, headers: httputil.HTTPHeaders) -> None:
        ...

    def func_dwq39xh5(self) -> bool:
        ...

    def func_ufzz39dk(self) -> None:
        ...

    def func_tci8hya2(self) -> None:
        ...

    def func_ccpxb3xq(self, chunk: bytes) -> None:
        ...


if __name__ == '__main__':
    AsyncHTTPClient.configure(SimpleAsyncHTTPClient)
    main()
