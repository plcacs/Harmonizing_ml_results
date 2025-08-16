from __future__ import annotations
import typing
from .._config import Limits, Proxy, create_ssl_context
from .._exceptions import ConnectError, ConnectTimeout, LocalProtocolError, NetworkError, PoolTimeout, ProtocolError, ProxyError, ReadError, ReadTimeout, RemoteProtocolError, TimeoutException, UnsupportedProtocol, WriteError, WriteTimeout
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport

class ResponseStream(SyncByteStream):

    def __init__(self, httpcore_stream: typing.Iterable[bytes]):
        self._httpcore_stream = httpcore_stream

class HTTPTransport(BaseTransport):

    def __init__(self, verify: bool = True, cert: CertTypes = None, trust_env: bool = True, http1: bool = True, http2: bool = False, limits: Limits = DEFAULT_LIMITS, proxy: ProxyTypes = None, uds: str = None, local_address: str = None, retries: int = 0, socket_options: typing.Optional[typing.List[SOCKET_OPTION]] = None):
        ...

    def handle_request(self, request: Request) -> Response:
        ...

    def close(self):
        ...

class AsyncResponseStream(AsyncByteStream):

    def __init__(self, httpcore_stream: typing.AsyncIterable[bytes]):
        self._httpcore_stream = httpcore_stream

class AsyncHTTPTransport(AsyncBaseTransport):

    def __init__(self, verify: bool = True, cert: CertTypes = None, trust_env: bool = True, http1: bool = True, http2: bool = False, limits: Limits = DEFAULT_LIMITS, proxy: ProxyTypes = None, uds: str = None, local_address: str = None, retries: int = 0, socket_options: typing.Optional[typing.List[SOCKET_OPTION]] = None):
        ...

    async def handle_async_request(self, request: Request) -> Response:
        ...

    async def aclose(self):
        ...
