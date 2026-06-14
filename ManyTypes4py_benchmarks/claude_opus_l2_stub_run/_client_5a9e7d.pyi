from __future__ import annotations

import datetime
import enum
import logging
import typing
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType

from ._auth import Auth
from ._config import Limits, Proxy, Timeout
from ._models import Cookies, Headers, Request, Response
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._types import (
    AsyncByteStream,
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    SyncByteStream,
    TimeoutTypes,
)
from ._urls import URL, QueryParams
from ._utils import URLPattern

if typing.TYPE_CHECKING:
    import ssl

__all__ = ["USE_CLIENT_DEFAULT", "AsyncClient", "Client"]

T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")

EventHook = typing.Callable[..., typing.Any]

logger: logging.Logger
USER_AGENT: str
ACCEPT_ENCODING: str

def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int | None: ...
def _same_origin(url: URL, other: URL) -> bool: ...

class UseClientDefault: ...

USE_CLIENT_DEFAULT: UseClientDefault

class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3

class BoundSyncStream(SyncByteStream):
    def __init__(
        self, stream: SyncByteStream, response: Response, start: float
    ) -> None: ...
    def __iter__(self) -> typing.Iterator[bytes]: ...
    def close(self) -> None: ...

class BoundAsyncStream(AsyncByteStream):
    def __init__(
        self, stream: AsyncByteStream, response: Response, start: float
    ) -> None: ...
    async def __aiter__(self) -> typing.AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...

class BaseClient:
    max_redirects: int
    follow_redirects: bool

    def __init__(
        self,
        *,
        auth: AuthTypes | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        max_redirects: int = ...,
        event_hooks: typing.Mapping[str, list[typing.Callable[..., typing.Any]]] | None = ...,
        base_url: URL | str = ...,
        trust_env: bool = ...,
        default_encoding: str | typing.Callable[[bytes], str] = ...,
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    def _enforce_trailing_slash(self, url: URL) -> URL: ...
    def _get_proxy_map(
        self, proxy: ProxyTypes | None, allow_env_proxies: bool
    ) -> dict[str, Proxy | None]: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...
    @property
    def event_hooks(
        self,
    ) -> dict[str, list[typing.Callable[..., typing.Any]]]: ...
    @event_hooks.setter
    def event_hooks(
        self,
        event_hooks: typing.Mapping[str, list[typing.Callable[..., typing.Any]]],
    ) -> None: ...
    @property
    def auth(self) -> Auth | None: ...
    @auth.setter
    def auth(self, auth: AuthTypes | None) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: URL | str) -> None: ...
    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: HeaderTypes) -> None: ...
    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: CookieTypes) -> None: ...
    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: QueryParamTypes) -> None: ...
    def build_request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Request: ...
    def _merge_url(self, url: URL | str) -> URL: ...
    def _merge_cookies(self, cookies: CookieTypes | None = ...) -> Cookies | None: ...
    def _merge_headers(self, headers: HeaderTypes | None = ...) -> Headers: ...
    def _merge_queryparams(
        self, params: QueryParamTypes | None = ...
    ) -> QueryParams | None: ...
    def _build_auth(self, auth: AuthTypes | None) -> Auth | None: ...
    def _build_request_auth(
        self, request: Request, auth: AuthTypes | UseClientDefault = ...
    ) -> Auth: ...
    def _build_redirect_request(
        self, request: Request, response: Response
    ) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(
        self, request: Request, url: URL, method: str
    ) -> Headers: ...
    def _redirect_stream(
        self, request: Request, method: str
    ) -> SyncByteStream | AsyncByteStream | None: ...
    def _set_timeout(self, request: Request) -> None: ...

class Client(BaseClient):
    _transport: BaseTransport
    _mounts: dict[URLPattern, BaseTransport | None]

    def __init__(
        self,
        *,
        auth: AuthTypes | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: ProxyTypes | None = ...,
        mounts: typing.Mapping[str, BaseTransport | None] | None = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: typing.Mapping[str, list[typing.Callable[..., typing.Any]]] | None = ...,
        base_url: URL | str = ...,
        transport: BaseTransport | None = ...,
        default_encoding: str | typing.Callable[[bytes], str] = ...,
    ) -> None: ...
    def _init_transport(
        self,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
        transport: BaseTransport | None = ...,
    ) -> BaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
    ) -> BaseTransport: ...
    def _transport_for_url(self, url: URL) -> BaseTransport: ...
    def request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    @contextmanager
    def stream(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> typing.Generator[Response, None, None]: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
    ) -> Response: ...
    def _send_handling_auth(
        self,
        request: Request,
        auth: Auth,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response: ...
    def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response: ...
    def _send_single_request(self, request: Request) -> Response: ...
    def get(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def options(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def head(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def post(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def put(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def patch(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def delete(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self: T) -> T: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> None: ...

class AsyncClient(BaseClient):
    _transport: AsyncBaseTransport
    _mounts: dict[URLPattern, AsyncBaseTransport | None]

    def __init__(
        self,
        *,
        auth: AuthTypes | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: ProxyTypes | None = ...,
        mounts: typing.Mapping[str, AsyncBaseTransport | None] | None = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: typing.Mapping[str, list[typing.Callable[..., typing.Any]]] | None = ...,
        base_url: URL | str = ...,
        transport: AsyncBaseTransport | None = ...,
        trust_env: bool = ...,
        default_encoding: str | typing.Callable[[bytes], str] = ...,
    ) -> None: ...
    def _init_transport(
        self,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
        transport: AsyncBaseTransport | None = ...,
    ) -> AsyncBaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | str | bool = ...,
        cert: CertTypes | None = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
    ) -> AsyncBaseTransport: ...
    def _transport_for_url(self, url: URL) -> AsyncBaseTransport: ...
    async def request(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> typing.AsyncGenerator[Response, None]: ...
    async def send(
        self,
        request: Request,
        *,
        stream: bool = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
    ) -> Response: ...
    async def _send_handling_auth(
        self,
        request: Request,
        auth: Auth,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response: ...
    async def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: list[Response],
    ) -> Response: ...
    async def _send_single_request(self, request: Request) -> Response: ...
    async def get(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def options(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def head(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def post(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def put(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def patch(
        self,
        url: URL | str,
        *,
        content: RequestContent | None = ...,
        data: RequestData | None = ...,
        files: RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def delete(
        self,
        url: URL | str,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: AuthTypes | UseClientDefault | None = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = ...,
    ) -> Response: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self: U) -> U: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> None: ...