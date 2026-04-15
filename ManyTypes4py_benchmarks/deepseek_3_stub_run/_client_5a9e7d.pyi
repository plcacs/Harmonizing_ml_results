from __future__ import annotations

import datetime
import enum
import logging
import ssl
import time
import typing
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType

from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG, Limits, Proxy, Timeout
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import InvalidURL, RemoteProtocolError, TooManyRedirects
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
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
    from typing_extensions import TypeGuard

__all__: typing.Tuple[str, ...] = ("USE_CLIENT_DEFAULT", "AsyncClient", "Client")

T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")


def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int: ...
def _same_origin(url: URL, other: URL) -> bool: ...


class UseClientDefault:
    """For some parameters such as `auth=...` and `timeout=...` we need to be able
    to indicate the default "unset" state, in a way that is distinctly different
    to using `None`.

    The default "unset" state indicates that whatever default is set on the
    client should be used. This is different to setting `None`, which
    explicitly disables the parameter, possibly overriding a client default.

    For example we use `timeout=USE_CLIENT_DEFAULT` in the `request()` signature.
    Omitting the `timeout` parameter will send a request using whatever default
    timeout has been configured on the client. Including `timeout=None` will
    ensure no timeout is used.

    Note that user code shouldn't need to use the `USE_CLIENT_DEFAULT` constant,
    but it is used internally when a parameter is not included.
    """


USE_CLIENT_DEFAULT: UseClientDefault = ...
logger: logging.Logger = ...
USER_AGENT: str = ...
ACCEPT_ENCODING: str = ...


class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3


class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None: ...
    def __iter__(self) -> typing.Iterator[bytes]: ...
    def close(self) -> None: ...


class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None: ...
    async def __aiter__(self) -> typing.AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...


EventHook = typing.Callable[..., typing.Any]


class BaseClient:
    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Mapping[str, typing.List[EventHook]] | None = None,
        base_url: str | URL = "",
        trust_env: bool = True,
        default_encoding: str = "utf-8",
    ) -> None: ...

    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    def _enforce_trailing_slash(self, url: URL) -> URL: ...
    def _get_proxy_map(
        self, proxy: ProxyTypes | None, allow_env_proxies: bool
    ) -> typing.Dict[str, Proxy | None]: ...

    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...

    @property
    def event_hooks(self) -> typing.Dict[str, typing.List[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: typing.Mapping[str, typing.List[EventHook]]) -> None: ...

    @property
    def auth(self) -> Auth | None: ...
    @auth.setter
    def auth(self, auth: AuthTypes | None) -> None: ...

    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: str | URL) -> None: ...

    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: HeaderTypes | None) -> None: ...

    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: CookieTypes | None) -> None: ...

    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: QueryParamTypes | None) -> None: ...

    def build_request(
        self,
        method: str,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Request: ...

    def _merge_url(self, url: str | URL) -> URL: ...
    def _merge_cookies(self, cookies: CookieTypes | None = None) -> CookieTypes | None: ...
    def _merge_headers(self, headers: HeaderTypes | None = None) -> Headers: ...
    def _merge_queryparams(self, params: QueryParamTypes | None = None) -> QueryParams | None: ...
    def _build_auth(self, auth: AuthTypes | None) -> Auth | None: ...
    def _build_request_auth(
        self, request: Request, auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT
    ) -> Auth: ...
    def _build_redirect_request(self, request: Request, response: Response) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers: ...
    def _redirect_stream(self, request: Request, method: str) -> SyncByteStream | None: ...
    def _set_timeout(self, request: Request) -> None: ...


class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: typing.Mapping[str, BaseTransport] | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Mapping[str, typing.List[EventHook]] | None = None,
        base_url: str | URL = "",
        transport: BaseTransport | None = None,
        default_encoding: str = "utf-8",
    ) -> None: ...

    def _init_transport(
        self,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: BaseTransport | None = None,
    ) -> BaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> BaseTransport: ...
    def _transport_for_url(self, url: URL) -> BaseTransport: ...

    def request(
        self,
        method: str,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    @contextmanager
    def stream(
        self,
        method: str,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Iterator[Response]: ...

    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Response: ...

    def _send_handling_auth(
        self,
        request: Request,
        auth: Auth,
        follow_redirects: bool,
        history: typing.List[Response],
    ) -> Response: ...
    def _send_handling_redirects(
        self, request: Request, follow_redirects: bool, history: typing.List[Response]
    ) -> Response: ...
    def _send_single_request(self, request: Request) -> Response: ...

    def get(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def options(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def head(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def post(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def put(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def patch(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def delete(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...

    def close(self) -> None: ...
    def __enter__(self) -> Client: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: typing.Mapping[str, AsyncBaseTransport] | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Mapping[str, typing.List[EventHook]] | None = None,
        base_url: str | URL = "",
        transport: AsyncBaseTransport | None = None,
        trust_env: bool = True,
        default_encoding: str = "utf-8",
    ) -> None: ...

    def _init_transport(
        self,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: AsyncBaseTransport | None = None,
    ) -> AsyncBaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: ssl.SSLContext | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> AsyncBaseTransport: ...
    def _transport_for_url(self, url: URL) -> AsyncBaseTransport: ...

    async def request(
        self,
        method: str,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any = None,
        params: QueryParamTypes | None = None,
        headers