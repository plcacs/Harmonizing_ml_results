from __future__ import annotations
import enum
import logging
import ssl
import typing
from types import TracebackType
from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG, Limits, Proxy, Timeout
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import InvalidURL, RemoteProtocolError, TooManyRedirects, request_context
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._types import AsyncByteStream, CookieTypes, HeaderTypes, QueryParamTypes, SyncByteStream
from ._urls import URL, QueryParams
from ._utils import URLPattern

__all__: list[str]
T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")


def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int | None: ...
def _same_origin(url: URL, other: URL) -> bool: ...


class UseClientDefault:
    ...


USE_CLIENT_DEFAULT: UseClientDefault
logger: logging.Logger
USER_AGENT: str
ACCEPT_ENCODING: str


class ClientState(enum.Enum):
    UNOPENED: typing.ClassVar["ClientState"]
    OPENED: typing.ClassVar["ClientState"]
    CLOSED: typing.ClassVar["ClientState"]


class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None: ...
    def __iter__(self) -> typing.Iterator[bytes]: ...
    def close(self) -> None: ...


class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None: ...
    def __aiter__(self) -> typing.AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...


EventHook = typing.Callable[..., typing.Any]


class BaseClient:
    def __init__(
        self,
        *,
        auth: tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        timeout: Timeout | typing.Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: dict[str, list[EventHook]] | None = ...,
        base_url: str | URL = "",
        trust_env: bool = True,
        default_encoding: str | typing.Callable[..., str] = "utf-8",
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    def _enforce_trailing_slash(self, url: URL) -> URL: ...
    def _get_proxy_map(
        self,
        proxy: str | URL | Proxy | typing.Mapping[str, str | URL | None | Proxy] | None,
        allow_env_proxies: bool,
    ) -> dict[str, Proxy | None]: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: Timeout | typing.Any) -> None: ...
    @property
    def event_hooks(self) -> dict[str, list[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: typing.Mapping[str, list[EventHook]]) -> None: ...
    @property
    def auth(self) -> Auth | None: ...
    @auth.setter
    def auth(self, auth: tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None) -> None: ...
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
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Request: ...
    def _merge_url(self, url: str | URL) -> URL: ...
    def _merge_cookies(self, cookies: CookieTypes | None = ...) -> Cookies | None: ...
    def _merge_headers(self, headers: HeaderTypes | None = ...) -> Headers: ...
    def _merge_queryparams(self, params: QueryParamTypes | None = ...) -> QueryParams | None: ...
    def _build_auth(self, auth: tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None) -> Auth | None: ...
    def _build_request_auth(self, request: Request, auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT) -> Auth: ...
    def _build_redirect_request(self, request: Request, response: Response) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers: ...
    def _redirect_stream(self, request: Request, method: str) -> SyncByteStream | AsyncByteStream | None: ...
    def _set_timeout(self, request: Request) -> None: ...


class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: str | URL | Proxy | typing.Mapping[str, str | URL | None | Proxy] | None = ...,
        mounts: typing.Mapping[str, BaseTransport] | None = ...,
        timeout: Timeout | typing.Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: dict[str, list[EventHook]] | None = ...,
        base_url: str | URL = "",
        transport: BaseTransport | None = ...,
        default_encoding: str | typing.Callable[..., str] = "utf-8",
    ) -> None: ...
    def _init_transport(
        self,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: BaseTransport | None = ...,
    ) -> BaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
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
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def stream(
        self,
        method: str,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> typing.Iterator[Response]: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
    ) -> Response: ...
    def _send_handling_auth(self, request: Request, auth: Auth, follow_redirects: bool, history: list[Response]) -> Response: ...
    def _send_handling_redirects(self, request: Request, follow_redirects: bool, history: list[Response]) -> Response: ...
    def _send_single_request(self, request: Request) -> Response: ...
    def get(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def options(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def head(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def post(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def put(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def patch(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def delete(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Client": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> None: ...


class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
        http1: bool = True,
        http2: bool = False,
        proxy: str | URL | Proxy | typing.Mapping[str, str | URL | None | Proxy] | None = ...,
        mounts: typing.Mapping[str, AsyncBaseTransport] | None = ...,
        timeout: Timeout | typing.Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: dict[str, list[EventHook]] | None = ...,
        base_url: str | URL = "",
        transport: AsyncBaseTransport | None = ...,
        trust_env: bool = True,
        default_encoding: str | typing.Callable[..., str] = "utf-8",
    ) -> None: ...
    def _init_transport(
        self,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: AsyncBaseTransport | None = ...,
    ) -> AsyncBaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: bool | ssl.SSLContext = True,
        cert: typing.Any = ...,
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
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def stream(
        self,
        method: str,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> typing.AsyncIterator[Response]: ...
    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
    ) -> Response: ...
    async def _send_handling_auth(self, request: Request, auth: Auth, follow_redirects: bool, history: list[Response]) -> Response: ...
    async def _send_handling_redirects(self, request: Request, follow_redirects: bool, history: list[Response]) -> Response: ...
    async def _send_single_request(self, request: Request) -> Response: ...
    async def get(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def options(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def head(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def post(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def put(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def patch(
        self,
        url: str | URL,
        *,
        content: typing.Any = ...,
        data: typing.Any = ...,
        files: typing.Any = ...,
        json: typing.Any = ...,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def delete(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = ...,
        headers: HeaderTypes | None = ...,
        cookies: CookieTypes | None = ...,
        auth: UseClientDefault | tuple[str, str] | Auth | typing.Callable[..., typing.Any] | None = USE_CLIENT_DEFAULT,
        follow_redirects: UseClientDefault | bool = USE_CLIENT_DEFAULT,
        timeout: UseClientDefault | Timeout | float | None | typing.Mapping[str, typing.Any] = USE_CLIENT_DEFAULT,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> Response: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> "AsyncClient": ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> None: ...