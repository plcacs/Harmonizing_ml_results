```python
from __future__ import annotations

import datetime
import enum
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

__all__: list[str]

T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")

def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int: ...
def _same_origin(url: URL, other: URL) -> bool: ...

class UseClientDefault:
    """
    For some parameters such as `auth=...` and `timeout=...` we need to be able
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

USE_CLIENT_DEFAULT: UseClientDefault

class BoundSyncStream(SyncByteStream):
    """
    A byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """

    def __init__(
        self, stream: SyncByteStream, response: Response, start: float
    ) -> None: ...
    def __iter__(self) -> typing.Iterator[bytes]: ...
    def close(self) -> None: ...

class BoundAsyncStream(AsyncByteStream):
    """
    An async byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """

    def __init__(
        self, stream: AsyncByteStream, response: Response, start: float
    ) -> None: ...
    async def __aiter__(self) -> typing.AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...

EventHook = typing.Callable[..., typing.Any]

class ClientState(enum.Enum):
    UNOPENED: int
    OPENED: int
    CLOSED: int

class BaseClient:
    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = False,
        max_redirects: int = ...,
        event_hooks: dict[str, list[EventHook]] | None = None,
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
    ) -> dict[str, Proxy | None]: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...
    @property
    def event_hooks(self) -> dict[str, list[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: dict[str, list[EventHook]]) -> None: ...
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
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Request: ...
    def _merge_url(self, url: str | URL) -> URL: ...
    def _merge_cookies(self, cookies: CookieTypes | None = None) -> Cookies | None: ...
    def _merge_headers(self, headers: HeaderTypes | None = None) -> Headers: ...
    def _merge_queryparams(
        self, params: QueryParamTypes | None = None
    ) -> QueryParams | None: ...
    def _build_auth(self, auth: AuthTypes | None) -> Auth | None: ...
    def _build_request_auth(
        self, request: Request, auth: AuthTypes | UseClientDefault = ...
    ) -> Auth: ...
    def _build_redirect_request(self, request: Request, response: Response) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers: ...
    def _redirect_stream(self, request: Request, method: str) -> SyncByteStream | None: ...
    def _set_timeout(self, request: Request) -> None: ...

class Client(BaseClient):
    """
    An HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.

    It can be shared between threads.

    Usage:

    ```python
    >>> client = httpx.Client()
    >>> response = client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: dict[str, BaseTransport] | None = None,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = False,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: dict[str, list[EventHook]] | None = None,
        base_url: str | URL = "",
        transport: BaseTransport | None = None,
        default_encoding: str = "utf-8",
    ) -> None: ...
    def _init_transport(
        self,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = ...,
        transport: BaseTransport | None = None,
    ) -> BaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = ...,
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
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
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
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> typing.Iterator[Response]: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault = ...,
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
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def options(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def head(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def post(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def put(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def patch(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def delete(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self: T) -> T: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None: ...

class AsyncClient(BaseClient):
    """
    An asynchronous HTTP client, with connection pooling, HTTP/2, redirects,
    cookie persistence, etc.

    It can be shared between tasks.

    Usage:

    ```python
    >>> async with httpx.AsyncClient() as client:
    >>>     response = await client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: dict[str, AsyncBaseTransport] | None = None,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = False,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: dict[str, list[EventHook]] | None = None,
        base_url: str | URL = "",
        transport: AsyncBaseTransport | None = None,
        trust_env: bool = True,
        default_encoding: str = "utf-8",
    ) -> None: ...
    def _init_transport(
        self,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = ...,
        transport: AsyncBaseTransport | None = None,
    ) -> AsyncBaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: CertTypes = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = ...,
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
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> typing.AsyncIterator[Response]: ...
    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault = ...,
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
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def options(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def head(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def post(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def put(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def patch(
        self,
        url: str | URL,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def delete(
        self,
        url: str | URL,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        auth: AuthTypes | UseClientDefault = ...,
        follow_redirects: bool | UseClientDefault = ...,
        timeout: TimeoutTypes | UseClientDefault = ...,
        extensions: RequestExtensions | None = None,
    ) -> Response: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self: U) -> U: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None: ...
```