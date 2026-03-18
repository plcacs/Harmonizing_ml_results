```python
from __future__ import annotations
import datetime
import enum
import logging
import time
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG, Limits, Proxy, Timeout
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import InvalidURL, RemoteProtocolError, TooManyRedirects, request_context
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._types import AsyncByteStream, AuthTypes, CertTypes, CookieTypes, HeaderTypes, ProxyTypes, QueryParamTypes, RequestContent, RequestData, RequestExtensions, RequestFiles, SyncByteStream, TimeoutTypes
from ._urls import URL, QueryParams
from ._utils import URLPattern, get_environment_proxies

if typing.TYPE_CHECKING:
    import ssl

__all__: typing.List[str] = ...

T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")

def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int: ...
def _same_origin(url: URL, other: URL) -> bool: ...

class UseClientDefault: ...
USE_CLIENT_DEFAULT: UseClientDefault = ...
logger: logging.Logger = ...
USER_AGENT: str = ...
ACCEPT_ENCODING: str = ...

class ClientState(enum.Enum):
    UNOPENED: typing.ClassVar[int] = ...
    OPENED: typing.ClassVar[int] = ...
    CLOSED: typing.ClassVar[int] = ...

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
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing.List[EventHook]]] = None,
        base_url: typing.Union[str, URL] = "",
        trust_env: bool = True,
        default_encoding: typing.Union[str, typing.Callable[[bytes], str]] = "utf-8"
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    def _enforce_trailing_slash(self, url: URL) -> URL: ...
    def _get_proxy_map(self, proxy: typing.Optional[ProxyTypes], allow_env_proxies: bool) -> typing.Dict[str, typing.Optional[Proxy]]: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...
    @property
    def event_hooks(self) -> typing.Dict[str, typing.List[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: typing.Dict[str, typing.List[EventHook]]) -> None: ...
    @property
    def auth(self) -> typing.Optional[Auth]: ...
    @auth.setter
    def auth(self, auth: typing.Optional[AuthTypes]) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: typing.Union[str, URL]) -> None: ...
    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: typing.Optional[HeaderTypes]) -> None: ...
    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: typing.Optional[CookieTypes]) -> None: ...
    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: typing.Optional[QueryParamTypes]) -> None: ...
    def build_request(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Request: ...
    def _merge_url(self, url: typing.Union[str, URL]) -> URL: ...
    def _merge_cookies(self, cookies: typing.Optional[CookieTypes] = None) -> typing.Optional[Cookies]: ...
    def _merge_headers(self, headers: typing.Optional[HeaderTypes] = None) -> Headers: ...
    def _merge_queryparams(self, params: typing.Optional[QueryParamTypes] = None) -> typing.Optional[QueryParams]: ...
    def _build_auth(self, auth: typing.Optional[AuthTypes]) -> typing.Optional[Auth]: ...
    def _build_request_auth(self, request: Request, auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT) -> Auth: ...
    def _build_redirect_request(self, request: Request, response: Response) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers: ...
    def _redirect_stream(self, request: Request, method: str) -> typing.Optional[SyncByteStream]: ...
    def _set_timeout(self, request: Request) -> None: ...

class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        verify: typing.Union[bool, ssl.SSLContext] = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: typing.Optional[ProxyTypes] = None,
        mounts: typing.Optional[typing.Dict[typing.Union[str, URLPattern], typing.Optional[BaseTransport]]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing.List[EventHook]]] = None,
        base_url: typing.Union[str, URL] = "",
        transport: typing.Optional[BaseTransport] = None,
        default_encoding: typing.Union[str, typing.Callable[[bytes], str]] = "utf-8"
    ) -> None: ...
    def _init_transport(
        self,
        verify: typing.Union[bool, ssl.SSLContext] = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: typing.Optional[BaseTransport] = None
    ) -> BaseTransport: ...
    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: typing.Union[bool, ssl.SSLContext] = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS
    ) -> BaseTransport: ...
    def _transport_for_url(self, url: URL) -> BaseTransport: ...
    def request(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    @contextmanager
    def stream(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> typing.Iterator[Response]: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT
    ) -> Response: ...
    def _send_handling_auth(self, request: Request, auth: Auth, follow_redirects: bool, history: typing.List[Response]) -> Response: ...
    def _send_handling_redirects(self, request: Request, follow_redirects: bool, history: typing.List[Response]) -> Response: ...
    def _send_single_request(self, request: Request) -> Response: ...
    def get(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def options(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def head(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def post(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def put(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def patch(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Any = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def delete(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self: T) -> T: ...
    def __exit__(self, exc_type: typing.Optional[typing.Type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None: ...

class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        verify: typing.Union[bool, ssl.SSLContext] = True,
        cert: typing.Optional[CertTypes] = None,
        http1: bool = True,
        http2: bool = False,
        proxy: typing.Optional[ProxyTypes] = None,
        mounts: typing.Optional[typing.Dict[typing.Union[str, URLPattern], typing.Optional[AsyncBaseTransport]]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing