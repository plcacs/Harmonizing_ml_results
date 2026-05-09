from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from enum import Enum
from contextlib import AsyncContextManager, ContextManager
from datetime import timedelta
from time import struct_time
from types import TracebackType
from urllib.parse import SplitResult

if TYPE_CHECKING:
    import ssl
    from types import TracebackType

from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import (
    DEFAULT_LIMITS,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_CONFIG,
    Limits,
    Proxy,
    Timeout,
)
from ._models import Cookies, Headers, Request, Response
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
from ._urls import URL, QueryParams, URLPattern

__all__ = ['USE_CLIENT_DEFAULT', 'AsyncClient', 'Client']

T = TypeVar('T', bound='Client')
U = TypeVar('U', bound='AsyncClient')

class UseClientDefault:
    ...

USE_CLIENT_DEFAULT = UseClientDefault()

class ClientState(Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3

class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None:
        ...

    def __iter__(self) -> Iterator[bytes]:
        ...

    def close(self) -> None:
        ...

class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None:
        ...

    async def __aiter__(self) -> AsyncIterator[bytes]:
        ...

    async def aclose(self) -> None:
        ...

EventHook = Callable[..., Any]

class BaseClient:
    def __init__(
        self,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: str = '',
        trust_env: bool = True,
        default_encoding: str = 'utf-8',
    ) -> None:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def trust_env(self) -> bool:
        ...

    def _enforce_trailing_slash(self, url: URL) -> URL:
        ...

    def _get_proxy_map(
        self, proxy: Optional[ProxyTypes], allow_env_proxies: bool
    ) -> Dict[str, Optional[Proxy]]:
        ...

    @property
    def timeout(self) -> Timeout:
        ...

    @timeout.setter
    def timeout(self, timeout: Union[TimeoutTypes, UseClientDefault]) -> None:
        ...

    @property
    def event_hooks(self) -> Dict[str, List[EventHook]]:
        ...

    @event_hooks.setter
    def event_hooks(self, event_hooks: Dict[str, List[EventHook]]) -> None:
        ...

    @property
    def auth(self) -> Optional[Auth]:
        ...

    @auth.setter
    def auth(self, auth: AuthTypes) -> None:
        ...

    @property
    def base_url(self) -> URL:
        ...

    @base_url.setter
    def base_url(self, url: str) -> None:
        ...

    @property
    def headers(self) -> Headers:
        ...

    @headers.setter
    def headers(self, headers: HeaderTypes) -> None:
        ...

    @property
    def cookies(self) -> Cookies:
        ...

    @cookies.setter
    def cookies(self, cookies: CookieTypes) -> None:
        ...

    @property
    def params(self) -> QueryParams:
        ...

    @params.setter
    def params(self, params: QueryParamTypes) -> None:
        ...

    def build_request(
        self,
        method: str,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Request:
        ...

    def _merge_url(self, url: str) -> URL:
        ...

    def _merge_cookies(self, cookies: Optional[CookieTypes]) -> Optional[Cookies]:
        ...

    def _merge_headers(self, headers: Optional[HeaderTypes]) -> Headers:
        ...

    def _merge_queryparams(self, params: Optional[QueryParamTypes]) -> Optional[QueryParams]:
        ...

    def _build_auth(self, auth: AuthTypes) -> Optional[Auth]:
        ...

    def _build_request_auth(
        self, request: Request, auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT
    ) -> Optional[Auth]:
        ...

    def _build_redirect_request(self, request: Request, response: Response) -> Request:
        ...

    def _redirect_method(self, request: Request, response: Response) -> str:
        ...

    def _redirect_url(self, request: Request, response: Response) -> URL:
        ...

    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers:
        ...

    def _redirect_stream(self, request: Request, method: str) -> Optional[SyncByteStream]:
        ...

    def _set_timeout(self, request: Request) -> None:
        ...

class Client(BaseClient):
    def __init__(
        self,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: Optional[ProxyTypes] = None,
        mounts: Optional[Dict[str, BaseTransport]] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: str = '',
        transport: Optional[BaseTransport] = None,
        default_encoding: str = 'utf-8',
    ) -> None:
        ...

    def _init_transport(
        self,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: Optional[BaseTransport] = None,
    ) -> BaseTransport:
        ...

    def _init_proxy_transport(
        self,
        proxy: ProxyTypes,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> BaseTransport:
        ...

    def _transport_for_url(self, url: URL) -> BaseTransport:
        ...

    def request(
        self,
        method: str,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> ContextManager[Response]:
        ...

    def send(
        self,
        request: Request,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> 'Client':
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        ...

    def get(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def options(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def head(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def post(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def put(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def patch(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    def delete(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

class AsyncClient(BaseClient):
    def __init__(
        self,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        http1: bool = True,
        http2: bool = False,
        proxy: Optional[ProxyTypes] = None,
        mounts: Optional[Dict[str, AsyncBaseTransport]] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: str = '',
        transport: Optional[AsyncBaseTransport] = None,
        trust_env: bool = True,
        default_encoding: str = 'utf-8',
    ) -> None:
        ...

    def _init_transport(
        self,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: Optional[AsyncBaseTransport] = None,
    ) -> AsyncBaseTransport:
        ...

    def _init_proxy_transport(
        self,
        proxy: ProxyTypes,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
    ) -> AsyncBaseTransport:
        ...

    def _transport_for_url(self, url: URL) -> AsyncBaseTransport:
        ...

    async def request(
        self,
        method: str,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> AsyncContextManager[Response]:
        ...

    async def send(
        self,
        request: Request,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response:
        ...

    async def aclose(self) -> None:
        ...

    async def __aenter__(self) -> 'AsyncClient':
        ...

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        ...

    async def get(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def options(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def head(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def post(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def put(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def patch(
        self,
        url: str,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...

    async def delete(
        self,
        url: str,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response:
        ...