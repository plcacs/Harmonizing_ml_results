from __future__ import annotations
import datetime
import enum
import time
from contextlib import contextmanager, asynccontextmanager
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import Limits, Proxy, Timeout
from ._models import Cookies, Headers, Request, Response
from ._transports.base import AsyncBaseTransport, BaseTransport
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
    RequestFiles,
    SyncByteStream,
)
from ._urls import URL, QueryParams
from ._utils import URLPattern

if TYPE_CHECKING:
    import ssl

__all__: List[str] = ['USE_CLIENT_DEFAULT', 'AsyncClient', 'Client']

T = TypeVar('T', bound='Client')
U = TypeVar('U', bound='AsyncClient')


class UseClientDefault:
    def __init__(self) -> None:
        ...


USE_CLIENT_DEFAULT: UseClientDefault = ...


class ClientState(enum.Enum):
    UNOPENED: str
    OPENED: str
    CLOSED: str


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
    _base_url: URL
    _auth: Optional[Auth]
    _params: QueryParams
    headers: Headers
    _cookies: Cookies
    _timeout: Timeout
    follow_redirects: bool
    max_redirects: int
    _event_hooks: Dict[str, List[Callable]]
    _trust_env: bool
    _default_encoding: str
    _state: ClientState

    def __init__(
        self,
        *,
        auth: AuthTypes = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Dict[str, List[Callable]] = None,
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
    def timeout(self, timeout: TimeoutTypes) -> None:
        ...

    @property
    def event_hooks(self) -> Dict[str, List[Callable]]:
        ...

    @event_hooks.setter
    def event_hooks(self, event_hooks: Dict[str, List[Callable]]) -> None:
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
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Request:
        ...

    def _merge_url(self, url: str) -> URL:
        ...

    def _merge_cookies(self, cookies: CookieTypes = None) -> Cookies:
        ...

    def _merge_headers(self, headers: HeaderTypes = None) -> Headers:
        ...

    def _merge_queryparams(self, params: QueryParamTypes = None) -> QueryParams:
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
    _transport: BaseTransport
    _mounts: Dict[URLPattern, Optional[BaseTransport]]

    def __init__(
        self,
        *,
        auth: AuthTypes = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes = None,
        mounts: Dict[str, BaseTransport] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Dict[str, List[Callable]] = None,
        base_url: str = '',
        transport: BaseTransport = None,
        default_encoding: str = 'utf-8',
    ) -> None:
        ...

    def _init_transport(
        self,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: BaseTransport = None,
    ) -> BaseTransport:
        ...

    def _init_proxy_transport(
        self,
        proxy: ProxyTypes,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
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
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Generator[Response, None, None]:
        ...

    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response:
        ...

    def _send_handling_auth(
        self,
        request: Request,
        auth: Optional[Auth],
        follow_redirects: bool,
        history: List[Response],
    ) -> Response:
        ...

    def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: List[Response],
    ) -> Response:
        ...

    def _send_single_request(self, request: Request) -> Response:
        ...

    def get(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def options(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def head(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def post(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def put(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def patch(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def delete(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> Client:
        ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        ...


class AsyncClient(BaseClient):
    _transport: AsyncBaseTransport
    _mounts: Dict[URLPattern, Optional[AsyncBaseTransport]]

    def __init__(
        self,
        *,
        auth: AuthTypes = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes = None,
        mounts: Dict[str, AsyncBaseTransport] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Dict[str, List[Callable]] = None,
        base_url: str = '',
        transport: AsyncBaseTransport = None,
        trust_env: bool = True,
        default_encoding: str = 'utf-8',
    ) -> None:
        ...

    def _init_transport(
        self,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: AsyncBaseTransport = None,
    ) -> AsyncBaseTransport:
        ...

    def _init_proxy_transport(
        self,
        proxy: ProxyTypes,
        verify: Union[bool, ssl.SSLContext] = True,
        cert: CertTypes = None,
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
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> AsyncGenerator[Response, None]:
        ...

    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response:
        ...

    async def _send_handling_auth(
        self,
        request: Request,
        auth: Optional[Auth],
        follow_redirects: bool,
        history: List[Response],
    ) -> Response:
        ...

    async def _send_handling_redirects(
        self,
        request: Request,
        follow_redirects: bool,
        history: List[Response],
    ) -> Response:
        ...

    async def _send_single_request(self, request: Request) -> Response:
        ...

    async def get(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def options(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def head(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def post(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def put(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def patch(
        self,
        url: str,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def delete(
        self,
        url: str,
        *,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[Timeout, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Dict[str, Any] = None,
    ) -> Response:
        ...

    async def aclose(self) -> None:
        ...

    async def __aenter__(self) -> AsyncClient:
        ...

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        ...