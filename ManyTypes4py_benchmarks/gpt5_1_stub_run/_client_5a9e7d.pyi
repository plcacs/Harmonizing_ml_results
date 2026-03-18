from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Iterator, AsyncIterator, ContextManager, AsyncContextManager
import enum
import logging
from types import TracebackType
from ._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG
from ._models import Cookies, Headers, Request, Response
from ._types import AsyncByteStream, SyncByteStream
from ._urls import URL, QueryParams

__all__: List[str] = ...
T = TypeVar("T", bound="Client")
U = TypeVar("U", bound="AsyncClient")

EventHook = Callable[..., Any]

class UseClientDefault: ...
USE_CLIENT_DEFAULT: UseClientDefault = ...

logger: logging.Logger = ...
USER_AGENT: str = ...
ACCEPT_ENCODING: str = ...

class ClientState(enum.Enum):
    UNOPENED: "ClientState"
    OPENED: "ClientState"
    CLOSED: "ClientState"

class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...

class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None: ...
    async def __aiter__(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...

class BaseClient:
    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        timeout: Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Any = ...,
        base_url: str = "",
        trust_env: bool = True,
        default_encoding: str = "utf-8",
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    def _enforce_trailing_slash(self, url: URL) -> URL: ...
    def _get_proxy_map(self, proxy: Any, allow_env_proxies: bool) -> Dict[str, Any]: ...
    @property
    def timeout(self) -> Any: ...
    @timeout.setter
    def timeout(self, timeout: Any) -> None: ...
    @property
    def event_hooks(self) -> Dict[str, List[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: Any) -> None: ...
    @property
    def auth(self) -> Any: ...
    @auth.setter
    def auth(self, auth: Any) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: Any) -> None: ...
    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: Any) -> None: ...
    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: Any) -> None: ...
    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: Any) -> None: ...
    def build_request(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Request: ...

class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        verify: Any = True,
        cert: Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: Any = None,
        mounts: Any = None,
        timeout: Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Any = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Any = None,
        base_url: str = "",
        transport: Any = None,
        default_encoding: str = "utf-8",
    ) -> None: ...
    def request(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def stream(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> ContextManager[Response]: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
    ) -> Response: ...
    def get(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def options(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def head(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def post(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def put(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def patch(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def delete(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Client": ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = ...,
        exc_value: Optional[BaseException] = ...,
        traceback: Optional[TracebackType] = ...,
    ) -> None: ...

class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        verify: Any = True,
        cert: Any = None,
        http1: bool = True,
        http2: bool = False,
        proxy: Any = None,
        mounts: Any = None,
        timeout: Any = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Any = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Any = None,
        base_url: str = "",
        transport: Any = None,
        trust_env: bool = True,
        default_encoding: str = "utf-8",
    ) -> None: ...
    async def request(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def stream(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> AsyncContextManager[Response]: ...
    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
    ) -> Response: ...
    async def get(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def options(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def head(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def post(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def put(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def patch(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def delete(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = USE_CLIENT_DEFAULT,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Any = ...,
    ) -> Response: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> "AsyncClient": ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = ...,
        exc_value: Optional[BaseException] = ...,
        traceback: Optional[TracebackType] = ...,
    ) -> None: ...