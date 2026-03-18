from __future__ import annotations
import enum
import logging
from types import TracebackType
from typing import Any, AsyncIterator, AsyncContextManager, Callable, ContextManager, Dict, Iterator, List, Optional, Type, TypeVar, Union
from ._auth import Auth
from ._config import Timeout
from ._models import Cookies, Headers, Request, Response
from ._types import AsyncByteStream, SyncByteStream
from ._urls import URL, QueryParams

__all__: List[str] = ...
T = TypeVar("T", bound="Client")
U = TypeVar("U", bound="AsyncClient")

def EventHook(*args: Any, **kwargs: Any) -> Any: ...

logger: logging.Logger = ...
USER_AGENT: str = ...
ACCEPT_ENCODING: str = ...

class UseClientDefault:
    ...

USE_CLIENT_DEFAULT: UseClientDefault = ...

class ClientState(enum.Enum):
    UNOPENED: ClientState
    OPENED: ClientState
    CLOSED: ClientState
    ...

class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...

class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None: ...
    async def __aiter__(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...

class BaseClient:
    follow_redirects: bool
    max_redirects: int

    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        timeout: Any = ...,
        follow_redirects: bool = ...,
        max_redirects: int = ...,
        event_hooks: Any = ...,
        base_url: Any = ...,
        trust_env: bool = ...,
        default_encoding: Any = ...
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: Any) -> None: ...
    @property
    def event_hooks(self) -> Dict[str, List[Callable[..., Any]]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: Dict[str, List[Callable[..., Any]]]) -> None: ...
    @property
    def auth(self) -> Optional[Auth]: ...
    @auth.setter
    def auth(self, auth: Any) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: Union[str, URL]) -> None: ...
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
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Request: ...

class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        verify: Any = ...,
        cert: Any = ...,
        trust_env: bool = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: Any = ...,
        mounts: Any = ...,
        timeout: Any = ...,
        follow_redirects: bool = ...,
        limits: Any = ...,
        max_redirects: int = ...,
        event_hooks: Any = ...,
        base_url: Any = ...,
        transport: Any = ...,
        default_encoding: Any = ...
    ) -> None: ...
    def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def stream(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> ContextManager[Response]: ...
    def send(self, request: Request, *, stream: bool = ..., auth: Any = ..., follow_redirects: Any = ...) -> Response: ...
    def get(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def options(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def head(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def post(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def put(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def patch(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def delete(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    def close(self) -> None: ...
    def __enter__(self) -> Client: ...
    def __exit__(self, exc_type: Optional[Type[BaseException]] = ..., exc_value: Optional[BaseException] = ..., traceback: Optional[TracebackType] = ...) -> None: ...

class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        verify: Any = ...,
        cert: Any = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: Any = ...,
        mounts: Any = ...,
        timeout: Any = ...,
        follow_redirects: bool = ...,
        limits: Any = ...,
        max_redirects: int = ...,
        event_hooks: Any = ...,
        base_url: Any = ...,
        transport: Any = ...,
        trust_env: bool = ...,
        default_encoding: Any = ...
    ) -> None: ...
    async def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def stream(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> AsyncContextManager[Response]: ...
    async def send(self, request: Request, *, stream: bool = ..., auth: Any = ..., follow_redirects: Any = ...) -> Response: ...
    async def get(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def options(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def head(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def post(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def put(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def patch(
        self,
        url: Union[str, URL],
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def delete(
        self,
        url: Union[str, URL],
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[Dict[str, Any]] = ...
    ) -> Response: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> AsyncClient: ...
    async def __aexit__(self, exc_type: Optional[Type[BaseException]] = ..., exc_value: Optional[BaseException] = ..., traceback: Optional[TracebackType] = ...) -> None: ...