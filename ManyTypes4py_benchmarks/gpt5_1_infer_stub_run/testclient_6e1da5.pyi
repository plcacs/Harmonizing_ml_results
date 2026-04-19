from typing import Any, Awaitable, Callable, ContextManager, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypedDict, Union
from types import TracebackType

import anyio
import anyio.abc
import httpx
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

_PortalFactoryType = Callable[[], ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = Callable[[Receive, Send], Awaitable[None]]
ASGI2App = Callable[[Scope], ASGIInstance]
ASGI3App = Callable[[Scope, Receive, Send], Awaitable[None]]
_RequestData = Mapping[str, Union[str, Iterable[str], bytes]]


def _is_asgi3(app: object) -> bool: ...


class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...


class _AsyncBackend(TypedDict):
    backend: str
    backend_options: Dict[str, Any]


class _Upgrade(Exception):
    session: "WebSocketTestSession"
    def __init__(self, session: "WebSocketTestSession") -> None: ...


class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect): ...


class WebSocketTestSession:
    accepted_subprotocol: Optional[str]
    extra_headers: Optional[List[Tuple[bytes, bytes]]]

    def __init__(
        self,
        app: ASGI3App,
        scope: Scope,
        portal_factory: _PortalFactoryType,
    ) -> None: ...
    def __enter__(self) -> "WebSocketTestSession": ...
    def __exit__(
        self,
        __typ: Optional[Type[BaseException]],
        __val: Optional[BaseException]],
        __tb: Optional[TracebackType],
    ) -> Optional[bool]: ...
    async def _run(self, *, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None: ...
    def _raise_on_close(self, message: Mapping[str, Any]) -> None: ...
    def send(self, message: Mapping[str, Any]) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = ...) -> None: ...
    def close(self, code: int = ..., reason: Optional[str] = ...) -> None: ...
    def receive(self) -> Dict[str, Any]: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> Any: ...


class _TestClientTransport(httpx.BaseTransport):
    app: ASGI3App
    raise_server_exceptions: bool
    root_path: str
    portal_factory: _PortalFactoryType
    app_state: Dict[str, Any]
    client: Tuple[str, int]

    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        *,
        client: Tuple[str, int],
        app_state: Dict[str, Any],
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...


class TestClient(httpx.Client):
    __test__: bool = ...
    portal: Optional[anyio.abc.BlockingPortal]

    def __init__(
        self,
        app: Union[ASGI3App, ASGI2App, Callable[..., Any]],
        base_url: str = ...,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        backend: str = ...,
        backend_options: Optional[Dict[str, Any]] = ...,
        cookies: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        follow_redirects: bool = ...,
        client: Tuple[str, int] = ...,
    ) -> None: ...
    def _portal_factory(self) -> ContextManager[anyio.abc.BlockingPortal]: ...
    def request(
        self,
        method: str,
        url: str,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def get(
        self,
        url: str,
        *,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def options(
        self,
        url: str,
        *,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def head(
        self,
        url: str,
        *,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def post(
        self,
        url: str,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def put(
        self,
        url: str,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def patch(
        self,
        url: str,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def delete(
        self,
        url: str,
        *,
        params: Any = ...,
        headers: Optional[Mapping[str, str]] = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Optional[MutableMapping[str, Any]] = ...,
    ) -> httpx.Response: ...
    def websocket_connect(
        self,
        url: str,
        subprotocols: Optional[Iterable[str]] = ...,
        **kwargs: Any,
    ) -> WebSocketTestSession: ...
    def __enter__(self) -> "TestClient": ...
    def __exit__(
        self,
        __typ: Optional[Type[BaseException]],
        __val: Optional[BaseException]],
        __tb: Optional[TracebackType],
    ) -> None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...