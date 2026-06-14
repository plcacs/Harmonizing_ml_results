from __future__ import annotations
import contextlib
from concurrent.futures import Future
from types import TracebackType
from typing import Any, Awaitable, Callable, ContextManager, Iterable, Mapping, Optional, TypedDict, Union

import anyio.abc
import httpx
from anyio.streams.stapled import StapledObjectStream
from httpx import BaseTransport, Client, Request, Response, URL
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

_PortalFactoryType = Callable[[], ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = Callable[[Receive, Send], Awaitable[None]]
ASGI2App = Callable[[Scope], ASGIInstance]
ASGI3App = Callable[[Scope, Receive, Send], Awaitable[None]]
_RequestData = Mapping[str, Union[str, Iterable[str], bytes]]

def _is_asgi3(app: Any) -> bool: ...

class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(TypedDict):
    backend: str
    backend_options: dict[str, Any]

class _Upgrade(Exception):
    session: WebSocketTestSession
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect): ...

class WebSocketTestSession:
    app: ASGI3App
    scope: Scope
    portal_factory: _PortalFactoryType
    accepted_subprotocol: Optional[str]
    extra_headers: Optional[list[tuple[bytes, bytes]]]

    def __init__(self, app: ASGI3App, scope: Scope, portal_factory: _PortalFactoryType) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool | None: ...
    async def _run(self, *, task_status: Any) -> None: ...
    def _raise_on_close(self, message: Message) -> None: ...
    def send(self, message: Message) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = ...) -> None: ...
    def close(self, code: int = ..., reason: Optional[str] = ...) -> None: ...
    def receive(self) -> Message: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> Any: ...

class _TestClientTransport(BaseTransport):
    app: ASGI3App
    raise_server_exceptions: bool
    root_path: str
    portal_factory: _PortalFactoryType
    app_state: dict[str, Any]
    client: tuple[str, int] | None

    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        *,
        client: tuple[str, int] | None,
        app_state: dict[str, Any],
    ) -> None: ...
    def handle_request(self, request: Request) -> Response: ...

class TestClient(Client):
    __test__: bool
    portal: anyio.abc.BlockingPortal | None
    stream_send: StapledObjectStream[Any]
    stream_receive: StapledObjectStream[Any]
    task: Future[None]
    exit_stack: contextlib.ExitStack
    async_backend: _AsyncBackend
    app: ASGI3App
    app_state: dict[str, Any]

    def __init__(
        self,
        app: ASGI3App | ASGI2App | type[Any],
        base_url: str = ...,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        backend: str = ...,
        backend_options: dict[str, Any] | None = ...,
        cookies: Any = ...,
        headers: Mapping[str, str] | None = ...,
        follow_redirects: bool = ...,
        client: tuple[str, int] = ...,
    ) -> None: ...
    def _portal_factory(self) -> contextlib.AbstractContextManager[anyio.abc.BlockingPortal]: ...
    def request(
        self,
        method: str,
        url: str | URL,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def get(
        self,
        url: str | URL,
        *,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def options(
        self,
        url: str | URL,
        *,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def head(
        self,
        url: str | URL,
        *,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def post(
        self,
        url: str | URL,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def put(
        self,
        url: str | URL,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def patch(
        self,
        url: str | URL,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def delete(
        self,
        url: str | URL,
        *,
        params: Any = ...,
        headers: Mapping[str, str] | None = ...,
        cookies: Any = ...,
        auth: Any = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = ...,
    ) -> Response: ...
    def websocket_connect(self, url: str, subprotocols: Iterable[str] | None = ..., **kwargs: Any) -> WebSocketTestSession: ...
    def __enter__(self) -> TestClient: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool | None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...