from typing import Any, Awaitable, Callable, ContextManager, Iterable, Mapping, Optional
import httpx
import anyio.abc
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

ASGIInstance = Callable[[Receive, Send], Awaitable[None]]
ASGI2App = Callable[[Scope], ASGIInstance]
ASGI3App = Callable[[Scope, Receive, Send], Awaitable[None]]

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect):
    ...

class WebSocketTestSession:
    app: Any
    scope: Any
    accepted_subprotocol: Optional[str]
    portal_factory: Any
    extra_headers: Any

    def __init__(self, app: Any, scope: Any, portal_factory: Any) -> None: ...
    def __enter__(self) -> "WebSocketTestSession": ...
    def __exit__(self, *args: Any) -> Any: ...
    def send(self, message: Any) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = ...) -> None: ...
    def close(self, code: int = ..., reason: Optional[str] = ...) -> None: ...
    def receive(self) -> Any: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> Any: ...

class TestClient(httpx.Client):
    __test__: bool = ...
    portal: Any = ...

    def __init__(
        self,
        app: Any,
        base_url: Any = ...,
        raise_server_exceptions: Any = ...,
        root_path: Any = ...,
        backend: Any = ...,
        backend_options: Any = ...,
        cookies: Any = ...,
        headers: Any = ...,
        follow_redirects: Any = ...,
        client: Any = ...,
    ) -> None: ...
    def request(
        self,
        method: Any,
        url: Any,
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
        extensions: Any = ...,
    ) -> httpx.Response: ...
    def get(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
    def options(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
    def head(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
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
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
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
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
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
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
    def delete(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: Any = ...,
        timeout: Any = ...,
        extensions: Any = ...,
    ) -> httpx.Response: ...
    def websocket_connect(self, url: Any, subprotocols: Any = ..., **kwargs: Any) -> WebSocketTestSession: ...
    def __enter__(self) -> "TestClient": ...
    def __exit__(self, *args: Any) -> Any: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...