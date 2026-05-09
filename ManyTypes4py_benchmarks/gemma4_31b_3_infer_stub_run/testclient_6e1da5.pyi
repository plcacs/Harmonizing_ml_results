from __future__ import annotations
import typing
import io
import httpx
from typing import Any, Callable, Awaitable, Optional, Union, Iterable, Mapping, cast
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
from anyio.streams.stapled import StapledObjectStream
from anyio.abc import BlockingPortal

_PortalFactoryType = Callable[[], typing.ContextManager[BlockingPortal]]
ASGIInstance = Callable[[Receive, Send], Awaitable[None]]
ASGI2App = Callable[[Scope], ASGIInstance]
ASGI3App = Callable[[Scope, Receive, Send], Awaitable[None]]
_RequestData = Mapping[str, Union[str, Iterable[str], bytes]]

def _is_asgi3(app: Any) -> bool: ...

class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: Mapping[str, Any]

class _Upgrade(Exception):
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect):
    """
    A special case of `WebSocketDisconnect`, raised in the `TestClient` if the
    `WebSocket` is closed before being accepted with a `send_denial_response()`.
    """
    ...

class WebSocketTestSession:
    app: ASGI3App
    scope: Scope
    accepted_subprotocol: Optional[str]
    portal_factory: _PortalFactoryType
    extra_headers: Optional[list[tuple[bytes, bytes]]]
    portal: BlockingPortal
    exit_stack: Any
    _receive_tx: Any
    _send_rx: Any

    def __init__(self, app: ASGI3App, scope: Scope, portal_factory: _PortalFactoryType) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: Any) -> Any: ...
    async def _run(self, *, task_status: Any) -> None: ...
    def _raise_on_close(self, message: Message) -> None: ...
    def send(self, message: Message) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = 'text') -> None: ...
    def close(self, code: int = 1000, reason: Optional[str] = None) -> None: ...
    def receive(self) -> Message: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = 'text') -> Any: ...

class _TestClientTransport(httpx.BaseTransport):
    def __init__(
        self, 
        app: ASGI3App, 
        portal_factory: _PortalFactoryType, 
        raise_server_exceptions: bool = True, 
        root_path: str = '', 
        *, 
        client: tuple[str, int], 
        app_state: dict[str, Any]
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

class TestClient(httpx.Client):
    __test__: bool
    portal: Optional[BlockingPortal]
    app: ASGI3App
    app_state: dict[str, Any]
    async_backend: _AsyncBackend
    stream_send: StapledObjectStream
    stream_receive: StapledObjectStream
    task: Any
    exit_stack: Any

    def __init__(
        self, 
        app: Union[ASGI2App, ASGI3App], 
        base_url: str = 'http://testserver', 
        raise_server_exceptions: bool = True, 
        root_path: str = '', 
        backend: str = 'asyncio', 
        backend_options: Optional[Mapping[str, Any]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        follow_redirects: bool = True, 
        client: tuple[str, int] = ('testclient', 50000)
    ) -> None: ...

    @typing.contextmanager
    def _portal_factory(self) -> typing.Iterator[BlockingPortal]: ...

    def request(
        self, 
        method: str, 
        url: str, 
        *, 
        content: Optional[Union[str, bytes, Iterable[bytes]]] = None, 
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None, 
        files: Optional[Mapping[str, Any]] = None, 
        json: Optional[Any] = None, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def get(
        self, 
        url: str, 
        *, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def options(
        self, 
        url: str, 
        *, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def head(
        self, 
        url: str, 
        *, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def post(
        self, 
        url: str, 
        *, 
        content: Optional[Union[str, bytes, Iterable[bytes]]] = None, 
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None, 
        files: Optional[Mapping[str, Any]] = None, 
        json: Optional[Any] = None, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def put(
        self, 
        url: str, 
        *, 
        content: Optional[Union[str, bytes, Iterable[bytes]]] = None, 
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None, 
        files: Optional[Mapping[str, Any]] = None, 
        json: Optional[Any] = None, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def patch(
        self, 
        url: str, 
        *, 
        content: Optional[Union[str, bytes, Iterable[bytes]]] = None, 
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None, 
        files: Optional[Mapping[str, Any]] = None, 
        json: Optional[Any] = None, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def delete(
        self, 
        url: str, 
        *, 
        params: Optional[Mapping[str, Any]] = None, 
        headers: Optional[Mapping[str, str]] = None, 
        cookies: Optional[Mapping[str, str]] = None, 
        auth: Any = httpx._client.USE_CLIENT_DEFAULT, 
        follow_redirects: Any = httpx._client.USE_CLIENT_DEFAULT, 
        timeout: Any = httpx._client.USE_CLIENT_DEFAULT, 
        extensions: Optional[Mapping[str, Any]] = None
    ) -> httpx.Response: ...

    def websocket_connect(self, url: str, subprotocols: Optional[Iterable[str]] = None, **kwargs: Any) -> WebSocketTestSession: ...

    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: Any) -> Any: ...

    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...