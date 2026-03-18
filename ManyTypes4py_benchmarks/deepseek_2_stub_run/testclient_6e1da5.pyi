```python
from __future__ import annotations
import contextlib
from concurrent.futures import Future
from types import GeneratorType
import typing
from typing import Any, Union, Iterable, Mapping, ContextManager, cast
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import httpx
import anyio
import anyio.abc
import anyio.from_thread
from anyio.streams.stapled import StapledObjectStream

_PortalFactoryType = typing.Callable[[], ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = typing.Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = typing.Callable[[Scope], ASGIInstance]
ASGI3App = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
_RequestData = Mapping[str, Union[str, Iterable[str], bytes]]

def _is_asgi3(app: Any) -> TypeGuard[ASGI3App]: ...

class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: dict[str, Any]

class _Upgrade(Exception):
    session: WebSocketTestSession
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect): ...

class WebSocketTestSession:
    app: ASGI3App
    scope: dict[str, Any]
    accepted_subprotocol: str | None
    portal_factory: _PortalFactoryType
    extra_headers: list[tuple[bytes, bytes]] | None
    portal: anyio.abc.BlockingPortal
    exit_stack: contextlib.ExitStack
    _receive_tx: anyio.abc.ObjectSendStream[dict[str, Any]]
    _send_rx: anyio.abc.ObjectReceiveStream[dict[str, Any]]
    
    def __init__(self, app: ASGI3App, scope: dict[str, Any], portal_factory: _PortalFactoryType) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: Any) -> None: ...
    async def _run(self, *, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None: ...
    def _raise_on_close(self, message: dict[str, Any]) -> None: ...
    def send(self, message: dict[str, Any]) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = "text") -> None: ...
    def close(self, code: int = 1000, reason: str | None = None) -> None: ...
    def receive(self) -> dict[str, Any]: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = "text") -> Any: ...

class _TestClientTransport(httpx.BaseTransport):
    app: ASGI3App
    raise_server_exceptions: bool
    root_path: str
    portal_factory: _PortalFactoryType
    app_state: dict[str, Any]
    client: tuple[str, int]
    
    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = True,
        root_path: str = "",
        *,
        client: tuple[str, int],
        app_state: dict[str, Any]
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

class TestClient(httpx.Client):
    __test__: bool
    portal: anyio.abc.BlockingPortal | None
    async_backend: _AsyncBackend
    app: ASGI3App
    app_state: dict[str, Any]
    stream_send: StapledObjectStream
    stream_receive: StapledObjectStream
    task: Future[Any]
    exit_stack: contextlib.ExitStack
    
    def __init__(
        self,
        app: Union[ASGI2App, ASGI3App],
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        backend: str = "asyncio",
        backend_options: dict[str, Any] | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = True,
        client: tuple[str, int] = ("testclient", 50000)
    ) -> None: ...
    
    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Iterator[anyio.abc.BlockingPortal]: ...
    
    def request(
        self,
        method: str,
        url: str,
        *,
        content: httpx._types.RequestContent | None = None,
        data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        json: Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def get(
        self,
        url: str,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def options(
        self,
        url: str,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def head(
        self,
        url: str,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def post(
        self,
        url: str,
        *,
        content: httpx._types.RequestContent | None = None,
        data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        json: Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def put(
        self,
        url: str,
        *,
        content: httpx._types.RequestContent | None = None,
        data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        json: Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def patch(
        self,
        url: str,
        *,
        content: httpx._types.RequestContent | None = None,
        data: httpx._types.RequestData | None = None,
        files: httpx._types.RequestFiles | None = None,
        json: Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def delete(
        self,
        url: str,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT,
        extensions: dict[str, Any] | None = None
    ) -> httpx.Response: ...
    
    def websocket_connect(
        self,
        url: str,
        subprotocols: list[str] | None = None,
        **kwargs: Any
    ) -> WebSocketTestSession: ...
    
    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: Any) -> None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...
```