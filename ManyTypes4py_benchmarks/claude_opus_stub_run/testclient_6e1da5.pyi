from __future__ import annotations

import contextlib
import typing
from concurrent.futures import Future

import anyio
import anyio.abc
import httpx
from anyio.streams.stapled import StapledObjectStream
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

_PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = typing.Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = typing.Callable[[Scope], ASGIInstance]
ASGI3App = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
_RequestData = typing.Mapping[str, typing.Union[str, typing.Iterable[str], bytes]]

def _is_asgi3(app: typing.Union[ASGI2App, ASGI3App]) -> bool: ...

class _WrapASGI2:
    app: ASGI2App
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: dict[str, typing.Any]

class _Upgrade(Exception):
    session: WebSocketTestSession
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect): ...

class WebSocketTestSession:
    app: ASGI3App
    scope: Scope
    accepted_subprotocol: str | None
    portal_factory: _PortalFactoryType
    extra_headers: list[typing.Any] | None
    portal: anyio.abc.BlockingPortal
    exit_stack: contextlib.ExitStack

    def __init__(
        self,
        app: ASGI3App,
        scope: Scope,
        portal_factory: _PortalFactoryType,
    ) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: typing.Any) -> bool | None: ...
    async def _run(self, *, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None: ...
    def _raise_on_close(self, message: typing.Any) -> None: ...
    def send(self, message: typing.Any) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: typing.Any, mode: str = "text") -> None: ...
    def close(self, code: int = 1000, reason: str | None = None) -> None: ...
    def receive(self) -> typing.Any: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = "text") -> typing.Any: ...

class _TestClientTransport(httpx.BaseTransport):
    app: ASGI3App
    raise_server_exceptions: bool
    root_path: str
    portal_factory: _PortalFactoryType
    app_state: dict[str, typing.Any]
    client: tuple[str, int]

    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = True,
        root_path: str = "",
        *,
        client: tuple[str, int],
        app_state: dict[str, typing.Any],
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

class TestClient(httpx.Client):
    __test__: bool
    portal: anyio.abc.BlockingPortal | None
    app: ASGI3App
    app_state: dict[str, typing.Any]
    async_backend: _AsyncBackend
    stream_send: StapledObjectStream[typing.Any]
    stream_receive: StapledObjectStream[typing.Any]
    task: Future[None]
    exit_stack: contextlib.ExitStack

    def __init__(
        self,
        app: ASGIApp,
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        backend: str = "asyncio",
        backend_options: dict[str, typing.Any] | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = True,
        client: tuple[str, int] = ("testclient", 50000),
    ) -> None: ...
    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Generator[anyio.abc.BlockingPortal, None, None]: ...
    def request(  # type: ignore[override]
        self,
        method: str,
        url: httpx._types.URLTypes,
        *,
        content: httpx._types.RequestContent | None = None,
        data: _RequestData | None = None,
        files: httpx._types.FileTypes | None = None,
        json: typing.Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def get(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def options(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def head(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def post(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        content: httpx._types.RequestContent | None = None,
        data: _RequestData | None = None,
        files: httpx._types.FileTypes | None = None,
        json: typing.Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def put(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        content: httpx._types.RequestContent | None = None,
        data: _RequestData | None = None,
        files: httpx._types.FileTypes | None = None,
        json: typing.Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def patch(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        content: httpx._types.RequestContent | None = None,
        data: _RequestData | None = None,
        files: httpx._types.FileTypes | None = None,
        json: typing.Any = None,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def delete(  # type: ignore[override]
        self,
        url: httpx._types.URLTypes,
        *,
        params: httpx._types.QueryParamTypes | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        cookies: httpx._types.CookieTypes | None = None,
        auth: httpx._types.AuthTypes | httpx._client.UseClientDefault = ...,
        follow_redirects: bool | httpx._client.UseClientDefault = ...,
        timeout: httpx._types.TimeoutTypes | httpx._client.UseClientDefault = ...,
        extensions: httpx._types.RequestExtensions | None = None,
    ) -> httpx.Response: ...
    def websocket_connect(
        self,
        url: str,
        subprotocols: typing.Sequence[str] | None = None,
        **kwargs: typing.Any,
    ) -> WebSocketTestSession: ...
    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: typing.Any) -> None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...