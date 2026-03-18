```pyi
from __future__ import annotations

import contextlib
import io
import json
import sys
import typing
from concurrent.futures import Future
from types import GeneratorType
from urllib.parse import unquote, urljoin

import anyio
import anyio.abc
import anyio.from_thread
from anyio.streams.stapled import StapledObjectStream
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import httpx

_PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = typing.Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = typing.Callable[[Scope], ASGIInstance]
ASGI3App = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
_RequestData = typing.Mapping[str, typing.Union[str, typing.Iterable[str], bytes]]

def _is_asgi3(app: typing.Any) -> bool: ...

class _WrapASGI2:
    app: typing.Any
    def __init__(self, app: typing.Any) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: dict[str, typing.Any]

class _Upgrade(Exception):
    session: WebSocketTestSession
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect):
    """
    A special case of `WebSocketDisconnect`, raised in the `TestClient` if the
    `WebSocket` is closed before being accepted with a `send_denial_response()`.
    """

class WebSocketTestSession:
    app: ASGIApp
    scope: Scope
    accepted_subprotocol: str | None
    portal_factory: _PortalFactoryType
    extra_headers: typing.Any
    portal: anyio.abc.BlockingPortal
    exit_stack: contextlib.ExitStack
    _receive_tx: typing.Any
    _send_rx: typing.Any

    def __init__(self, app: ASGIApp, scope: Scope, portal_factory: _PortalFactoryType) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: typing.Any) -> None: ...
    async def _run(self, *, task_status: typing.Any) -> None: ...
    def _raise_on_close(self, message: Message) -> None: ...
    def send(self, message: Message) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: typing.Any, mode: str = ...) -> None: ...
    def close(self, code: int = ..., reason: str | None = ...) -> None: ...
    def receive(self) -> Message: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> typing.Any: ...

class _TestClientTransport(httpx.BaseTransport):
    app: ASGIApp
    raise_server_exceptions: bool
    root_path: str
    portal_factory: _PortalFactoryType
    app_state: dict[str, typing.Any]
    client: tuple[str, int]

    def __init__(
        self,
        app: ASGIApp,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        *,
        client: tuple[str, int],
        app_state: dict[str, typing.Any],
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

class TestClient(httpx.Client):
    __test__: bool
    portal: anyio.abc.BlockingPortal | None
    async_backend: _AsyncBackend
    app: ASGIApp
    app_state: dict[str, typing.Any]
    stream_send: StapledObjectStream
    stream_receive: StapledObjectStream
    task: Future[typing.Any]
    exit_stack: contextlib.ExitStack

    def __init__(
        self,
        app: ASGIApp,
        base_url: str = ...,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        backend: str = ...,
        backend_options: dict[str, typing.Any] | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        headers: typing.Mapping[str, str] | None = ...,
        follow_redirects: bool = ...,
        client: tuple[str, int] = ...,
    ) -> None: ...
    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Iterator[anyio.abc.BlockingPortal]: ...
    def request(
        self,
        method: str,
        url: httpx.URLTypes,
        *,
        content: httpx.RequestContent | None = ...,
        data: httpx.RequestData | None = ...,
        files: httpx.RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def get(
        self,
        url: httpx.URLTypes,
        *,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def options(
        self,
        url: httpx.URLTypes,
        *,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def head(
        self,
        url: httpx.URLTypes,
        *,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def post(
        self,
        url: httpx.URLTypes,
        *,
        content: httpx.RequestContent | None = ...,
        data: httpx.RequestData | None = ...,
        files: httpx.RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def put(
        self,
        url: httpx.URLTypes,
        *,
        content: httpx.RequestContent | None = ...,
        data: httpx.RequestData | None = ...,
        files: httpx.RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def patch(
        self,
        url: httpx.URLTypes,
        *,
        content: httpx.RequestContent | None = ...,
        data: httpx.RequestData | None = ...,
        files: httpx.RequestFiles | None = ...,
        json: typing.Any | None = ...,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def delete(
        self,
        url: httpx.URLTypes,
        *,
        params: httpx.QueryParamTypes | None = ...,
        headers: httpx.HeaderTypes | None = ...,
        cookies: httpx.CookieTypes | None = ...,
        auth: httpx.AuthTypes | httpx._client._UseClientDefault = ...,
        follow_redirects: bool | httpx._client._UseClientDefault = ...,
        timeout: httpx.TimeoutTypes | httpx._client._UseClientDefault = ...,
        extensions: dict[str, typing.Any] | None = ...,
    ) -> httpx.Response: ...
    def websocket_connect(
        self,
        url: httpx.URLTypes,
        subprotocols: list[str] | None = ...,
        **kwargs: typing.Any,
    ) -> WebSocketTestSession: ...
    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: typing.Any) -> None: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...
```