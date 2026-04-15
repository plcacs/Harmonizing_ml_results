from __future__ import annotations

import contextlib
import io
import sys
import typing
from concurrent.futures import Future
from types import GeneratorType
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import anyio.abc
import httpx
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

_PortalFactoryType = Callable[[], ContextManager[anyio.abc.BlockingPortal]]
ASGIInstance = Callable[[Receive, Send], typing.Awaitable[None]]
ASGI2App = Callable[[Scope], ASGIInstance]
ASGI3App = Callable[[Scope, Receive, Send], typing.Awaitable[None]]
_RequestData = Mapping[str, Union[str, Iterable[str], bytes]]

def _is_asgi3(app: Any) -> TypeGuard[ASGI3App]: ...

class _WrapASGI2:
    def __init__(self, app: ASGI2App) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class _AsyncBackend(typing.TypedDict):
    backend: str
    backend_options: Dict[str, Any]

class _Upgrade(Exception):
    def __init__(self, session: WebSocketTestSession) -> None: ...

class WebSocketDenialResponse(httpx.Response, WebSocketDisconnect): ...

class WebSocketTestSession:
    def __init__(self, app: ASGI3App, scope: Dict[str, Any], portal_factory: _PortalFactoryType) -> None: ...
    def __enter__(self) -> WebSocketTestSession: ...
    def __exit__(self, *args: Any) -> Optional[bool]: ...
    async def _run(self, *, task_status: Any) -> None: ...
    def _raise_on_close(self, message: Dict[str, Any]) -> None: ...
    def send(self, message: Dict[str, Any]) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = "text") -> None: ...
    def close(self, code: int = 1000, reason: Optional[str] = None) -> None: ...
    def receive(self) -> Dict[str, Any]: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = "text") -> Any: ...
    accepted_subprotocol: Optional[str]
    extra_headers: Optional[List[Tuple[bytes, bytes]]]

class _TestClientTransport(httpx.BaseTransport):
    def __init__(
        self,
        app: ASGI3App,
        portal_factory: _PortalFactoryType,
        raise_server_exceptions: bool = True,
        root_path: str = "",
        *,
        client: Tuple[str, int],
        app_state: Dict[str, Any]
    ) -> None: ...
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

class TestClient(httpx.Client):
    __test__: bool = False
    portal: Optional[anyio.abc.BlockingPortal]
    
    def __init__(
        self,
        app: Union[ASGI2App, ASGI3App],
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        backend: str = "asyncio",
        backend_options: Optional[Dict[str, Any]] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        follow_redirects: bool = True,
        client: Tuple[str, int] = ("testclient", 50000)
    ) -> None: ...
    
    @contextlib.contextmanager
    def _portal_factory(self) -> typing.Generator[anyio.abc.BlockingPortal, None, None]: ...
    
    def request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[httpx._types.RequestContent] = None,
        data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def get(
        self,
        url: str,
        *,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def options(
        self,
        url: str,
        *,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def head(
        self,
        url: str,
        *,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def post(
        self,
        url: str,
        *,
        content: Optional[httpx._types.RequestContent] = None,
        data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def put(
        self,
        url: str,
        *,
        content: Optional[httpx._types.RequestContent] = None,
        data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def patch(
        self,
        url: str,
        *,
        content: Optional[httpx._types.RequestContent] = None,
        data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def delete(
        self,
        url: str,
        *,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        auth: Union[httpx._types.AuthTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, object] = httpx._client.USE_CLIENT_DEFAULT,
        timeout: Union[httpx._types.TimeoutTypes, object] = httpx._client.USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None
    ) -> httpx.Response: ...
    
    def websocket_connect(
        self,
        url: str,
        subprotocols: Optional[List[str]] = None,
        **kwargs: Any
    ) -> WebSocketTestSession: ...
    
    def __enter__(self) -> TestClient: ...
    def __exit__(self, *args: Any) -> Optional[bool]: ...
    async def lifespan(self) -> None: ...
    async def wait_startup(self) -> None: ...
    async def wait_shutdown(self) -> None: ...
    
    app: ASGI3App
    app_state: Dict[str, Any]
    async_backend: _AsyncBackend
    stream_send: anyio.abc.ObjectSendStream
    stream_receive: anyio.abc.ObjectReceiveStream
    task: Future[Any]
    exit_stack: contextlib.ExitStack
    
    def _merge_url(self, url: str) -> str: ...