from __future__ import annotations
from contextlib import ContextManager
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from urllib.parse import SplitResult
from uuid import UUID

import anyio
import anyio.abc
import httpx
import starlette.types
from starlette.types import ASGIApp, Message, Receive, Scope, Send

class _WrapASGI2:
    def __init__(self, app: ASGIApp) -> None:
        ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        ...

class WebSocketTestSession:
    def __init__(self, app: ASGIApp, scope: Scope, portal_factory: Callable[[], ContextManager[anyio.abc.BlockingPortal]]) -> None:
        ...

    def __enter__(self) -> WebSocketTestSession:
        ...

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        ...

    async def _run(self, *, task_status: anyio.abc.TaskStatus) -> None:
        ...

    def _raise_on_close(self, message: Message) -> None:
        ...

    def send(self, message: Message) -> None:
        ...

    def send_text(self, data: str) -> None:
        ...

    def send_bytes(self, data: bytes) -> None:
        ...

    def send_json(self, data: Any, mode: str) -> None:
        ...

    def close(self, code: int, reason: Optional[str]) -> None:
        ...

    def receive(self) -> Message:
        ...

    def receive_text(self) -> str:
        ...

    def receive_bytes(self) -> bytes:
        ...

    def receive_json(self, mode: str) -> Any:
        ...

class _TestClientTransport(httpx.BaseTransport):
    def __init__(self, app: ASGIApp, portal_factory: Callable[[], ContextManager[anyio.abc.BlockingPortal]], raise_server_exceptions: bool, root_path: str, client: Tuple[str, int], app_state: Dict[str, Any]) -> None:
        ...

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ...

class TestClient(httpx.Client):
    __test__: bool

    def __init__(self, app: ASGIApp, base_url: str, raise_server_exceptions: bool, root_path: str, backend: str, backend_options: Dict[str, Any], cookies: Optional[Dict[str, str]], headers: Optional[Dict[str, str]], follow_redirects: bool, client: Tuple[str, int]) -> None:
        ...

    @contextlib.contextmanager
    def _portal_factory(self) -> Generator[anyio.abc.BlockingPortal, None, None]:
        ...

    def request(self, method: str, url: Union[str, SplitResult], content: Optional[bytes] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def get(self, url: Union[str, SplitResult], params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def options(self, url: Union[str, SplitResult], params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def head(self, url: Union[str, SplitResult], params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def post(self, url: Union[str, SplitResult], content: Optional[bytes] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def put(self, url: Union[str, SplitResult], content: Optional[bytes] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def patch(self, url: Union[str, SplitResult], content: Optional[bytes] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def delete(self, url: Union[str, SplitResult], params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> httpx.Response:
        ...

    def websocket_connect(self, url: str, subprotocols: Optional[List[str]] = ..., **kwargs: Any) -> WebSocketTestSession:
        ...

    def __enter__(self) -> TestClient:
        ...

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        ...

    async def lifespan(self) -> None:
        ...

    async def wait_startup(self) -> None:
        ...

    async def wait_shutdown(self) -> None:
        ...