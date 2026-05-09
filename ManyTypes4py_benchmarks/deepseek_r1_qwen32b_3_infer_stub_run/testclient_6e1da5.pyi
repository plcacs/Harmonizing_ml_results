import contextlib
import io
import json
import math
import sys
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
    TypeGuard,
    Union,
    overload,
)
from concurrent.futures import Future
from urllib.parse import unquote, urljoin
from anyio.abc import BlockingPortal
from anyio.streams.stapled import StapledObjectStream
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from httpx import (
    BaseTransport,
    Client,
    Request,
    Response,
    ByteStream,
    WebSocket,
)
from httpx._client import USE_CLIENT_DEFAULT

class _Upgrade(Exception):
    def __init__(self, session: 'WebSocketTestSession') -> None:
        ...

class WebSocketDenialResponse(Response, WebSocketDisconnect):
    def __init__(self, status_code: int, headers: List[Tuple[bytes, bytes]], content: bytes) -> None:
        ...

class WebSocketTestSession:
    def __init__(self, app: ASGIApp, scope: Scope, portal_factory: Callable[[], ContextManager[BlockingPortal]]) -> None:
        ...

    def __enter__(self) -> 'WebSocketTestSession':
        ...

    def __exit__(self, *args: Any) -> None:
        ...

    def send(self, message: Message) -> None:
        ...

    def send_text(self, data: str) -> None:
        ...

    def send_bytes(self, data: bytes) -> None:
        ...

    def send_json(self, data: Any, mode: str) -> None:
        ...

    def close(self, code: int = ..., reason: Optional[str] = ...) -> None:
        ...

    def receive(self) -> Message:
        ...

    def receive_text(self) -> str:
        ...

    def receive_bytes(self) -> bytes:
        ...

    def receive_json(self, mode: str) -> Any:
        ...

class _TestClientTransport(BaseTransport):
    def __init__(self, app: ASGIApp, portal_factory: Callable[[], ContextManager[BlockingPortal]], raise_server_exceptions: bool, root_path: str, client: Tuple[str, int], app_state: Dict[str, Any]) -> None:
        ...

    def handle_request(self, request: Request) -> Response:
        ...

class TestClient(Client):
    def __init__(self, app: ASGIApp, base_url: str = ..., raise_server_exceptions: bool = ..., root_path: str = ..., backend: str = ..., backend_options: Dict[str, Any] = ..., cookies: Optional[Dict[str, str]] = ..., headers: Optional[Dict[str, str]] = ..., follow_redirects: bool = ..., client: Tuple[str, int] = ...) -> None:
        ...

    @contextlib.contextmanager
    def _portal_factory(self) -> Generator[BlockingPortal, None, None]:
        ...

    def request(self, method: str, url: str, content: Optional[Any] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def get(self, url: str, params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def options(self, url: str, params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def head(self, url: str, params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def post(self, url: str, content: Optional[Any] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def put(self, url: str, content: Optional[Any] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def patch(self, url: str, content: Optional[Any] = ..., data: Optional[Any] = ..., files: Optional[Any] = ..., json: Optional[Any] = ..., params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def delete(self, url: str, params: Optional[Any] = ..., headers: Optional[Dict[str, str]] = ..., cookies: Optional[Dict[str, str]] = ..., auth: Any = ..., follow_redirects: Any = ..., timeout: Any = ..., extensions: Optional[Dict[str, Any]] = ...) -> Response:
        ...

    def websocket_connect(self, url: str, subprotocols: Optional[List[str]] = ..., **kwargs: Any) -> WebSocketTestSession:
        ...

    def __enter__(self) -> 'TestClient':
        ...

    def __exit__(self, *args: Any) -> None:
        ...