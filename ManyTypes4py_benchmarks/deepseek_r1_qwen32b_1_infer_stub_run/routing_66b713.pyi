from __future__ import annotations
import re
import typing
from enum import Enum
from starlette.datastructures import URLPath
from starlette.types import ASGIApp
from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

if typing.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.types import Receive, Scope, Send
    from starlette.websockets import WebSocket

class NoMatchFound(Exception):
    def __init__(self, name: str, path_params: dict) -> None:
        ...

class Match(Enum):
    NONE: int
    PARTIAL: int
    FULL: int

def iscoroutinefunction_or_partial(obj: Any) -> bool:
    ...

def request_response(func: Any) -> ASGIApp:
    ...

def websocket_session(func: Any) -> ASGIApp:
    ...

def get_name(endpoint: Any) -> str:
    ...

def replace_params(path: str, param_convertors: dict, path_params: dict) -> tuple[str, dict]:
    ...

def compile_path(path: str) -> tuple[re.Pattern, str, dict]:
    ...

class BaseRoute:
    def matches(self, scope: dict) -> tuple[Match, dict]:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def handle(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

class Route(BaseRoute):
    def __init__(
        self,
        path: str,
        endpoint: Any,
        *,
        methods: Optional[list[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
        middleware: Optional[list[tuple[Any, list, dict]]] = None,
    ) -> None:
        ...

    def matches(self, scope: dict) -> tuple[Match, dict]:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def handle(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class WebSocketRoute(BaseRoute):
    def __init__(
        self,
        path: str,
        endpoint: Any,
        *,
        name: Optional[str] = None,
        middleware: Optional[list[tuple[Any, list, dict]]] = None,
    ) -> None:
        ...

    def matches(self, scope: dict) -> tuple[Match, dict]:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def handle(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class Mount(BaseRoute):
    def __init__(
        self,
        path: str,
        app: Optional[Any] = None,
        routes: Optional[list[BaseRoute]] = None,
        name: Optional[str] = None,
        *,
        middleware: Optional[list[tuple[Any, list, dict]]] = None,
    ) -> None:
        ...

    def matches(self, scope: dict) -> tuple[Match, dict]:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def handle(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class Host(BaseRoute):
    def __init__(self, host: str, app: Any, name: Optional[str] = None) -> None:
        ...

    def matches(self, scope: dict) -> tuple[Match, dict]:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def handle(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class _AsyncLiftContextManager:
    def __init__(self, cm: ContextManager[Any]) -> None:
        ...

    async def __aenter__(self) -> Any:
        ...

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[types.TracebackType]) -> Optional[bool]:
        ...

class _DefaultLifespan:
    def __init__(self, router: Any) -> None:
        ...

    async def __aenter__(self) -> None:
        ...

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[types.TracebackType]) -> None:
        ...

    def __call__(self, app: Any) -> Any:
        ...

class Router:
    def __init__(
        self,
        routes: Optional[list[BaseRoute]] = None,
        redirect_slashes: bool = True,
        default: Optional[Callable[[dict, Receive, Send], Any]] = None,
        on_startup: Optional[list[Callable[[], Any]]] = None,
        on_shutdown: Optional[list[Callable[[], Any]]] = None,
        lifespan: Optional[Any] = None,
        *,
        middleware: Optional[list[tuple[Any, list, dict]]] = None,
    ) -> None:
        ...

    async def not_found(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        ...

    async def startup(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

    async def lifespan(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    async def app(self, scope: dict, receive: Receive, send: Send) -> None:
        ...

    def mount(self, path: str, app: Any, name: Optional[str] = None) -> None:
        ...

    def host(self, host: str, app: Any, name: Optional[str] = None) -> None:
        ...

    def add_route(
        self,
        path: str,
        endpoint: Any,
        methods: Optional[list[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> None:
        ...

    def add_websocket_route(self, path: str, endpoint: Any, name: Optional[str] = None) -> None:
        ...

    def route(
        self,
        path: str,
        methods: Optional[list[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> Callable[[Callable], Callable]:
        ...

    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[Callable], Callable]:
        ...

    def add_event_handler(self, event_type: str, func: Callable) -> None:
        ...

    def on_event(self, event_type: str) -> Callable[[Callable], Callable]:
        ...