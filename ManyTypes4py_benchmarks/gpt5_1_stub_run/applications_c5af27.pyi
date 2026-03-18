from typing import Any, Dict, List, Optional, Sequence, TypeVar
from typing_extensions import ParamSpec
import sys
import typing
import warnings
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

AppType = TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")


class Starlette:
    def __init__(
        self,
        debug: bool = ...,
        routes: Any = ...,
        middleware: Any = ...,
        exception_handlers: Any = ...,
        on_startup: Any = ...,
        on_shutdown: Any = ...,
        lifespan: Any = ...,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> List[BaseRoute]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def on_event(self, event_type: str) -> Any: ...
    def mount(self, path: str, app: Any, name: Optional[str] = ...) -> None: ...
    def host(self, host: str, app: Any, name: Optional[str] = ...) -> None: ...
    def add_middleware(self, middleware_class: Any, *args: Any, **kwargs: Any) -> None: ...
    def add_exception_handler(self, exc_class_or_status_code: Any, handler: Any) -> None: ...
    def add_event_handler(self, event_type: str, func: Any) -> None: ...
    def add_route(
        self,
        path: str,
        route: Any,
        methods: Optional[Sequence[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> None: ...
    def add_websocket_route(self, path: str, route: Any, name: Optional[str] = ...) -> None: ...
    def exception_handler(self, exc_class_or_status_code: Any) -> Any: ...
    def route(
        self,
        path: str,
        methods: Optional[Sequence[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> Any: ...
    def websocket_route(self, path: str, name: Optional[str] = ...) -> Any: ...
    def middleware(self, middleware_type: str) -> Any: ...

    debug: bool
    state: State
    router: Router
    exception_handlers: Dict[Any, Any]
    user_middleware: List[Middleware]
    middleware_stack: Optional[ASGIApp]