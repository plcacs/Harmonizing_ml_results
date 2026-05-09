from __future__ import annotations
import typing
from typing import Any, Callable, Dict, List, Optional, Union
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

AppType = typing.TypeVar('AppType', bound='Starlette')
P = typing.ParamSpec('P')

class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: Dict[Any, Callable]
    user_middleware: List[Middleware]
    middleware_stack: Optional[ASGIApp]

    def __init__(self, debug: bool = ..., routes: Optional[List[BaseRoute]] = ..., middleware: Optional[List[Middleware]] = ..., exception_handlers: Optional[Dict[Any, Callable]] = ..., on_startup: Optional[List[Callable]] = ..., on_shutdown: Optional[List[Callable]] = ..., lifespan: Optional[Lifespan] = ...) -> None: ...

    def build_middleware_stack(self) -> ASGIApp: ...

    @property
    def routes(self) -> List[BaseRoute]: ...

    def url_path_for(self, name: str, **path_params: Any) -> URLPath: ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

    def on_event(self, event_type: str) -> Callable: ...

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...

    def host(self, host: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...

    def add_middleware(self, middleware_class: type, *args: Any, **kwargs: Any) -> None: ...

    def add_exception_handler(self, exc_class_or_status_code: Any, handler: Callable) -> None: ...

    def add_event_handler(self, event_type: str, func: Callable) -> None: ...

    def add_route(self, path: str, route: Union[BaseRoute, Callable], methods: Optional[List[str]] = ..., name: Optional[str] = ..., include_in_schema: bool = ...) -> None: ...

    def add_websocket_route(self, path: str, route: Union[BaseRoute, Callable], name: Optional[str] = ...) -> None: ...

    def exception_handler(self, exc_class_or_status_code: Any) -> Callable: ...

    def route(self, path: str, methods: Optional[List[str]] = ..., name: Optional[str] = ..., include_in_schema: bool = ...) -> Callable: ...

    def websocket_route(self, path: str, name: Optional[str] = ...) -> Callable: ...

    def middleware(self, middleware_type: str) -> Callable: ...