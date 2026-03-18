```pyi
from __future__ import annotations

import sys
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

AppType = TypeVar('AppType', bound='Starlette')
P = ParamSpec('P')

class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: Dict[Union[int, type], Any]
    user_middleware: List[Middleware]
    middleware_stack: Optional[ASGIApp]

    def __init__(
        self,
        debug: bool = False,
        routes: Optional[List[BaseRoute]] = None,
        middleware: Optional[List[Middleware]] = None,
        exception_handlers: Optional[Mapping[Union[int, type], ExceptionHandler]] = None,
        on_startup: Optional[List[Callable[[], Any]]] = None,
        on_shutdown: Optional[List[Callable[[], Any]]] = None,
        lifespan: Optional[Lifespan] = None,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> List[BaseRoute]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def mount(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None: ...
    def host(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None: ...
    def add_middleware(self, middleware_class: type, *args: Any, **kwargs: Any) -> None: ...
    def add_exception_handler(self, exc_class_or_status_code: Union[int, type], handler: ExceptionHandler) -> None: ...
    def add_event_handler(self, event_type: str, func: Callable[..., Any]) -> None: ...
    def add_route(
        self,
        path: str,
        route: Callable[..., Any],
        methods: Optional[List[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> None: ...
    def add_websocket_route(self, path: str, route: Callable[..., Any], name: Optional[str] = None) -> None: ...
    def exception_handler(self, exc_class_or_status_code: Union[int, type]) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def route(
        self,
        path: str,
        methods: Optional[List[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def middleware(self, middleware_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
```