```python
from __future__ import annotations
import typing
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec
    P = ParamSpec('P')

AppType = TypeVar('AppType', bound='Starlette')

class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: Dict[Union[int, Type[Exception]], ExceptionHandler]
    user_middleware: List[Middleware]
    middleware_stack: Optional[ASGIApp]
    
    def __init__(
        self,
        debug: bool = ...,
        routes: Optional[List[BaseRoute]] = ...,
        middleware: Optional[List[Middleware]] = ...,
        exception_handlers: Optional[Dict[Union[int, Type[Exception]], ExceptionHandler]] = ...,
        on_startup: Optional[List[Callable[[], Any]]] = ...,
        on_shutdown: Optional[List[Callable[[], Any]]] = ...,
        lifespan: Optional[Lifespan] = ...
    ) -> None: ...
    
    def build_middleware_stack(self) -> ASGIApp: ...
    
    @property
    def routes(self) -> List[BaseRoute]: ...
    
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    
    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    
    def mount(self, path: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...
    
    def host(self, host: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...
    
    def add_middleware(self, middleware_class: Type[BaseHTTPMiddleware], *args: Any, **kwargs: Any) -> None: ...
    
    def add_exception_handler(self, exc_class_or_status_code: Union[int, Type[Exception]], handler: ExceptionHandler) -> None: ...
    
    def add_event_handler(self, event_type: str, func: Callable[[], Any]) -> None: ...
    
    def add_route(
        self,
        path: str,
        route: Callable[[Request], Any],
        methods: Optional[List[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...
    ) -> None: ...
    
    def add_websocket_route(
        self,
        path: str,
        route: Callable[[WebSocket], Any],
        name: Optional[str] = ...
    ) -> None: ...
    
    def exception_handler(self, exc_class_or_status_code: Union[int, Type[Exception]]) -> Callable[[ExceptionHandler], ExceptionHandler]: ...
    
    def route(
        self,
        path: str,
        methods: Optional[List[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...
    ) -> Callable[[Callable[[Request], Any]], Callable[[Request], Any]]: ...
    
    def websocket_route(self, path: str, name: Optional[str] = ...) -> Callable[[Callable[[WebSocket], Any]], Callable[[WebSocket], Any]]: ...
    
    def middleware(self, middleware_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
```