from __future__ import annotations
import sys
import typing
from typing import Any, AnyStr, Callable, Dict, List, Optional, Union
from typing_extensions import ParamSpec
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.routing import BaseRoute, Router
from starlette.types import (
    ASGIApp,
    ExceptionHandler,
    Lifespan,
    Receive,
    Scope,
    Send,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket

AppType = typing.TypeVar('AppType', bound='Starlette')
P = ParamSpec('P')

class Starlette:
    def __init__(
        self,
        debug: bool = ...,
        routes: Optional[List[Any]] = ...,
        middleware: Optional[List[Middleware]] = ...,
        exception_handlers: Optional[Dict[Any, Any]] = ...,
        on_startup: Optional[List[Callable]] = ...,
        on_shutdown: Optional[List[Callable]] = ...,
        lifespan: Optional[Lifespan] = ...,
    ) -> None:
        ...

    def build_middleware_stack(self) -> ASGIApp:
        ...

    @property
    def routes(self) -> List[BaseRoute]:
        ...

    def url_path_for(self, name: str, **path_params: Any) -> URLPath:
        ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        ...

    def on_event(self, event_type: str) -> Callable[[Callable], Callable]:
        ...

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = ...) -> None:
        ...

    def host(self, path: str, app: ASGIApp, name: Optional[str] = ...) -> None:
        ...

    def add_middleware(
        self,
        middleware_class: typing.Type[BaseHTTPMiddleware],
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...

    def add_exception_handler(
        self,
        exc_class_or_status_code: Union[int, typing.Type[Exception]],
        handler: ExceptionHandler,
    ) -> None:
        ...

    def add_event_handler(
        self, event_type: str, func: Callable[..., Optional[Any]]
    ) -> None:
        ...

    def add_route(
        self,
        path: Union[str, URLPath],
        route: Union[Callable, BaseRoute],
        methods: Optional[List[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> None:
        ...

    def add_websocket_route(
        self, path: Union[str, URLPath], route: Union[Callable, BaseRoute], name: Optional[str] = ...
    ) -> None:
        ...

    def exception_handler(
        self, exc_class_or_status_code: Union[int, typing.Type[Exception]]
    ) -> Callable[[Callable], Callable]:
        ...

    def route(
        self,
        path: Union[str, URLPath],
        methods: Optional[List[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> Callable[[Callable], Callable]:
        ...

    def websocket_route(
        self, path: Union[str, URLPath], name: Optional[str] = ...
    ) -> Callable[[Callable], Callable]:
        ...

    def middleware(self, middleware_type: str) -> Callable[[Callable], Callable]:
        ...