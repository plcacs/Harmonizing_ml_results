from __future__ import annotations

import sys
import typing
from typing import Any, Callable, Sequence

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareFactory
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send

AppType = typing.TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")

class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: dict[Any, ExceptionHandler]
    user_middleware: list[Middleware]
    middleware_stack: ASGIApp | None

    def __init__(
        self,
        debug: bool = ...,
        routes: Sequence[BaseRoute] | None = ...,
        middleware: Sequence[Middleware] | None = ...,
        exception_handlers: typing.Mapping[Any, ExceptionHandler] | None = ...,
        on_startup: Sequence[Callable[[], Any]] | None = ...,
        on_shutdown: Sequence[Callable[[], Any]] | None = ...,
        lifespan: Lifespan[Starlette] | None = ...,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def on_event(self, event_type: str) -> typing.Callable[..., Any]: ...
    def mount(self, path: str, app: ASGIApp, name: str | None = ...) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str | None = ...) -> None: ...
    def add_middleware(
        self, middleware_class: type[_MiddlewareFactory[P]], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...
    def add_exception_handler(
        self, exc_class_or_status_code: int | type[Exception], handler: ExceptionHandler
    ) -> None: ...
    def add_event_handler(self, event_type: str, func: Callable[[], Any]) -> None: ...
    def add_route(
        self,
        path: str,
        route: Callable[..., Any],
        methods: list[str] | None = ...,
        name: str | None = ...,
        include_in_schema: bool = ...,
    ) -> None: ...
    def add_websocket_route(
        self, path: str, route: Callable[..., Any], name: str | None = ...
    ) -> None: ...
    def exception_handler(
        self, exc_class_or_status_code: int | type[Exception]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def route(
        self,
        path: str,
        methods: list[str] | None = ...,
        name: str | None = ...,
        include_in_schema: bool = ...,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def websocket_route(
        self, path: str, name: str | None = ...
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
    def middleware(
        self, middleware_type: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...