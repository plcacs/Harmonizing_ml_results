from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar
from typing import ParamSpec

from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareFactory
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
F = TypeVar("F", bound=Callable[..., Any])


class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: dict[int | type[Exception], ExceptionHandler]
    user_middleware: list[Middleware]
    middleware_stack: ASGIApp | None

    def __init__(
        self,
        debug: bool = ...,
        routes: Sequence[BaseRoute] | None = ...,
        middleware: Sequence[Middleware] | None = ...,
        exception_handlers: Mapping[int | type[Exception], ExceptionHandler] | None = ...,
        on_startup: Sequence[Callable[[], Any]] | None = ...,
        on_shutdown: Sequence[Callable[[], Any]] | None = ...,
        lifespan: Lifespan[Any] | None = ...,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> Sequence[BaseRoute]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]: ...
    def on_event(self, event_type: str) -> Callable[[F], F]: ...
    def mount(self, path: str, app: ASGIApp, name: str | None = ...) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str | None = ...) -> None: ...
    def add_middleware(self, middleware_class: type[BaseHTTPMiddleware], *args: Any, **kwargs: Any) -> None: ...
    def add_exception_handler(self, exc_class_or_status_code: int | type[Exception], handler: ExceptionHandler) -> None: ...
    def add_event_handler(self, event_type: str, func: Callable[[], Any]) -> None: ...
    def add_route(
        self,
        path: str,
        route: ASGIApp | Callable[..., Any],
        methods: Sequence[str] | None = ...,
        name: str | None = ...,
        include_in_schema: bool = ...,
    ) -> None: ...
    def add_websocket_route(self, path: str, route: ASGIApp | Callable[..., Any], name: str | None = ...) -> None: ...
    def exception_handler(self, exc_class_or_status_code: int | type[Exception]) -> Callable[[ExceptionHandler], ExceptionHandler]: ...
    def route(
        self,
        path: str,
        methods: Sequence[str] | None = ...,
        name: str | None = ...,
        include_in_schema: bool = ...,
    ) -> Callable[[F], F]: ...
    def websocket_route(self, path: str, name: str | None = ...) -> Callable[[F], F]: ...
    def middleware(self, middleware_type: str) -> Callable[[F], F]: ...