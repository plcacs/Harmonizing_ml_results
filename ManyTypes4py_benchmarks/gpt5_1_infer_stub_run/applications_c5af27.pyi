from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar, ParamSpec
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareFactory
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

AppType = TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")


class Starlette:
    debug: bool
    state: State
    router: Router
    exception_handlers: dict[int | type[Exception], ExceptionHandler]
    user_middleware: list[Middleware]
    middleware_stack: ASGIApp | None

    def __init__(
        self,
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[int | type[Exception], ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any] | Callable[[], Awaitable[Any]]] | None = None,
        on_shutdown: Sequence[Callable[[], Any] | Callable[[], Awaitable[Any]]] | None = None,
        lifespan: Lifespan | None = None,
    ) -> None: ...
    def build_middleware_stack(self) -> ASGIApp: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def on_event(
        self, event_type: str
    ) -> Callable[
        [Callable[[], Any] | Callable[[], Awaitable[Any]]],
        Callable[[], Any] | Callable[[], Awaitable[Any]],
    ]: ...
    def mount(self, path: str, app: ASGIApp, name: str | None = None) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str | None = None) -> None: ...
    def add_middleware(
        self, middleware_class: type[_MiddlewareFactory] | _MiddlewareFactory, *args: Any, **kwargs: Any
    ) -> None: ...
    def add_exception_handler(self, exc_class_or_status_code: int | type[Exception], handler: ExceptionHandler) -> None: ...
    def add_event_handler(self, event_type: str, func: Callable[[], Any] | Callable[[], Awaitable[Any]]) -> None: ...
    def add_route(
        self,
        path: str,
        route: Callable[[Request], Response | Awaitable[Response]] | ASGIApp,
        methods: Sequence[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> None: ...
    def add_websocket_route(
        self,
        path: str,
        route: Callable[[WebSocket], Awaitable[None]] | ASGIApp,
        name: str | None = None,
    ) -> None: ...
    def exception_handler(
        self, exc_class_or_status_code: int | type[Exception]
    ) -> Callable[[ExceptionHandler], ExceptionHandler]: ...
    def route(
        self,
        path: str,
        methods: Sequence[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> Callable[
        [Callable[[Request], Response | Awaitable[Response]]],
        Callable[[Request], Response | Awaitable[Response]],
    ]: ...
    def websocket_route(
        self, path: str, name: str | None = None
    ) -> Callable[[Callable[[WebSocket], Awaitable[None]]], Callable[[WebSocket], Awaitable[None]]]: ...
    def middleware(
        self, middleware_type: str
    ) -> Callable[
        [
            Callable[
                [Request, Callable[[Request], Awaitable[Response]]],
                Awaitable[Response],
            ]
        ],
        Callable[
            [Request, Callable[[Request], Awaitable[Response]]],
            Awaitable[Response],
        ],
    ]: ...