from __future__ import annotations

import sys
import typing
import warnings

if sys.version_info >= (3, 10):  # pragma: no cover
    from typing import ParamSpec
else:  # pragma: no cover
    from typing_extensions import ParamSpec

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

AppType = typing.TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")


class Starlette:
    def __init__(
        self: AppType,
        debug: bool = False,
        routes: typing.Sequence[BaseRoute] | None = None,
        middleware: typing.Sequence[Middleware] | None = None,
        exception_handlers: typing.Mapping[typing.Any, ExceptionHandler] | None = None,
        on_startup: typing.Sequence[typing.Callable[[], typing.Any]] | None = None,
        on_shutdown: typing.Sequence[typing.Callable[[], typing.Any]] | None = None,
        lifespan: Lifespan[AppType] | None = None,
    ) -> None:
        assert lifespan is None or (on_startup is None and on_shutdown is None), (
            "Use either 'lifespan' or 'on_startup'/'on_shutdown', not both."
        )

        self.debug = debug
        self.state = State()
        self.router = Router(routes, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan)
        self.exception_handlers = {} if exception_handlers is None else dict(exception_handlers)
        self.user_middleware = [] if middleware is None else list(middleware)
        self.middleware_stack: ASGIApp | None = None

    def build_middleware_stack(self) -> ASGIApp:
        debug = self.debug
        error_handler = None
        exception_handlers: dict[typing.Any, typing.Callable[[Request, Exception], Response]] = {}

        for key, value in self.exception_handlers.items():
            if key in (500, Exception):
                error_handler = value
            else:
                exception_handlers[key] = value

        middleware = (
            [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)]
            + self.user_middleware
            + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
        )

        app = self.router
        for cls, args, kwargs in reversed(middleware):
            app = cls(app, *args, **kwargs)
        return app

    @property
    def routes(self) -> list[BaseRoute]:
        return self.router.routes

    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath:
        return self.router.url_path_for(name, **path_params)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope["app"] = self
        if self.middleware_stack is None:
            self.middleware_stack = self.build_middleware_stack()
        await self.middleware_stack(scope, receive, send)

    def on_event(self, event_type: str) -> typing.Callable[[typing.Callable], typing.Callable]:
        return self.router.on_event(event_type)

    def mount(self, path: str, app: ASGIApp, name: str | None = None) -> None:
        self.router.mount(path, app=app, name=name)

    def host(self, host: str, app: ASGIApp, name: str | None = None) -> None:
        self.router.host(host, app=app, name=name)

    def add_middleware(
        self,
        middleware_class: _MiddlewareFactory[P],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        if self.middleware_stack is not None:
            raise RuntimeError("Cannot add middleware after an application has started")
        self.user_middleware.insert(0, Middleware(middleware_class, *args, **kwargs))

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: ExceptionHandler,
    ) -> None:
        self.exception_handlers[exc_class_or_status_code] = handler

    def add_event_handler(
        self,
        event_type: str,
        func: typing.Callable[[], typing.Any],
    ) -> None:
        self.router.add_event_handler(event_type, func)

    def add_route(
        self,
        path: str,
        route: typing.Callable[[Request], typing.Awaitable[Response] | Response],
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> None:
        self.router.add_route(path, route, methods=methods, name=name, include_in_schema=include_in_schema)

    def add_websocket_route(
        self,
        path: str,
        route: typing.Callable[[WebSocket], typing.Awaitable[None]],
        name: str | None = None,
    ) -> None:
        self.router.add_websocket_route(path, route, name=name)

    def exception_handler(self, exc_class_or_status_code: int | type[Exception]) -> typing.Callable[[typing.Callable], typing.Callable]:
        warnings.warn(
            "The `exception_handler` decorator is deprecated, and will be removed in version 1.0.0. "
            "Refer to https://www.starlette.io/exceptions/ for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: typing.Callable) -> typing.Callable:
            self.add_exception_handler(exc_class_or_status_code, func)
            return func

        return decorator

    def route(
        self,
        path: str,
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> typing.Callable[[typing.Callable], typing.Callable]:
        warnings.warn(
            "The `route` decorator is deprecated, and will be removed in version 1.0.0. "
            "Refer to https://www.starlette.io/routing/ for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: typing.Callable) -> typing.Callable:
            self.router.add_route(
                path,
                func,
                methods=methods,
                name=name,
                include_in_schema=include_in_schema,
            )
            return func

        return decorator

    def websocket_route(self, path: str, name: str | None = None) -> typing.Callable[[typing.Callable], typing.Callable]:
        warnings.warn(
            "The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. "
            "Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: typing.Callable) -> typing.Callable:
            self.router.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def middleware(self, middleware_type: str) -> typing.Callable[[typing.Callable], typing.Callable]:
        warnings.warn(
            "The `middleware` decorator is deprecated, and will be removed in version 1.0.0. "
            "Refer to https://www.starlette.io/middleware/#using-middleware for recommended approach.",
            DeprecationWarning,
        )
        assert middleware_type == "http", 'Currently only middleware("http") is supported.'

        def decorator(func: typing.Callable) -> typing.Callable:
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func

        return decorator
