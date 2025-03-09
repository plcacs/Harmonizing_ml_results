from __future__ import annotations
import sys
import typing
import warnings
if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
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
AppType = typing.TypeVar('AppType', bound='Starlette')
P = ParamSpec('P')

class Starlette:

    def __init__(self, debug=False, routes=None, middleware=None, exception_handlers=None, on_startup=None, on_shutdown=None, lifespan=None):
        assert lifespan is None or (on_startup is None and on_shutdown is None), "Use either 'lifespan' or 'on_startup'/'on_shutdown', not both."
        self.debug = debug
        self.state = State()
        self.router = Router(routes, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan)
        self.exception_handlers = {} if exception_handlers is None else dict(exception_handlers)
        self.user_middleware = [] if middleware is None else list(middleware)
        self.middleware_stack: ASGIApp | None = None

    def build_middleware_stack(self):
        debug = self.debug
        error_handler = None
        exception_handlers: dict[typing.Any, typing.Callable[[Request, Exception], Response]] = {}
        for key, value in self.exception_handlers.items():
            if key in (500, Exception):
                error_handler = value
            else:
                exception_handlers[key] = value
        middleware = [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)] + self.user_middleware + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
        app = self.router
        for cls, args, kwargs in reversed(middleware):
            app = cls(app, *args, **kwargs)
        return app

    @property
    def routes(self):
        return self.router.routes

    def url_path_for(self, name: str, /, **path_params: typing.Any):
        return self.router.url_path_for(name, **path_params)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope['app'] = self
        if self.middleware_stack is None:
            self.middleware_stack = self.build_middleware_stack()
        await self.middleware_stack(scope, receive, send)

    def on_event(self, event_type):
        return self.router.on_event(event_type)

    def mount(self, path, app, name=None):
        self.router.mount(path, app=app, name=name)

    def host(self, host, app, name=None):
        self.router.host(host, app=app, name=name)

    def add_middleware(self, middleware_class, *args: P.args, **kwargs: P.kwargs):
        if self.middleware_stack is not None:
            raise RuntimeError('Cannot add middleware after an application has started')
        self.user_middleware.insert(0, Middleware(middleware_class, *args, **kwargs))

    def add_exception_handler(self, exc_class_or_status_code, handler):
        self.exception_handlers[exc_class_or_status_code] = handler

    def add_event_handler(self, event_type, func):
        self.router.add_event_handler(event_type, func)

    def add_route(self, path, route, methods=None, name=None, include_in_schema=True):
        self.router.add_route(path, route, methods=methods, name=name, include_in_schema=include_in_schema)

    def add_websocket_route(self, path, route, name=None):
        self.router.add_websocket_route(path, route, name=name)

    def exception_handler(self, exc_class_or_status_code):
        warnings.warn('The `exception_handler` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/exceptions/ for the recommended approach.', DeprecationWarning)

        def decorator(func):
            self.add_exception_handler(exc_class_or_status_code, func)
            return func
        return decorator

    def route(self, path, methods=None, name=None, include_in_schema=True):
        warnings.warn('The `route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/ for the recommended approach.', DeprecationWarning)

        def decorator(func):
            self.router.add_route(path, func, methods=methods, name=name, include_in_schema=include_in_schema)
            return func
        return decorator

    def websocket_route(self, path, name=None):
        warnings.warn('The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.', DeprecationWarning)

        def decorator(func):
            self.router.add_websocket_route(path, func, name=name)
            return func
        return decorator

    def middleware(self, middleware_type):
        warnings.warn('The `middleware` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/middleware/#using-middleware for recommended approach.', DeprecationWarning)
        assert middleware_type == 'http', 'Currently only middleware("http") is supported.'

        def decorator(func):
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func
        return decorator