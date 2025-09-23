from __future__ import annotations
import sys
import typing
import warnings
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
)
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

AppType = TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")


class Starlette:
    """Creates an Starlette application."""

    def __init__(
        self,
        debug: bool = False,
        routes: Optional[List[BaseRoute]] = None,
        middleware: Optional[List[Middleware]] = None,
        exception_handlers: Optional[
            Mapping[Union[int, type[Exception]], ExceptionHandler]
        ] = None,
        on_startup: Optional[List[Callable[[], Union[Awaitable[Any], Any]]]] = None,
        on_shutdown: Optional[List[Callable[[], Union[Awaitable[Any], Any]]]] = None,
        lifespan: Optional[Lifespan] = None,
    ) -> None:
        """Initializes the application.

        Parameters:
            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette
                application will always automatically include two middleware classes.
                `ServerErrorMiddleware` is added as the very outermost middleware, to handle
                any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal
                with handled exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes,
                or exception class types onto callables which handle the exceptions.
                Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or
                async functions.
            on_startup: A list of callables to run on application startup.
                Startup handler callables do not take any arguments, and may be either
                standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown.
                Shutdown handler callables do not take any arguments, and may be either
                standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform
                startup and shutdown tasks. This is a newer style that replaces the
                `on_startup` and `on_shutdown` handlers. Use one or the other, not both.
        """
        assert lifespan is None or (on_startup is None and on_shutdown is None), (
            "Use either 'lifespan' or 'on_startup'/'on_shutdown', not both."
        )
        self.debug: bool = debug
        self.state: State = State()
        self.router: Router = Router(
            routes, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan
        )
        self.exception_handlers: Dict[
            Union[int, type[Exception]], ExceptionHandler
        ] = {} if exception_handlers is None else dict(exception_handlers)
        self.user_middleware: List[Middleware] = [] if middleware is None else list(middleware)
        self.middleware_stack: Optional[ASGIApp] = None

    def build_middleware_stack(self) -> ASGIApp:
        debug: bool = self.debug
        error_handler: Optional[ExceptionHandler] = None
        exception_handlers: Dict[Any, Callable[[Request, Exception], Response]] = {}
        for key, value in self.exception_handlers.items():
            if key in (500, Exception):
                error_handler = value
            else:
                exception_handlers[key] = value
        middleware: List[Middleware] = (
            [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)]
            + self.user_middleware
            + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
        )
        app: ASGIApp = self.router
        for (cls, args, kwargs) in reversed(middleware):
            app = cls(app, *args, **kwargs)
        return app

    @property
    def routes(self) -> List[BaseRoute]:
        return self.router.routes

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        return self.router.url_path_for(name, **path_params)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope["app"] = self
        if self.middleware_stack is None:
            self.middleware_stack = self.build_middleware_stack()
        await self.middleware_stack(scope, receive, send)

    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Any]:
        return self.router.on_event(event_type)

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None:
        self.router.mount(path, app=app, name=name)

    def host(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None:
        self.router.host(host, app=app, name=name)

    def add_middleware(
        self, middleware_class: _MiddlewareFactory[P], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if self.middleware_stack is not None:
            raise RuntimeError("Cannot add middleware after an application has started")
        self.user_middleware.insert(0, Middleware(middleware_class, *args, **kwargs))

    def add_exception_handler(
        self, exc_class_or_status_code: Union[int, type[Exception]], handler: ExceptionHandler
    ) -> None:
        self.exception_handlers[exc_class_or_status_code] = handler

    def add_event_handler(self, event_type: str, func: Callable[..., Any]) -> None:
        self.router.add_event_handler(event_type, func)

    def add_route(
        self,
        path: str,
        route: Callable[
            [Request], Union[Awaitable[Response], Response]
        ],
        methods: Optional[List[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> None:
        self.router.add_route(path, route, methods=methods, name=name, include_in_schema=include_in_schema)

    def add_websocket_route(
        self,
        path: str,
        route: Callable[[WebSocket], Awaitable[None]],
        name: Optional[str] = None,
    ) -> None:
        self.router.add_websocket_route(path, route, name=name)

    def exception_handler(
        self, exc_class_or_status_code: Union[int, type[Exception]]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            "The `exception_handler` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/exceptions/ for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_exception_handler(exc_class_or_status_code, func)
            return func

        return decorator

    def route(
        self,
        path: str,
        methods: Optional[List[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            "The `route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/ for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.router.add_route(
                path, func, methods=methods, name=name, include_in_schema=include_in_schema
            )
            return func

        return decorator

    def websocket_route(
        self, path: str, name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            "The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.",
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.router.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def middleware(
        self, middleware_type: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            "The `middleware` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/middleware/#using-middleware for recommended approach.",
            DeprecationWarning,
        )
        assert middleware_type == "http", 'Currently only middleware("http") is supported.'

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func

        return decorator