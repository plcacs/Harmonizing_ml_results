from __future__ import annotations
import typing
from typing import Any, Callable, Optional, Sequence, Union, TypeVar, ParamSpec
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send

AppType = TypeVar('AppType', bound='Starlette')
P = ParamSpec('P')

class Starlette:
    """Creates an Starlette application."""

    debug: bool
    state: State
    router: Router
    exception_handlers: dict[Union[int, typing.Type[Exception]], ExceptionHandler]
    user_middleware: list[Middleware]
    middleware_stack: Optional[ASGIApp]

    def __init__(
        self,
        debug: bool = False,
        routes: Optional[Sequence[BaseRoute]] = None,
        middleware: Optional[Sequence[Middleware]] = None,
        exception_handlers: Optional[Union[dict[Union[int, typing.Type[Exception]], ExceptionHandler], Sequence[typing.Tuple[Union[int, typing.Type[Exception]], ExceptionHandler]]]] = None,
        on_startup: Optional[Sequence[Callable[[], Any]]] = None,
        on_shutdown: Optional[Sequence[Callable[[], Any]]] = None,
        lifespan: Optional[Lifespan] = None,
    ) -> None: ...

    def build_middleware_stack(self) -> ASGIApp: ...

    @property
    def routes(self) -> Sequence[BaseRoute]: ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

    def on_event(self, event_type: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]: ...

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None: ...

    def host(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None: ...

    def add_middleware(self, middleware_class: typing.Type[Any], *args: Any, **kwargs: Any) -> None: ...

    def add_exception_handler(self, exc_class_or_status_code: Union[int, typing.Type[Exception]], handler: ExceptionHandler) -> None: ...

    def add_event_handler(self, event_type: str, func: Callable[[], Any]) -> None: ...

    def add_route(
        self,
        path: str,
        route: Callable[[Scope, Receive, Send], Any],
        methods: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> None: ...

    def add_websocket_route(self, path: str, route: Callable[[Scope, Receive, Send], Any], name: Optional[str] = None) -> None: ...

    def exception_handler(self, exc_class_or_status_code: Union[int, typing.Type[Exception]]) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def route(
        self,
        path: str,
        methods: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def middleware(self, middleware_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...