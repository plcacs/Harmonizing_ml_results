from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, TypeVar, Union
from typing_extensions import ParamSpec
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareFactory
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send

AppType = TypeVar("AppType", bound="Starlette")
P = ParamSpec("P")


class Starlette:
    def __init__(
        self,
        debug: bool = ...,
        routes: Optional[Sequence[BaseRoute]] = ...,
        middleware: Optional[Sequence[Union[Middleware, _MiddlewareFactory]]] = ...,
        exception_handlers: Optional[Mapping[Union[int, Type[Exception]], ExceptionHandler]] = ...,
        on_startup: Optional[Sequence[Callable[..., Any]]] = ...,
        on_shutdown: Optional[Sequence[Callable[..., Any]]] = ...,
        lifespan: Optional[Lifespan[Any]] = ...,
    ) -> None: ...
    debug: bool
    state: State
    router: Router
    exception_handlers: Dict[Union[int, Type[Exception]], ExceptionHandler]
    user_middleware: List[Union[Middleware, _MiddlewareFactory]]
    middleware_stack: Optional[ASGIApp]

    def build_middleware_stack(self) -> ASGIApp: ...

    @property
    def routes(self) -> List[BaseRoute]: ...

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath: ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

    def on_event(self, event_type: str) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...

    def host(self, host: str, app: ASGIApp, name: Optional[str] = ...) -> None: ...

    def add_middleware(self, middleware_class: Type[Any], *args: Any, **kwargs: Any) -> None: ...

    def add_exception_handler(self, exc_class_or_status_code: Union[int, Type[Exception]], handler: ExceptionHandler) -> None: ...

    def add_event_handler(self, event_type: str, func: Callable[..., Any]) -> None: ...

    def add_route(
        self,
        path: str,
        route: Any,
        methods: Optional[Sequence[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> None: ...

    def add_websocket_route(self, path: str, route: Any, name: Optional[str] = ...) -> None: ...

    def exception_handler(self, exc_class_or_status_code: Union[int, Type[Exception]]) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...

    def route(
        self,
        path: str,
        methods: Optional[Sequence[str]] = ...,
        name: Optional[str] = ...,
        include_in_schema: bool = ...,
    ) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...

    def websocket_route(self, path: str, name: Optional[str] = ...) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...

    def middleware(self, middleware_type: str) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...