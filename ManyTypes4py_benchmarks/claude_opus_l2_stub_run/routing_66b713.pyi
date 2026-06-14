from __future__ import annotations

import contextlib
import re
import typing
from enum import Enum

from starlette.convertors import Convertor
from starlette.datastructures import URLPath
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

class NoMatchFound(Exception):
    def __init__(self, name: str, path_params: dict[str, typing.Any]) -> None: ...

class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

def iscoroutinefunction_or_partial(obj: typing.Any) -> bool: ...

def request_response(func: typing.Callable[..., typing.Any]) -> ASGIApp: ...

def websocket_session(func: typing.Callable[..., typing.Any]) -> ASGIApp: ...

def get_name(endpoint: typing.Callable[..., typing.Any]) -> str: ...

def replace_params(
    path: str,
    param_convertors: dict[str, Convertor[typing.Any]],
    path_params: dict[str, typing.Any],
) -> tuple[str, dict[str, typing.Any]]: ...

PARAM_REGEX: re.Pattern[str]

def compile_path(
    path: str,
) -> tuple[re.Pattern[str], str, dict[str, Convertor[typing.Any]]]: ...

class BaseRoute:
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class Route(BaseRoute):
    path: str
    endpoint: typing.Callable[..., typing.Any]
    name: str
    include_in_schema: bool
    app: ASGIApp
    methods: set[str] | None
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: dict[str, Convertor[typing.Any]]

    def __init__(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        *,
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
        middleware: typing.Sequence[Middleware] | None = None,
    ) -> None: ...
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class WebSocketRoute(BaseRoute):
    path: str
    endpoint: typing.Callable[..., typing.Any]
    name: str
    app: ASGIApp
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: dict[str, Convertor[typing.Any]]

    def __init__(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        *,
        name: str | None = None,
        middleware: typing.Sequence[Middleware] | None = None,
    ) -> None: ...
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class Mount(BaseRoute):
    path: str
    app: ASGIApp
    name: str | None
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: dict[str, Convertor[typing.Any]]

    def __init__(
        self,
        path: str,
        app: ASGIApp | None = None,
        routes: typing.Sequence[BaseRoute] | None = None,
        name: str | None = None,
        *,
        middleware: typing.Sequence[Middleware] | None = None,
    ) -> None: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class Host(BaseRoute):
    host: str
    app: ASGIApp
    name: str | None
    host_regex: re.Pattern[str]
    host_format: str
    param_convertors: dict[str, Convertor[typing.Any]]

    def __init__(
        self,
        host: str,
        app: ASGIApp,
        name: str | None = None,
    ) -> None: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

_T = typing.TypeVar("_T")

class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):
    _cm: contextlib.AbstractContextManager[_T]
    def __init__(self, cm: contextlib.AbstractContextManager[_T]) -> None: ...
    async def __aenter__(self) -> _T: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: typing.Any,
    ) -> bool | None: ...

def _wrap_gen_lifespan_context(
    lifespan_context: typing.Callable[..., typing.Generator[typing.Any, None, None]],
) -> typing.Callable[..., _AsyncLiftContextManager[typing.Any]]: ...

class _DefaultLifespan:
    _router: Router
    def __init__(self, router: Router) -> None: ...
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *exc_info: typing.Any) -> None: ...
    def __call__(self, app: typing.Any) -> _DefaultLifespan: ...

class Router:
    routes: list[BaseRoute]
    redirect_slashes: bool
    default: ASGIApp
    on_startup: list[typing.Callable[[], typing.Any]]
    on_shutdown: list[typing.Callable[[], typing.Any]]
    lifespan_context: typing.Any
    middleware_stack: ASGIApp

    def __init__(
        self,
        routes: typing.Sequence[BaseRoute] | None = None,
        redirect_slashes: bool = True,
        default: ASGIApp | None = None,
        on_startup: typing.Sequence[typing.Callable[[], typing.Any]] | None = None,
        on_shutdown: typing.Sequence[typing.Callable[[], typing.Any]] | None = None,
        lifespan: Lifespan[typing.Any] | None = None,
        *,
        middleware: typing.Sequence[Middleware] | None = None,
    ) -> None: ...
    async def not_found(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def lifespan(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def app(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def mount(self, path: str, app: ASGIApp, name: str | None = None) -> None: ...
    def host(self, host: str, app: ASGIApp, name: str | None = None) -> None: ...
    def add_route(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> None: ...
    def add_websocket_route(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        name: str | None = None,
    ) -> None: ...
    def route(
        self,
        path: str,
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
    ) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...
    def websocket_route(
        self,
        path: str,
        name: str | None = None,
    ) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...
    def add_event_handler(
        self, event_type: str, func: typing.Callable[[], typing.Any]
    ) -> None: ...
    def on_event(
        self, event_type: str
    ) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...