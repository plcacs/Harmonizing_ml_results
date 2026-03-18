```pyi
from __future__ import annotations

import contextlib
import functools
import inspect
import re
import traceback
import types
import typing
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from starlette._exception_handler import wrap_app_handling_exceptions
from starlette._utils import get_route_path, is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.convertors import CONVERTOR_TYPES, Convertor
from starlette.datastructures import URL, Headers, URLPath
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketClose

class NoMatchFound(Exception):
    def __init__(self, name: str, path_params: dict[str, typing.Any]) -> None: ...

class Match(Enum):
    NONE: int
    PARTIAL: int
    FULL: int

def iscoroutinefunction_or_partial(obj: typing.Any) -> bool: ...

def request_response(func: typing.Callable[[Request], typing.Awaitable[Response]] | typing.Callable[[Request], Response]) -> ASGIApp: ...

def websocket_session(func: typing.Callable[[WebSocket], typing.Awaitable[None]]) -> ASGIApp: ...

def get_name(endpoint: typing.Any) -> str: ...

def replace_params(
    path: str,
    param_convertors: dict[str, Convertor],
    path_params: dict[str, typing.Any],
) -> tuple[str, dict[str, typing.Any]]: ...

PARAM_REGEX: re.Pattern[str]

def compile_path(path: str) -> tuple[re.Pattern[str], str, dict[str, Convertor]]: ...

class BaseRoute:
    def matches(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class Route(BaseRoute):
    path: str
    endpoint: typing.Any
    name: str
    include_in_schema: bool
    app: ASGIApp
    methods: set[str] | None
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: dict[str, Convertor]

    def __init__(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        *,
        methods: list[str] | None = None,
        name: str | None = None,
        include_in_schema: bool = True,
        middleware: list[tuple[type[typing.Any], tuple[typing.Any, ...], dict[str, typing.Any]]] | None = None,
    ) -> None: ...
    def matches(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class WebSocketRoute(BaseRoute):
    path: str
    endpoint: typing.Any
    name: str
    app: ASGIApp
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: dict[str, Convertor]

    def __init__(
        self,
        path: str,
        endpoint: typing.Callable[..., typing.Any],
        *,
        name: str | None = None,
        middleware: list[tuple[type[typing.Any], tuple[typing.Any, ...], dict[str, typing.Any]]] | None = None,
    ) -> None: ...
    def matches(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]: ...
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
    param_convertors: dict[str, Convertor]
    _base_app: ASGIApp

    def __init__(
        self,
        path: str,
        app: ASGIApp | None = None,
        routes: list[BaseRoute] | None = None,
        name: str | None = None,
        *,
        middleware: list[tuple[type[typing.Any], tuple[typing.Any, ...], dict[str, typing.Any]]] | None = None,
    ) -> None: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def matches(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]: ...
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
    param_convertors: dict[str, Convertor]

    def __init__(self, host: str, app: ASGIApp, name: str | None = None) -> None: ...
    @property
    def routes(self) -> list[BaseRoute]: ...
    def matches(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

_T = typing.TypeVar('_T')

class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):
    _cm: typing.ContextManager[_T]
    def __init__(self, cm: typing.ContextManager[_T]) -> None: ...
    async def __aenter__(self) -> _T: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def _wrap_gen_lifespan_context(
    lifespan_context: typing.Callable[[typing.Any], typing.Generator[typing.Any, None, None]],
) -> typing.Callable[[typing.Any], _AsyncLiftContextManager[typing.Any]]: ...

class _DefaultLifespan:
    _router: Router
    def __init__(self, router: Router) -> None: ...
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *exc_info: typing.Any) -> None: ...
    def __call__(self, app: typing.Any) -> _DefaultLifespan: ...

class Router:
    routes: list[BaseRoute]
    redirect_slashes: bool
    default: typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
    on_startup: list[typing.Callable[[], typing.Any]]
    on_shutdown: list[typing.Callable[[], typing.Any]]
    lifespan_context: typing.AsyncContextManager[typing.Any]
    middleware_stack: ASGIApp

    def __init__(
        self,
        routes: list[BaseRoute] | None = None,
        redirect_slashes: bool = True,
        default: typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]] | None = None,
        on_startup: list[typing.Callable[[], typing.Any]] | None = None,
        on_shutdown: list[typing.Callable[[], typing.Any]] | None = None,
        lifespan: typing.Callable[[typing.Any], typing.AsyncContextManager[typing.Any]] | None = None,
        *,
        middleware: list[tuple[type[typing.Any], tuple[typing.Any, ...], dict[str, typing.Any]]] | None = None,
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
    def add_websocket_route(self, path: str, endpoint: typing.Callable[..., typing.Any], name: str | None = None) -> None: ...
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
    def add_event_handler(self, event_type: str, func: typing.Callable[..., typing.Any]) -> None: ...
    def on_event(
        self,
        event_type: str,
    ) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...
```