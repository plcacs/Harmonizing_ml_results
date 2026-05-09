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
    def __init__(self, name: str, path_params: dict):
        super().__init__()

class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

def iscoroutinefunction_or_partial(obj: typing.Any) -> bool:
    return False

def request_response(func: typing.Callable) -> ASGIApp:
    return ASGIApp()

def websocket_session(func: typing.Callable) -> ASGIApp:
    return ASGIApp()

def get_name(endpoint: typing.Union[typing.Callable, type]) -> str:
    return ""

def replace_params(path: str, param_convertors: dict[str, Convertor], path_params: dict[str, typing.Any]) -> tuple[str, dict[str, typing.Any]]:
    return ("", {})

PARAM_REGEX = re.compile('{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}')

def compile_path(path: str) -> tuple[re.Pattern, str, dict[str, Convertor]]:
    return (re.compile(""), "", {})

class BaseRoute:
    def matches(self, scope: Scope) -> tuple[Match, dict]:
        return (Match.NONE, {})

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

class Route(BaseRoute):
    def __init__(self, path: str, endpoint: typing.Callable, methods: typing.Optional[list[str]] = None, name: typing.Optional[str] = None, include_in_schema: bool = True, middleware: typing.Optional[list[Middleware]] = None):
        pass

    def matches(self, scope: Scope) -> tuple[Match, dict]:
        return (Match.NONE, {})

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

class WebSocketRoute(BaseRoute):
    def __init__(self, path: str, endpoint: typing.Callable, name: typing.Optional[str] = None, middleware: typing.Optional[list[Middleware]] = None):
        pass

    def matches(self, scope: Scope) -> tuple[Match, dict]:
        return (Match.NONE, {})

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

class Mount(BaseRoute):
    def __init__(self, path: str, app: typing.Optional[ASGIApp] = None, routes: typing.Optional[list[BaseRoute]] = None, name: typing.Optional[str] = None, middleware: typing.Optional[list[Middleware]] = None):
        pass

    def matches(self, scope: Scope) -> tuple[Match, dict]:
        return (Match.NONE, {})

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

class Host(BaseRoute):
    def __init__(self, host: str, app: ASGIApp, name: typing.Optional[str] = None):
        pass

    def matches(self, scope: Scope) -> tuple[Match, dict]:
        return (Match.NONE, {})

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

_T = typing.TypeVar('_T')

class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):
    def __init__(self, cm: typing.ContextManager[_T]):
        pass

    async def __aenter__(self) -> _T:
        return typing.cast(_T, None)

    async def __aexit__(self, exc_type: typing.Optional[typing.Type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
        return None

def _wrap_gen_lifespan_context(lifespan_context: typing.Callable[[ASGIApp], typing.ContextManager[typing.Optional[dict]]]) -> typing.Callable[[ASGIApp], _AsyncLiftContextManager]:
    def wrapper(app: ASGIApp) -> _AsyncLiftContextManager:
        return _AsyncLiftContextManager(lifespan_context(app))
    return wrapper

class _DefaultLifespan:
    def __init__(self, router: Router):
        pass

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, exc_type: typing.Optional[typing.Type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> None:
        pass

    def __call__(self, app: ASGIApp) -> '_DefaultLifespan':
        return self

class Router:
    def __init__(self, routes: typing.Optional[list[BaseRoute]] = None, redirect_slashes: bool = True, default: typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]] = None, on_startup: typing.Optional[list[typing.Callable]] = None, on_shutdown: typing.Optional[list[typing.Callable]] = None, lifespan: typing.Optional[typing.Union[typing.Callable, typing.AsyncGenerator]] = None, middleware: typing.Optional[list[Middleware]] = None):
        pass

    async def not_found(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

    def url_path_for(self, name: str, **path_params: typing.Any) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

    async def app(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass

    def mount(self, path: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
        pass

    def host(self, host: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
        pass

    def add_route(self, path: str, endpoint: typing.Callable, methods: typing.Optional[list[str]] = None, name: typing.Optional[str] = None, include_in_schema: bool = True) -> None:
        pass

    def add_websocket_route(self, path: str, endpoint: typing.Callable, name: typing.Optional[str] = None) -> None:
        pass

    def route(self, path: str, methods: typing.Optional[list[str]] = None, name: typing.Optional[str] = None, include_in_schema: bool = True) -> typing.Callable:
        def decorator(func: typing.Callable) -> typing.Callable:
            return func
        return decorator

    def websocket_route(self, path: str, name: typing.Optional[str] = None) -> typing.Callable:
        def decorator(func: typing.Callable) -> typing.Callable:
            return func
        return decorator

    def add_event_handler(self, event_type: str, func: typing.Callable) -> None:
        pass

    def on_event(self, event_type: str) -> typing.Callable:
        def decorator(func: typing.Callable) -> typing.Callable:
            return func
        return decorator