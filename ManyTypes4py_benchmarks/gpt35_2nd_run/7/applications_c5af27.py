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
    def __init__(self, debug: bool = False, routes: typing.Optional[typing.List[BaseRoute]] = None, middleware: typing.Optional[typing.List[Middleware]] = None, exception_handlers: typing.Optional[typing.Dict[typing.Union[int, typing.Type[BaseException]], typing.Callable]] = None, on_startup: typing.Optional[typing.List[typing.Callable]] = None, on_shutdown: typing.Optional[typing.List[typing.Callable]] = None, lifespan: typing.Optional[Lifespan] = None) -> None:
    
    def build_middleware_stack(self) -> ASGIApp:
    
    @property
    def routes(self) -> typing.List[BaseRoute]:
    
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath:
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
    
    def on_event(self, event_type: str) -> typing.Callable:
    
    def mount(self, path: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
    
    def host(self, host: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
    
    def add_middleware(self, middleware_class: typing.Type[BaseHTTPMiddleware], *args: typing.Any, **kwargs: typing.Any) -> None:
    
    def add_exception_handler(self, exc_class_or_status_code: typing.Union[int, typing.Type[BaseException]], handler: typing.Callable) -> None:
    
    def add_event_handler(self, event_type: str, func: typing.Callable) -> None:
    
    def add_route(self, path: str, route: typing.Callable, methods: typing.Optional[typing.List[str]] = None, name: typing.Optional[str] = None, include_in_schema: bool = True) -> None:
    
    def add_websocket_route(self, path: str, route: typing.Callable, name: typing.Optional[str] = None) -> None:
    
    def exception_handler(self, exc_class_or_status_code: typing.Union[int, typing.Type[BaseException]]) -> typing.Callable:
    
    def route(self, path: str, methods: typing.Optional[typing.List[str]] = None, name: typing.Optional[str] = None, include_in_schema: bool = True) -> typing.Callable:
    
    def websocket_route(self, path: str, name: typing.Optional[str] = None) -> typing.Callable:
    
    def middleware(self, middleware_type: str) -> typing.Callable:
