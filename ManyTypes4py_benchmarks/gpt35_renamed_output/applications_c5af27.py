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
    def __init__(self, debug: bool = False, routes: typing.Optional[typing.List[BaseRoute]] = None, middleware: typing.Optional[typing.List[Middleware]] = None,
        exception_handlers: typing.Optional[typing.Dict[typing.Union[int, typing.Type[BaseException]], typing.Callable]] = None, on_startup: typing.Optional[typing.List[typing.Callable]] = None, on_shutdown: typing.Optional[typing.List[typing.Callable]] = None,
        lifespan: typing.Optional[Lifespan] = None) -> None:
        
    def func_z2zwcvqt(self) -> ASGIApp:
        
    @property
    def func_ie6ite3b(self) -> typing.List[BaseRoute]:
        
    def func_9a1t1u43(self, name: str, /, **path_params: typing.Any) -> URLPath:
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        
    def func_e6lwhw4w(self, event_type: str) -> typing.Callable:
        
    def func_rq9lkh2i(self, path: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
        
    def func_vpu4zoww(self, host: str, app: ASGIApp, name: typing.Optional[str] = None) -> None:
        
    def func_5glz2xld(self, middleware_class: typing.Type[BaseHTTPMiddleware], *args: typing.Any, **kwargs: typing.Any) -> None:
        
    def func_divt7vet(self, exc_class_or_status_code: typing.Union[int, typing.Type[BaseException]], handler: typing.Callable) -> None:
        
    def func_70f36ruy(self, event_type: str, func: typing.Callable) -> None:
        
    def func_30hcxufm(self, path: str, route: typing.Callable, methods: typing.Optional[typing.List[str]] = None, name: typing.Optional[str] = None,
        include_in_schema: bool = True) -> None:
        
    def func_9zk3wfxb(self, path: str, route: typing.Callable, name: typing.Optional[str] = None) -> None:
        
    def func_7fsc8g2b(self, exc_class_or_status_code: typing.Union[int, typing.Type[BaseException]]) -> typing.Callable:
        
    def func_k67i7aqa(self, path: str, methods: typing.Optional[typing.List[str]] = None, name: typing.Optional[str] = None,
        include_in_schema: bool = True) -> typing.Callable:
        
    def func_e2i8k3mr(self, path: str, name: typing.Optional[str] = None) -> typing.Callable:
        
    def func_1cqpi3j4(self, middleware_type: str) -> typing.Callable:
