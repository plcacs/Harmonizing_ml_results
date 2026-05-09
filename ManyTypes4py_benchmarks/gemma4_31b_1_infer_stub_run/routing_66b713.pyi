from __future__ import annotations
import functools
import re
import typing
from enum import Enum
from starlette.datastructures import URL, URLPath
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.convertors import Convertor

NoMatchFound = type[Exception]

class NoMatchFound(Exception):
    def __init__(self, name: str, path_params: typing.Mapping[str, typing.Any]) -> None: ...

class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

def iscoroutinefunction_or_partial(obj: typing.Any) -> bool: ...

def request_response(func: typing.Callable[[typing.Any], Response]) -> ASGIApp: ...

def websocket_session(func: typing.Callable[[typing.Any], typing.Awaitable[None]]) -> ASGIApp: ...

def get_name(endpoint: typing.Any) -> str: ...

def replace_params(
    path: str, 
    param_convertors: typing.Dict[str, Convertor], 
    path_params: typing.Dict[str, typing.Any]
) -> typing.Tuple[str, typing.Dict[str, typing.Any]]: ...

PARAM_REGEX: re.Pattern[str] ...

def compile_path(path: str) -> typing.Tuple[re.Pattern[str], str, typing.Dict[str, Convertor]]: ...

class BaseRoute:
    def matches(self, scope: Scope) -> typing.Tuple[Match, typing.Dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class Route(BaseRoute):
    path: str
    endpoint: typing.Any
    name: str
    include_in_schema: bool
    app: ASGIApp
    methods: typing.Optional[typing.Set[str]]
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: typing.Dict[str, Convertor]

    def __init__(
        self, 
        path: str, 
        endpoint: typing.Any, 
        *, 
        methods: typing.Optional[typing.Iterable[str]] = None, 
        name: typing.Optional[str] = None, 
        include_in_schema: bool = True, 
        middleware: typing.Optional[typing.Iterable[typing.Tuple[typing.Type[typing.Any], typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]] = None
    ) -> None: ...

    def matches(self, scope: Scope) -> typing.Tuple[Match, typing.Dict[str, typing.Any]]: ...
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
    param_convertors: typing.Dict[str, Convertor]

    def __init__(
        self, 
        path: str, 
        endpoint: typing.Any, 
        *, 
        name: typing.Optional[str] = None, 
        middleware: typing.Optional[typing.Iterable[typing.Tuple[typing.Type[typing.Any], typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]] = None
    ) -> None: ...

    def matches(self, scope: Scope) -> typing.Tuple[Match, typing.Dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class Mount(BaseRoute):
    path: str
    app: ASGIApp
    name: typing.Optional[str]
    path_regex: re.Pattern[str]
    path_format: str
    param_convertors: typing.Dict[str, Convertor]

    def __init__(
        self, 
        path: str, 
        app: typing.Optional[ASGIApp] = None, 
        routes: typing.Optional[typing.Iterable[BaseRoute]] = None, 
        name: typing.Optional[str] = None, 
        *, 
        middleware: typing.Optional[typing.Iterable[typing.Tuple[typing.Type[typing.Any], typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]] = None
    ) -> None: ...

    @property
    def routes(self) -> typing.List[BaseRoute]: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, typing.Dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class Host(BaseRoute):
    host: str
    app: ASGIApp
    name: typing.Optional[str]
    host_regex: re.Pattern[str]
    host_format: str
    param_convertors: typing.Dict[str, Convertor]

    def __init__(self, host: str, app: ASGIApp, name: typing.Optional[str] = None) -> None: ...

    @property
    def routes(self) -> typing.List[BaseRoute]: ...
    def matches(self, scope: Scope) -> typing.Tuple[Match, typing.Dict[str, typing.Any]]: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...

class Router:
    routes: typing.List[BaseRoute]
    redirect_slashes: bool
    default: typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]
    on_startup: typing.List[typing.Callable[[], typing.Any]]
    on_shutdown: typing.List[typing.Callable[[], typing.Any]]
    lifespan_context: typing.Any
    middleware_stack: ASGIApp

    def __init__(
        self, 
        routes: typing.Optional[typing.Iterable[BaseRoute]] = None, 
        redirect_slashes: bool = True, 
        default: typing.Optional[typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]] = None, 
        on_startup: typing.Optional[typing.Iterable[typing.Callable[[], typing.Any]]] = None, 
        on_shutdown: typing.Optional[typing.Iterable[typing.Callable[[], typing.Any]]] = None, 
        lifespan: typing.Optional[typing.Any] = None, 
        *, 
        middleware: typing.Optional[typing.Iterable[typing.Tuple[typing.Type[typing.Any], typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]] = None
    ) -> None: ...

    async def not_found(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def lifespan(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def app(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def mount(self, path: str, app: ASGIApp, name: typing.Optional[str] = None) -> None: ...
    def host(self, host: str, app: ASGIApp, name: typing.Optional[str] = None) -> None: ...
    def add_route(
        self, 
        path: str, 
        endpoint: typing.Any, 
        methods: typing.Optional[typing.Iterable[str]] = None, 
        name: typing.Optional[str] = None, 
        include_in_schema: bool = True
    ) -> None: ...
    def add_websocket_route(self, path: str, endpoint: typing.Any, name: typing.Optional[str] = None) -> None: ...
    def route(
        self, 
        path: str, 
        methods: typing.Optional[typing.Iterable[str]] = None, 
        name: typing.Optional[str] = None, 
        include_in_schema: bool = True
    ) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...
    def websocket_route(self, path: str, name: typing.Optional[str] = None) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...
    def add_event_handler(self, event_type: typing.Literal['startup', 'shutdown'], func: typing.Callable[[], typing.Any]) -> None: ...
    def on_event(self, event_type: typing.Literal['startup', 'shutdown']) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]: ...