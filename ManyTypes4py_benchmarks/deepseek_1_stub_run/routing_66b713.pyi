```python
from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar, overload
from typing import AsyncContextManager, AsyncGenerator, AsyncIterable, Awaitable
from typing import Union, Optional, Dict, List, Set, Tuple, Pattern
from enum import Enum
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket

class NoMatchFound(Exception):
    def __init__(self, name: str, path_params: Dict[str, Any]) -> None: ...

class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

def iscoroutinefunction_or_partial(obj: Any) -> bool: ...

def request_response(func: Callable[..., Any]) -> ASGIApp: ...

def websocket_session(func: Callable[..., Any]) -> ASGIApp: ...

def get_name(endpoint: Any) -> str: ...

def replace_params(
    path: str, 
    param_convertors: Dict[str, Any], 
    path_params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]: ...

PARAM_REGEX: Pattern[str]

def compile_path(path: str) -> Tuple[Pattern[str], str, Dict[str, Any]]: ...

class BaseRoute:
    def matches(self, scope: Dict[str, Any]) -> Tuple[Match, Dict[str, Any]]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def handle(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...

class Route(BaseRoute):
    path: str
    endpoint: Any
    name: str
    include_in_schema: bool
    app: ASGIApp
    methods: Optional[Set[str]]
    path_regex: Pattern[str]
    path_format: str
    param_convertors: Dict[str, Any]
    
    def __init__(
        self, 
        path: str, 
        endpoint: Any, 
        *, 
        methods: Optional[Iterable[str]] = None, 
        name: Optional[str] = None, 
        include_in_schema: bool = True, 
        middleware: Optional[Sequence[Middleware]] = None
    ) -> None: ...
    
    def matches(self, scope: Dict[str, Any]) -> Tuple[Match, Dict[str, Any]]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def handle(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...

class WebSocketRoute(BaseRoute):
    path: str
    endpoint: Any
    name: str
    app: ASGIApp
    path_regex: Pattern[str]
    path_format: str
    param_convertors: Dict[str, Any]
    
    def __init__(
        self, 
        path: str, 
        endpoint: Any, 
        *, 
        name: Optional[str] = None, 
        middleware: Optional[Sequence[Middleware]] = None
    ) -> None: ...
    
    def matches(self, scope: Dict[str, Any]) -> Tuple[Match, Dict[str, Any]]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def handle(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...

class Mount(BaseRoute):
    path: str
    app: ASGIApp
    name: Optional[str]
    path_regex: Pattern[str]
    path_format: str
    param_convertors: Dict[str, Any]
    
    @property
    def routes(self) -> List[Any]: ...
    
    def __init__(
        self, 
        path: str, 
        app: Optional[ASGIApp] = None, 
        routes: Optional[Iterable[Any]] = None, 
        name: Optional[str] = None, 
        *, 
        middleware: Optional[Sequence[Middleware]] = None
    ) -> None: ...
    
    def matches(self, scope: Dict[str, Any]) -> Tuple[Match, Dict[str, Any]]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def handle(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...

class Host(BaseRoute):
    host: str
    app: ASGIApp
    name: Optional[str]
    host_regex: Pattern[str]
    host_format: str
    param_convertors: Dict[str, Any]
    
    @property
    def routes(self) -> List[Any]: ...
    
    def __init__(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None: ...
    
    def matches(self, scope: Dict[str, Any]) -> Tuple[Match, Dict[str, Any]]: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def handle(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...

_T = TypeVar("_T")

class _AsyncLiftContextManager(AsyncContextManager[_T]):
    def __init__(self, cm: Any) -> None: ...
    async def __aenter__(self) -> _T: ...
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any: ...

def _wrap_gen_lifespan_context(lifespan_context: Any) -> Any: ...

class _DefaultLifespan:
    def __init__(self, router: Any) -> None: ...
    async def __aenter__(self) -> Any: ...
    async def __aexit__(self, *exc_info: Any) -> Any: ...
    def __call__(self, app: Any) -> _DefaultLifespan: ...

class Router:
    routes: List[Any]
    redirect_slashes: bool
    default: Callable[..., Any]
    on_startup: List[Any]
    on_shutdown: List[Any]
    lifespan_context: Any
    middleware_stack: ASGIApp
    
    def __init__(
        self, 
        routes: Optional[Iterable[Any]] = None, 
        redirect_slashes: bool = True, 
        default: Optional[Callable[..., Any]] = None, 
        on_startup: Optional[Iterable[Any]] = None, 
        on_shutdown: Optional[Iterable[Any]] = None, 
        lifespan: Optional[Any] = None, 
        *, 
        middleware: Optional[Sequence[Middleware]] = None
    ) -> None: ...
    
    async def not_found(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def url_path_for(self, name: str, /, **path_params: Any) -> Any: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def lifespan(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    async def __call__(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    async def app(self, scope: Dict[str, Any], receive: Receive, send: Send) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def mount(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None: ...
    def host(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None: ...
    def add_route(
        self, 
        path: str, 
        endpoint: Any, 
        methods: Optional[Iterable[str]] = None, 
        name: Optional[str] = None, 
        include_in_schema: bool = True
    ) -> None: ...
    def add_websocket_route(self, path: str, endpoint: Any, name: Optional[str] = None) -> None: ...
    
    @overload
    def route(
        self, 
        path: str, 
        methods: Optional[Iterable[str]] = None, 
        name: Optional[str] = None, 
        include_in_schema: bool = True
    ) -> Callable[[Any], Any]: ...
    
    def route(self, *args: Any, **kwargs: Any) -> Any: ...
    
    @overload
    def websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[Any], Any]: ...
    
    def websocket_route(self, *args: Any, **kwargs: Any) -> Any: ...
    
    def add_event_handler(self, event_type: str, func: Any) -> None: ...
    
    @overload
    def on_event(self, event_type: str) -> Callable[[Any], Any]: ...
    
    def on_event(self, *args: Any, **kwargs: Any) -> Any: ...
```