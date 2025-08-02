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

if typing.TYPE_CHECKING:
    from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

class NoMatchFound(Exception):
    """
    Raised by `.url_for(name, **path_params)` and `.url_path_for(name, **path_params)`
    if no matching route exists.
    """

    def __init__(self, name: str, path_params: Dict[str, Any]) -> None:
        params = ', '.join(list(path_params.keys()))
        super().__init__(
            f'No route exists for name "{name}" and params "{params}".')


class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


def func_lh241roy(obj: Any) -> bool:
    """
    Correctly determines if an object is a coroutine function,
    including those wrapped in functools.partial objects.
    """
    warnings.warn(
        'iscoroutinefunction_or_partial is deprecated, and will be removed in a future release.'
        , DeprecationWarning)
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj)


def func_u8rp4mi8(func: Callable[..., Response] | Callable[..., Awaitable[Response]]) -> ASGIApp:
    """
    Takes a function or coroutine `func(request) -> response`,
    and returns an ASGI application.
    """
    f: Callable[..., Awaitable[Response]]

    if is_async_callable(func):
        f = func  # type: ignore
    else:
        f = functools.partial(run_in_threadpool, func)  # type: ignore

    async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive, send)

        async def inner_app(scope_inner: Scope, receive_inner: Receive, send_inner: Send) -> None:
            response = await f(request)
            await response(scope_inner, receive_inner, send_inner)
        
        await wrap_app_handling_exceptions(inner_app, request)(scope, receive, send)
    return func_e6293lre


def func_w1pbf9do(func: Callable[[WebSocket], Any] | Callable[[WebSocket], Awaitable[Any]]) -> ASGIApp:
    """
    Takes a coroutine `func(session)`, and returns an ASGI application.
    """

    if is_async_callable(func):
        f = func
    else:
        f = func  # Assuming middleware makes it async

    async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
        session = WebSocket(scope, receive=receive, send=send)

        async def inner_app(scope_inner: Scope, receive_inner: Receive, send_inner: Send) -> None:
            await f(session)
        
        await wrap_app_handling_exceptions(inner_app, session)(scope, receive, send)
    return func_e6293lre


def func_mtudy4p9(endpoint: Any) -> str:
    return getattr(endpoint, '__name__', endpoint.__class__.__name__)


def func_0pue9e4m(path: str, param_convertors: Dict[str, Convertor], path_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    for key, value in list(path_params.items()):
        if '{' + key + '}' in path:
            convertor = param_convertors[key]
            value = convertor.to_string(value)
            path = path.replace('{' + key + '}', value)
            path_params.pop(key)
    return path, path_params


PARAM_REGEX = re.compile('{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}'
    )


def func_mjfkrmqb(path: str) -> Tuple[re.Pattern, str, Dict[str, Convertor]]:
    """
    Given a path string, like: "/{username:str}",
    or a host string, like: "{subdomain}.mydomain.org", return a three-tuple
    of (regex, format, {param_name:convertor}).

    regex:      "/(?P<username>[^/]+)"
    format:     "/{username}"
    convertors: {"username": StringConvertor()}
    """
    is_host = not path.startswith('/')
    path_regex = '^'
    path_format = ''
    duplicated_params: Set[str] = set()
    idx = 0
    param_convertors: Dict[str, Convertor] = {}
    for match in PARAM_REGEX.finditer(path):
        param_name, convertor_type = match.groups('str')
        convertor_type = convertor_type.lstrip(':')
        assert convertor_type in CONVERTOR_TYPES, f"Unknown path convertor '{convertor_type}'"
        convertor = CONVERTOR_TYPES[convertor_type]
        path_regex += re.escape(path[idx:match.start()])
        path_regex += f'(?P<{param_name}>{convertor.regex})'
        path_format += path[idx:match.start()]
        path_format += '{%s}' % param_name
        if param_name in param_convertors:
            duplicated_params.add(param_name)
        param_convertors[param_name] = convertor
        idx = match.end()
    if duplicated_params:
        names = ', '.join(sorted(duplicated_params))
        ending = 's' if len(duplicated_params) > 1 else ''
        raise ValueError(
            f'Duplicated param name{ending} {names} at path {path}')
    if is_host:
        hostname = path[idx:].split(':')[0]
        path_regex += re.escape(hostname) + '$'
    else:
        path_regex += re.escape(path[idx:]) + '$'
    path_format += path[idx:]
    return re.compile(path_regex), path_format, param_convertors


class BaseRoute:

    def func_e6b06el3(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        raise NotImplementedError()

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        raise NotImplementedError()

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        raise NotImplementedError()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        A route may be used in isolation as a stand-alone ASGI app.
        This is a somewhat contrived case, as they'll almost always be used
        within a Router, but could be useful for some tooling and minimal apps.
        """
        match, child_scope = self.func_e6b06el3(scope)
        if match == Match.NONE:
            if scope['type'] == 'http':
                response = PlainTextResponse('Not Found', status_code=404)
                await response(scope, receive, send)
            elif scope['type'] == 'websocket':
                websocket_close = WebSocketClose()
                await websocket_close(scope, receive, send)
            return
        scope.update(child_scope)
        await self.func_zkflwkfi(scope, receive, send)


class Route(BaseRoute):

    def __init__(self, path: str, endpoint: Any, *, methods: Optional[List[str]] = None, name: Optional[str] = None,
        include_in_schema: bool = True, middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path: str = path
        self.endpoint: Any = endpoint
        self.name: str = func_mtudy4p9(endpoint) if name is None else name
        self.include_in_schema: bool = include_in_schema
        endpoint_handler: Any = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(endpoint_handler):
            self.app: ASGIApp = func_u8rp4mi8(endpoint)
            if methods is None:
                methods = ['GET']
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        if methods is None:
            self.methods: Optional[Set[str]] = None
        else:
            self.methods = {method.upper() for method in methods}
            if 'GET' in self.methods:
                self.methods.add('HEAD')
        self.path_regex: re.Pattern
        self.path_format: str
        self.param_convertors: Dict[str, Convertor]
        self.path_regex, self.path_format, self.param_convertors = func_mjfkrmqb(path)

    def func_e6b06el3(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] == 'http':
            route_path = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(
                        value)
                path_params = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'endpoint': self.endpoint, 'path_params': path_params}
                if self.methods and scope['method'] not in self.methods:
                    return Match.PARTIAL, child_scope
                else:
                    return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        seen_params: Set[str] = set(path_params.keys())
        expected_params: Set[str] = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = func_0pue9e4m(self.path_format, self.param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='http')

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods and scope['method'] not in self.methods:
            headers = {'Allow': ', '.join(self.methods)}
            if 'app' in scope:
                raise HTTPException(status_code=405, headers=headers)
            else:
                response = PlainTextResponse('Method Not Allowed',
                    status_code=405, headers=headers)
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Route) and self.path == other.path and 
            self.endpoint == other.endpoint and self.methods == other.methods)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        methods = sorted(self.methods or [])
        path, name = self.path, self.name
        return (
            f'{class_name}(path={path!r}, name={name!r}, methods={methods!r})')


class WebSocketRoute(BaseRoute):

    def __init__(self, path: str, endpoint: Any, *, name: Optional[str] = None, middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path: str = path
        self.endpoint: Any = endpoint
        self.name: str = func_mtudy4p9(endpoint) if name is None else name
        endpoint_handler: Any = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(endpoint_handler):
            self.app: ASGIApp = func_w1pbf9do(endpoint)
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        self.path_regex: re.Pattern
        self.path_format: str
        self.param_convertors: Dict[str, Convertor]
        self.path_regex, self.path_format, self.param_convertors = func_mjfkrmqb(path)

    def func_e6b06el3(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] == 'websocket':
            route_path = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(
                        value)
                path_params = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'endpoint': self.endpoint, 'path_params': path_params}
                return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        seen_params: Set[str] = set(path_params.keys())
        expected_params: Set[str] = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = func_0pue9e4m(self.path_format, self.param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='websocket')

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, WebSocketRoute
            ) and self.path == other.path and self.endpoint == other.endpoint

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(path={self.path!r}, name={self.name!r})'
            )


class Mount(BaseRoute):

    def __init__(self, path: str, app: Optional[ASGIApp] = None, routes: Optional[List[BaseRoute]] = None, name: Optional[str] = None, *,
        middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None) -> None:
        assert path == '' or path.startswith('/'), "Routed paths must start with '/'"
        assert app is not None or routes is not None, "Either 'app=...', or 'routes=' must be specified"
        self.path: str = path.rstrip('/')
        if app is not None:
            self._base_app: ASGIApp = app
        else:
            self._base_app = Router(routes=routes)  # type: ignore
        self.app: ASGIApp = self._base_app
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        self.name: Optional[str] = name
        self.path_regex: re.Pattern
        self.path_format: str
        self.param_convertors: Dict[str, Convertor]
        self.path_regex, self.path_format, self.param_convertors = func_mjfkrmqb(self.path + '/{path:path}')

    @property
    def func_m9rtai2w(self) -> List[BaseRoute]:
        return getattr(self._base_app, 'routes', [])

    def func_e6b06el3(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] in ('http', 'websocket'):
            root_path = scope.get('root_path', '')
            route_path = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(
                        value)
                remaining_path = '/' + matched_params.pop('path')
                matched_path = route_path[:-len(remaining_path)]
                path_params = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {
                    'path_params': path_params,
                    'app_root_path': scope.get('app_root_path', root_path),
                    'root_path': root_path + matched_path,
                    'endpoint': self.app
                }
                return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        if (self.name is not None and name == self.name and 'path' in path_params):
            path_params['path'] = path_params['path'].lstrip('/')
            path, remaining_params = func_0pue9e4m(self.path_format, self.param_convertors, path_params)
            if not remaining_params:
                return URLPath(path=path)
        elif self.name is None or name.startswith(self.name + ':'):
            if self.name is None:
                remaining_name = name
            else:
                remaining_name = name[len(self.name) + 1:]
            path_kwarg = path_params.get('path')
            path_params['path'] = ''
            path_prefix, remaining_params = func_0pue9e4m(self.path_format, self.param_convertors, path_params)
            if path_kwarg is not None:
                remaining_params['path'] = path_kwarg
            for route in (self.routes or []):
                try:
                    url = route.func_ey72marj(remaining_name, **remaining_params)
                    return URLPath(path=path_prefix.rstrip('/') + str(url),
                        protocol=url.protocol)
                except NoMatchFound:
                    pass
        raise NoMatchFound(name, path_params)

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mount
            ) and self.path == other.path and self.app == other.app

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        name = self.name or ''
        return (
            f'{class_name}(path={self.path!r}, name={name!r}, app={self.app!r})'
            )


class Host(BaseRoute):

    def __init__(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None:
        assert not host.startswith('/'), "Host must not start with '/'"
        self.host: str = host
        self.app: ASGIApp = app
        self.name: Optional[str] = name
        self.host_regex: re.Pattern
        self.host_format: str
        self.param_convertors: Dict[str, Convertor]
        self.host_regex, self.host_format, self.param_convertors = func_mjfkrmqb(host)

    @property
    def func_m9rtai2w(self) -> List[BaseRoute]:
        return getattr(self.app, 'routes', [])

    def func_e6b06el3(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] in ('http', 'websocket'):
            headers = Headers(scope=scope)
            host = headers.get('host', '').split(':')[0]
            match = self.host_regex.match(host)
            if match:
                matched_params = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(
                        value)
                path_params = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'path_params': path_params, 'endpoint': self.app}
                return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        if (self.name is not None and name == self.name and 'path' in path_params):
            path = path_params.pop('path')
            host, remaining_params = func_0pue9e4m(self.host_format, self.param_convertors, path_params)
            if not remaining_params:
                return URLPath(path=path, host=host)
        elif self.name is None or name.startswith(self.name + ':'):
            if self.name is None:
                remaining_name = name
            else:
                remaining_name = name[len(self.name) + 1:]
            host, remaining_params = func_0pue9e4m(self.host_format, self.param_convertors, path_params)
            for route in (self.routes or []):
                try:
                    url = route.func_ey72marj(remaining_name, **remaining_params)
                    return URLPath(path=str(url), protocol=url.protocol,
                        host=host)
                except NoMatchFound:
                    pass
        raise NoMatchFound(name, path_params)

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Host
            ) and self.host == other.host and self.app == other.app

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        name = self.name or ''
        return (
            f'{class_name}(host={self.host!r}, name={name!r}, app={self.app!r})'
            )


_T = typing.TypeVar('_T')


class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):

    def __init__(self, cm: contextlib.AbstractContextManager[_T]) -> None:
        self._cm: contextlib.AbstractContextManager[_T] = cm

    async def __aenter__(self) -> _T:
        return self._cm.__enter__()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[types.TracebackType]) -> Optional[bool]:
        return self._cm.__exit__(exc_type, exc_value, traceback)


def func_wiy024s2(lifespan_context: Callable[[ASGIApp], contextlib.AbstractContextManager[Any]]) -> Callable[[ASGIApp], _AsyncLiftContextManager[Any]]:
    cmgr = contextlib.contextmanager(lifespan_context)

    @functools.wraps(cmgr)
    def func_6i1vxgix(app: ASGIApp) -> _AsyncLiftContextManager[Any]:
        return _AsyncLiftContextManager(cmgr(app))
    return func_6i1vxgix


class _DefaultLifespan:

    def __init__(self, router: Router) -> None:
        self._router: Router = router

    async def __aenter__(self) -> None:
        await self._router.func_kbb49wrj()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[types.TracebackType]) -> None:
        await self._router.func_wcoh934p()

    def __call__(self, app: ASGIApp) -> _DefaultLifespan:
        return self


class Router:

    def __init__(self, routes: Optional[List[BaseRoute]] = None, redirect_slashes: bool = True, default: Optional[Callable[[Scope, Receive, Send], Awaitable[None]]] = None,
        on_startup: Optional[List[Callable[[], Any]]] = None, on_shutdown: Optional[List[Callable[[], Any]]] = None, lifespan: Optional[Callable[[ASGIApp], contextlib.AsyncContextManager[Any]]] = None, *, middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None) -> None:
        self.routes: List[BaseRoute] = [] if routes is None else list(routes)
        self.redirect_slashes: bool = redirect_slashes
        self.default: Callable[[Scope, Receive, Send], Awaitable[None]] = self.func_1n65mjse if default is None else default
        self.on_startup: List[Callable[[], Any]] = [] if on_startup is None else list(on_startup)
        self.on_shutdown: List[Callable[[], Any]] = [] if on_shutdown is None else list(on_shutdown)
        if on_startup or on_shutdown:
            warnings.warn(
                'The on_startup and on_shutdown parameters are deprecated, and they will be removed on version 1.0. Use the lifespan parameter instead. See more about it on https://www.starlette.io/lifespan/.'
                , DeprecationWarning)
            if lifespan:
                warnings.warn(
                    'The `lifespan` parameter cannot be used with `on_startup` or `on_shutdown`. Both `on_startup` and `on_shutdown` will be ignored.'
                    )
        if lifespan is None:
            self.lifespan_context: Callable[[ASGIApp], contextlib.AsyncContextManager[Any]] = _DefaultLifespan(self)
        elif inspect.isasyncgenfunction(lifespan):
            warnings.warn(
                'async generator function lifespans are deprecated, use an @contextlib.asynccontextmanager function instead'
                , DeprecationWarning)
            self.lifespan_context = asynccontextmanager(lifespan)
        elif inspect.isgeneratorfunction(lifespan):
            warnings.warn(
                'generator function lifespans are deprecated, use an @contextlib.asynccontextmanager function instead'
                , DeprecationWarning)
            self.lifespan_context = func_wiy024s2(lifespan)
        else:
            self.lifespan_context = lifespan
        self.middleware_stack: ASGIApp = self.func_e6293lre
        if middleware:
            for cls, args, kwargs in reversed(middleware):
                self.middleware_stack = cls(self.middleware_stack, *args,
                    **kwargs)

    async def func_1n65mjse(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'websocket':
            websocket_close = WebSocketClose()
            await websocket_close(scope, receive, send)
            return
        if 'app' in scope:
            raise HTTPException(status_code=404)
        else:
            response = PlainTextResponse('Not Found', status_code=404)
        await response(scope, receive, send)

    def func_ey72marj(self, name: str, /, **path_params: Any) -> URLPath:
        for route in self.routes:
            try:
                return route.func_ey72marj(name, **path_params)
            except NoMatchFound:
                pass
        raise NoMatchFound(name, path_params)

    async def func_kbb49wrj(self) -> None:
        """
        Run any `.on_startup` event handlers.
        """
        for handler in self.on_startup:
            if is_async_callable(handler):
                await handler()
            else:
                handler()

    async def func_wcoh934p(self) -> None:
        """
        Run any `.on_shutdown` event handlers.
        """
        for handler in self.on_shutdown:
            if is_async_callable(handler):
                await handler()
            else:
                handler()

    async def func_4egab4xp(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Handle ASGI lifespan messages, which allows us to manage application
        startup and shutdown events.
        """
        started: bool = False
        app = scope.get('app')
        await receive()
        try:
            async with self.lifespan_context(app) as maybe_state:
                if maybe_state is not None:
                    if 'state' not in scope:
                        raise RuntimeError(
                            'The server does not support "state" in the lifespan scope.'
                            )
                    scope['state'].update(maybe_state)
                await send({'type': 'lifespan.startup.complete'})
                started = True
                await receive()
        except BaseException:
            exc_text = traceback.format_exc()
            if started:
                await send({'type': 'lifespan.shutdown.failed', 'message':
                    exc_text})
            else:
                await send({'type': 'lifespan.startup.failed', 'message':
                    exc_text})
            raise
        else:
            await send({'type': 'lifespan.shutdown.complete'})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        The main entry point to the Router class.
        """
        await self.middleware_stack(scope, receive, send)

    async def func_e6293lre(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert scope['type'] in ('http', 'websocket', 'lifespan')
        if 'router' not in scope:
            scope['router'] = self
        if scope['type'] == 'lifespan':
            await self.func_4egab4xp(scope, receive, send)
            return
        partial: Optional[BaseRoute] = None
        partial_scope: Dict[str, Any] = {}
        for route in self.routes:
            match, child_scope = route.func_e6b06el3(scope)
            if match == Match.FULL:
                scope.update(child_scope)
                await route.func_zkflwkfi(scope, receive, send)
                return
            elif match == Match.PARTIAL and partial is None:
                partial = route
                partial_scope = child_scope
        if partial is not None:
            scope.update(partial_scope)
            await partial.func_zkflwkfi(scope, receive, send)
            return
        route_path = get_route_path(scope)
        if scope['type'] == 'http' and self.redirect_slashes and route_path != '/':
            redirect_scope: Scope = dict(scope)
            if route_path.endswith('/'):
                redirect_scope['path'] = redirect_scope['path'].rstrip('/')
            else:
                redirect_scope['path'] = redirect_scope['path'] + '/'
            for route in self.routes:
                match, child_scope = route.func_e6b06el3(redirect_scope)
                if match != Match.NONE:
                    redirect_url = URL(scope=redirect_scope)
                    response = RedirectResponse(url=str(redirect_url))
                    await response(scope, receive, send)
                    return
        await self.default(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Router) and self.routes == other.routes

    def func_1v6oo7zb(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None:
        route = Mount(path, app=app, name=name)
        self.routes.append(route)

    def func_zeqkj69x(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None:
        route = Host(host, app=app, name=name)
        self.routes.append(route)

    def func_kigfsltn(self, path: str, endpoint: Any, methods: Optional[List[str]] = None, name: Optional[str] = None,
        include_in_schema: bool = True) -> None:
        route = Route(path, endpoint=endpoint, methods=methods, name=name,
            include_in_schema=include_in_schema)
        self.routes.append(route)

    def func_kjby33wr(self, path: str, endpoint: Any, name: Optional[str] = None) -> None:
        route = WebSocketRoute(path, endpoint=endpoint, name=name)
        self.routes.append(route)

    def func_3d7p8gd1(self, path: str, methods: Optional[List[str]] = None, name: Optional[str] = None,
        include_in_schema: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        We no longer document this decorator style API, and its usage is discouraged.
        Instead you should use the following approach:

        >>> routes = [Route(path, endpoint=...), ...]
        >>> app = Starlette(routes=routes)
        """
        warnings.warn(
            'The `route` decorator is deprecated, and will be removed in version 1.0.0.Refer to https://www.starlette.io/routing/#http-routing for the recommended approach.'
            , DeprecationWarning)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_route(path, func, methods=methods, name=name,
                include_in_schema=include_in_schema)
            return func
        return decorator

    def func_vn3cfcla(self, path: str, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        We no longer document this decorator style API, and its usage is discouraged.
        Instead you should use the following approach:

        >>> routes = [WebSocketRoute(path, endpoint=...), ...]
        >>> app = Starlette(routes=routes)
        """
        warnings.warn(
            'The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.'
            , DeprecationWarning)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_websocket_route(path, func, name=name)
            return func
        return decorator

    def func_127s765m(self, event_type: str, func: Callable[..., Any]) -> None:
        assert event_type in ('startup', 'shutdown')
        if event_type == 'startup':
            self.on_startup.append(func)
        else:
            self.on_shutdown.append(func)

    def func_gftusa2u(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            'The `on_event` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/lifespan/ for recommended approach.'
            , DeprecationWarning)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_event_handler(event_type, func)
            return func
        return decorator
