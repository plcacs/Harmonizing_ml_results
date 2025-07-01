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
from typing import (
    Any,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from types import TracebackType

class NoMatchFound(Exception):
    """
    Raised by `.url_for(name, **path_params)` and `.url_path_for(name, **path_params)`
    if no matching route exists.
    """

    def __init__(self, name: str, path_params: Dict[str, Any]) -> None:
        params = ', '.join(list(path_params.keys()))
        super().__init__(f'No route exists for name "{name}" and params "{params}".')


class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


def iscoroutinefunction_or_partial(obj: Any) -> bool:
    """
    Correctly determines if an object is a coroutine function,
    including those wrapped in functools.partial objects.
    """
    warnings.warn(
        'iscoroutinefunction_or_partial is deprecated, and will be removed in a future release.',
        DeprecationWarning,
    )
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj)


def request_response(func: Callable[..., Response]) -> ASGIApp:
    """
    Takes a function or coroutine `func(request) -> response`,
    and returns an ASGI application.
    """
    if is_async_callable(func):
        f: Callable[[Request], Awaitable[Response]] = func  # type: ignore
    else:
        f = functools.partial(run_in_threadpool, func)  # type: ignore

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive, send)

        async def inner_app(scope: Scope, receive: Receive, send: Send) -> None:
            response = await f(request)
            await response(scope, receive, send)

        await wrap_app_handling_exceptions(inner_app, request)(scope, receive, send)

    return app


def websocket_session(func: Callable[..., Awaitable[None]]) -> ASGIApp:
    """
    Takes a coroutine `func(session)`, and returns an ASGI application.
    """

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        session = WebSocket(scope, receive=receive, send=send)

        async def inner_app(scope: Scope, receive: Receive, send: Send) -> None:
            await func(session)

        await wrap_app_handling_exceptions(inner_app, session)(scope, receive, send)

    return app


def get_name(endpoint: Any) -> str:
    return getattr(endpoint, '__name__', endpoint.__class__.__name__)


def replace_params(
    path: str, param_convertors: Dict[str, Convertor], path_params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    for key, value in list(path_params.items()):
        if '{' + key + '}' in path:
            convertor = param_convertors[key]
            value = convertor.to_string(value)
            path = path.replace('{' + key + '}', value)
            path_params.pop(key)
    return (path, path_params)


PARAM_REGEX: Pattern = re.compile(r'{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}')


def compile_path(path: str) -> Tuple[Pattern, str, Dict[str, Convertor]]:
    """
    Given a path string, like: "/{username:str}",
    or a host string, like: "{subdomain}.mydomain.org", return a three-tuple
    of (regex, format, {param_name:convertor}).

    regex:      "/(?P<username>[^/]+)"
    format:     "/{username}"
    convertors: {"username": StringConvertor()}
    """
    is_host: bool = not path.startswith('/')
    path_regex: str = '^'
    path_format: str = ''
    duplicated_params: Set[str] = set()
    idx: int = 0
    param_convertors: Dict[str, Convertor] = {}
    for match in PARAM_REGEX.finditer(path):
        param_name, convertor_type = match.groups('str')
        convertor_type = convertor_type.lstrip(':')
        assert (
            convertor_type in CONVERTOR_TYPES
        ), f"Unknown path convertor '{convertor_type}'"
        convertor: Convertor = CONVERTOR_TYPES[convertor_type]
        path_regex += re.escape(path[idx : match.start()])
        path_regex += f'(?P<{param_name}>{convertor.regex})'
        path_format += path[idx : match.start()]
        path_format += f'{{{param_name}}}'
        if param_name in param_convertors:
            duplicated_params.add(param_name)
        param_convertors[param_name] = convertor
        idx = match.end()
    if duplicated_params:
        names = ', '.join(sorted(duplicated_params))
        ending = 's' if len(duplicated_params) > 1 else ''
        raise ValueError(f'Duplicated param name{ending} {names} at path {path}')
    if is_host:
        hostname = path[idx:].split(':')[0]
        path_regex += re.escape(hostname) + '$'
    else:
        path_regex += re.escape(path[idx:]) + '$'
    path_format += path[idx:]
    return (re.compile(path_regex), path_format, param_convertors)


class BaseRoute:

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        raise NotImplementedError()

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        raise NotImplementedError()

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        raise NotImplementedError()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        A route may be used in isolation as a stand-alone ASGI app.
        This is a somewhat contrived case, as they'll almost always be used
        within a Router, but could be useful for some tooling and minimal apps.
        """
        match, child_scope = self.matches(scope)
        if match == Match.NONE:
            if scope['type'] == 'http':
                response = PlainTextResponse('Not Found', status_code=404)
                await response(scope, receive, send)
            elif scope['type'] == 'websocket':
                websocket_close = WebSocketClose()
                await websocket_close(scope, receive, send)
            return
        scope.update(child_scope)
        await self.handle(scope, receive, send)


class Route(BaseRoute):

    def __init__(
        self,
        path: str,
        endpoint: ASGIApp,
        *,
        methods: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
        middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None,
    ) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path: str = path
        self.endpoint: ASGIApp = endpoint
        self.name: str = get_name(endpoint) if name is None else name
        self.include_in_schema: bool = include_in_schema
        endpoint_handler = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(endpoint_handler):
            self.app: ASGIApp = request_response(endpoint)
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
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] == 'http':
            route_path: str = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params: Dict[str, Any] = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(value)
                path_params: Dict[str, Any] = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'endpoint': self.endpoint, 'path_params': path_params}
                if self.methods and scope['method'] not in self.methods:
                    return (Match.PARTIAL, child_scope)
                else:
                    return (Match.FULL, child_scope)
        return (Match.NONE, {})

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        seen_params: Set[str] = set(path_params.keys())
        expected_params: Set[str] = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = replace_params(self.path_format, self.param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='http')

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods and scope['method'] not in self.methods:
            headers: Dict[str, str] = {'Allow': ', '.join(self.methods)}
            if 'app' in scope:
                raise HTTPException(status_code=405, headers=headers)
            else:
                response: Response = PlainTextResponse('Method Not Allowed', status_code=405, headers=headers)
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Route) and self.path == other.path and (self.endpoint == other.endpoint) and (self.methods == other.methods)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        methods = sorted(self.methods or [])
        path, name = (self.path, self.name)
        return f'{class_name}(path={path!r}, name={name!r}, methods={methods!r})'


class WebSocketRoute(BaseRoute):

    def __init__(
        self,
        path: str,
        endpoint: ASGIApp,
        *,
        name: Optional[str] = None,
        middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None,
    ) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path: str = path
        self.endpoint: ASGIApp = endpoint
        self.name: str = get_name(endpoint) if name is None else name
        endpoint_handler = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(endpoint_handler):
            self.app: ASGIApp = websocket_session(endpoint)
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] == 'websocket':
            route_path: str = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params: Dict[str, Any] = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(value)
                path_params: Dict[str, Any] = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'endpoint': self.endpoint, 'path_params': path_params}
                return (Match.FULL, child_scope)
        return (Match.NONE, {})

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        seen_params: Set[str] = set(path_params.keys())
        expected_params: Set[str] = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = replace_params(self.path_format, self.param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='websocket')

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, WebSocketRoute) and self.path == other.path and (self.endpoint == other.endpoint)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(path={self.path!r}, name={self.name!r})'


class Mount(BaseRoute):

    def __init__(
        self,
        path: str,
        app: Optional[ASGIApp] = None,
        routes: Optional[List[BaseRoute]] = None,
        name: Optional[str] = None,
        *,
        middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None,
    ) -> None:
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
        self.path_regex, self.path_format, self.param_convertors = compile_path(self.path + '/{path:path}')

    @property
    def routes(self) -> List[BaseRoute]:
        return getattr(self._base_app, 'routes', [])

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] in ('http', 'websocket'):
            root_path: str = scope.get('root_path', '')
            route_path: str = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params: Dict[str, Any] = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(value)
                remaining_path: str = '/' + matched_params.pop('path')
                matched_path: str = route_path[:-len(remaining_path)]
                path_params: Dict[str, Any] = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {
                    'path_params': path_params,
                    'app_root_path': scope.get('app_root_path', root_path),
                    'root_path': root_path + matched_path,
                    'endpoint': self.app,
                }
                return (Match.FULL, child_scope)
        return (Match.NONE, {})

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        if self.name is not None and name == self.name and ('path' in path_params):
            path_params['path'] = path_params['path'].lstrip('/')
            path, remaining_params = replace_params(self.path_format, self.param_convertors, path_params)
            if not remaining_params:
                return URLPath(path=path)
        elif self.name is None or name.startswith(self.name + ':'):
            if self.name is None:
                remaining_name: str = name
            else:
                remaining_name = name[len(self.name) + 1 :]
            path_kwarg: Optional[str] = path_params.get('path')
            path_params['path'] = ''
            path_prefix, remaining_params = replace_params(self.path_format, self.param_convertors, path_params)
            if path_kwarg is not None:
                remaining_params['path'] = path_kwarg
            for route in self.routes or []:
                try:
                    url: URLPath = route.url_path_for(remaining_name, **remaining_params)
                    return URLPath(path=path_prefix.rstrip('/') + str(url), protocol=url.protocol)
                except NoMatchFound:
                    pass
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mount) and self.path == other.path and (self.app == other.app)

    def __repr__(self) -> str:
        class_name: str = self.__class__.__name__
        name: str = self.name or ''
        return f'{class_name}(path={self.path!r}, name={name!r}, app={self.app!r})'


class Host(BaseRoute):

    def __init__(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None:
        assert not host.startswith('/'), "Host must not start with '/'"
        self.host: str = host
        self.app: ASGIApp = app
        self.name: Optional[str] = name
        self.host_regex, self.host_format, self.param_convertors = compile_path(host)

    @property
    def routes(self) -> List[BaseRoute]:
        return getattr(self.app, 'routes', [])

    def matches(self, scope: Scope) -> Tuple[Match, Dict[str, Any]]:
        if scope['type'] in ('http', 'websocket'):
            headers: Headers = Headers(scope=scope)
            host: str = headers.get('host', '').split(':')[0]
            match = self.host_regex.match(host)
            if match:
                matched_params: Dict[str, Any] = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(value)
                path_params: Dict[str, Any] = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope: Dict[str, Any] = {'path_params': path_params, 'endpoint': self.app}
                return (Match.FULL, child_scope)
        return (Match.NONE, {})

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        if self.name is not None and name == self.name and ('path' in path_params):
            path: str = path_params.pop('path')
            host, remaining_params = replace_params(self.host_format, self.param_convertors, path_params)
            if not remaining_params:
                return URLPath(path=path, host=host)
        elif self.name is None or name.startswith(self.name + ':'):
            if self.name is None:
                remaining_name: str = name
            else:
                remaining_name = name[len(self.name) + 1 :]
            host, remaining_params = replace_params(self.host_format, self.param_convertors, path_params)
            for route in self.routes or []:
                try:
                    url: URLPath = route.url_path_for(remaining_name, **remaining_params)
                    return URLPath(path=str(url), protocol=url.protocol, host=host)
                except NoMatchFound:
                    pass
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Host) and self.host == other.host and (self.app == other.app)

    def __repr__(self) -> str:
        class_name: str = self.__class__.__name__
        name: str = self.name or ''
        return f'{class_name}(host={self.host!r}, name={name!r}, app={self.app!r})'


_T = TypeVar('_T')


class _AsyncLiftContextManager(typing.AsyncContextManager[_T]):

    def __init__(self, cm: ContextManager[_T]) -> None:
        self._cm = cm

    async def __aenter__(self) -> _T:
        return self._cm.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self._cm.__exit__(exc_type, exc_value, traceback)


def _wrap_gen_lifespan_context(
    lifespan_context: Callable[[ASGIApp], ContextManager[Any]]
) -> Callable[[ASGIApp], typing.AsyncContextManager[Any]]:
    cmgr = contextlib.contextmanager(lifespan_context)

    @functools.wraps(cmgr)
    def wrapper(app: ASGIApp) -> typing.AsyncContextManager[Any]:
        return _AsyncLiftContextManager(cmgr(app))

    return wrapper


class _DefaultLifespan:

    def __init__(self, router: Router) -> None:
        self._router = router

    async def __aenter__(self) -> None:
        await self._router.startup()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self._router.shutdown()

    def __call__(self, app: ASGIApp) -> _DefaultLifespan:
        return self


class Router:

    def __init__(
        self,
        routes: Optional[Iterable[BaseRoute]] = None,
        redirect_slashes: bool = True,
        default: Optional[ASGIApp] = None,
        on_startup: Optional[List[Callable[[], Any]]] = None,
        on_shutdown: Optional[List[Callable[[], Any]]] = None,
        lifespan: Optional[Callable[[ASGIApp], typing.AsyncContextManager[Any]]] = None,
        *,
        middleware: Optional[List[Tuple[Type[Middleware], Tuple[Any, ...], Dict[str, Any]]]] = None,
    ) -> None:
        self.routes: List[BaseRoute] = [] if routes is None else list(routes)
        self.redirect_slashes: bool = redirect_slashes
        self.default: ASGIApp = self.not_found if default is None else default
        self.on_startup: List[Callable[[], Any]] = [] if on_startup is None else list(on_startup)
        self.on_shutdown: List[Callable[[], Any]] = [] if on_shutdown is None else list(on_shutdown)
        if on_startup or on_shutdown:
            warnings.warn(
                'The on_startup and on_shutdown parameters are deprecated, and they will be removed on version 1.0. Use the lifespan parameter instead. See more about it on https://www.starlette.io/lifespan/.',
                DeprecationWarning,
            )
            if lifespan:
                warnings.warn(
                    'The `lifespan` parameter cannot be used with `on_startup` or `on_shutdown`. Both `on_startup` and `on_shutdown` will be ignored.',
                    DeprecationWarning,
                )
        if lifespan is None:
            self.lifespan_context: typing.AsyncContextManager[Any] = _DefaultLifespan(self)
        elif inspect.isasyncgenfunction(lifespan):
            warnings.warn(
                'async generator function lifespans are deprecated, use an @contextlib.asynccontextmanager function instead',
                DeprecationWarning,
            )
            self.lifespan_context = asynccontextmanager(lifespan)
        elif inspect.isgeneratorfunction(lifespan):
            warnings.warn(
                'generator function lifespans are deprecated, use an @contextlib.asynccontextmanager function instead',
                DeprecationWarning,
            )
            self.lifespan_context = _wrap_gen_lifespan_context(lifespan)
        else:
            self.lifespan_context = lifespan
        self.middleware_stack: ASGIApp = self.app
        if middleware:
            for cls, args, kwargs in reversed(middleware):
                self.middleware_stack = cls(self.middleware_stack, *args, **kwargs)

    async def not_found(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'websocket':
            websocket_close = WebSocketClose()
            await websocket_close(scope, receive, send)
            return
        if 'app' in scope:
            raise HTTPException(status_code=404)
        else:
            response: Response = PlainTextResponse('Not Found', status_code=404)
        await response(scope, receive, send)

    def url_path_for(self, name: str, /, **path_params: Any) -> URLPath:
        for route in self.routes:
            try:
                return route.url_path_for(name, **path_params)
            except NoMatchFound:
                pass
        raise NoMatchFound(name, path_params)

    async def startup(self) -> None:
        """
        Run any `.on_startup` event handlers.
        """
        for handler in self.on_startup:
            if is_async_callable(handler):
                await handler()
            else:
                handler()

    async def shutdown(self) -> None:
        """
        Run any `.on_shutdown` event handlers.
        """
        for handler in self.on_shutdown:
            if is_async_callable(handler):
                await handler()
            else:
                handler()

    async def lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Handle ASGI lifespan messages, which allows us to manage application
        startup and shutdown events.
        """
        started: bool = False
        app: Any = scope.get('app')
        await receive()
        try:
            async with self.lifespan_context(app) as maybe_state:
                if maybe_state is not None:
                    if 'state' not in scope:
                        raise RuntimeError('The server does not support "state" in the lifespan scope.')
                    scope['state'].update(maybe_state)
                await send({'type': 'lifespan.startup.complete'})
                started = True
                await receive()
        except BaseException:
            exc_text: str = traceback.format_exc()
            if started:
                await send({'type': 'lifespan.shutdown.failed', 'message': exc_text})
            else:
                await send({'type': 'lifespan.startup.failed', 'message': exc_text})
            raise
        else:
            await send({'type': 'lifespan.shutdown.complete'})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        The main entry point to the Router class.
        """
        await self.middleware_stack(scope, receive, send)

    async def app(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert scope['type'] in ('http', 'websocket', 'lifespan')
        if 'router' not in scope:
            scope['router'] = self
        if scope['type'] == 'lifespan':
            await self.lifespan(scope, receive, send)
            return
        partial: Optional[BaseRoute] = None
        partial_scope: Dict[str, Any] = {}
        for route in self.routes:
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                scope.update(child_scope)
                await route.handle(scope, receive, send)
                return
            elif match == Match.PARTIAL and partial is None:
                partial = route
                partial_scope = child_scope
        if partial is not None:
            scope.update(partial_scope)
            await partial.handle(scope, receive, send)
            return
        route_path: str = get_route_path(scope)
        if scope['type'] == 'http' and self.redirect_slashes and (route_path != '/'):
            redirect_scope: Scope = dict(scope)
            if route_path.endswith('/'):
                redirect_scope['path'] = redirect_scope['path'].rstrip('/')
            else:
                redirect_scope['path'] = redirect_scope['path'] + '/'
            for route in self.routes:
                match, child_scope = route.matches(redirect_scope)
                if match != Match.NONE:
                    redirect_url: URL = URL(scope=redirect_scope)
                    response: Response = RedirectResponse(url=str(redirect_url))
                    await response(scope, receive, send)
                    return
        await self.default(scope, receive, send)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Router) and self.routes == other.routes

    def mount(self, path: str, app: ASGIApp, name: Optional[str] = None) -> None:
        route = Mount(path, app=app, name=name)
        self.routes.append(route)

    def host(self, host: str, app: ASGIApp, name: Optional[str] = None) -> None:
        route = Host(host, app=app, name=name)
        self.routes.append(route)

    def add_route(
        self,
        path: str,
        endpoint: ASGIApp,
        methods: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> None:
        route = Route(path, endpoint=endpoint, methods=methods, name=name, include_in_schema=include_in_schema)
        self.routes.append(route)

    def add_websocket_route(
        self,
        path: str,
        endpoint: ASGIApp,
        name: Optional[str] = None,
    ) -> None:
        route = WebSocketRoute(path, endpoint=endpoint, name=name)
        self.routes.append(route)

    def route(
        self,
        path: str,
        methods: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        include_in_schema: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        We no longer document this decorator style API, and its usage is discouraged.
        Instead you should use the following approach:

        >>> routes = [Route(path, endpoint=...), ...]
        >>> app = Starlette(routes=routes)
        """
        warnings.warn(
            'The `route` decorator is deprecated, and will be removed in version 1.0.0.Refer to https://www.starlette.io/routing/#http-routing for the recommended approach.',
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_route(path, func, methods=methods, name=name, include_in_schema=include_in_schema)
            return func

        return decorator

    def websocket_route(
        self,
        path: str,
        name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        We no longer document this decorator style API, and its usage is discouraged.
        Instead you should use the following approach:

        >>> routes = [WebSocketRoute(path, endpoint=...), ...]
        >>> app = Starlette(routes=routes)
        """
        warnings.warn(
            'The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.',
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def add_event_handler(self, event_type: str, func: Callable[..., Any]) -> None:
        assert event_type in ('startup', 'shutdown')
        if event_type == 'startup':
            self.on_startup.append(func)
        else:
            self.on_shutdown.append(func)

    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        warnings.warn(
            'The `on_event` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/lifespan/ for recommended approach.',
            DeprecationWarning,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_event_handler(event_type, func)
            return func

        return decorator
