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
    """
    Raised by `.url_for(name, **path_params)` and `.url_path_for(name, **path_params)`
    if no matching route exists.
    """

    def __init__(self, name: str, path_params: dict[str, typing.Any]) -> None:
        params = ', '.join(list(path_params.keys()))
        super().__init__(
            f'No route exists for name "{name}" and params "{params}".')


class Match(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


def func_lh241roy(obj: typing.Any) -> bool:
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


def func_u8rp4mi8(func: typing.Callable[[Request], typing.Union[Response, typing.Awaitable[Response]]]) -> ASGIApp:
    """
    Takes a function or coroutine `func(request) -> response`,
    and returns an ASGI application.
    """
    f = func if is_async_callable(func) else functools.partial(
        run_in_threadpool, func)

    async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive, send)

        async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
            response = await f(request)
            await response(scope, receive, send)
        await wrap_app_handling_exceptions(app, request)(scope, receive, send)
    return app


def func_w1pbf9do(func: typing.Callable[[WebSocket], typing.Awaitable[None]]) -> ASGIApp:
    """
    Takes a coroutine `func(session)`, and returns an ASGI application.
    """

    async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
        session = WebSocket(scope, receive=receive, send=send)

        async def func_e6293lre(scope: Scope, receive: Receive, send: Send) -> None:
            await func(session)
        await wrap_app_handling_exceptions(app, session)(scope, receive, send)
    return app


def func_mtudy4p9(endpoint: typing.Any) -> str:
    return getattr(endpoint, '__name__', endpoint.__class__.__name__)


def func_0pue9e4m(path: str, param_convertors: dict[str, Convertor], path_params: dict[str, typing.Any]) -> tuple[str, dict[str, typing.Any]]:
    for key, value in list(path_params.items()):
        if '{' + key + '}' in path:
            convertor = param_convertors[key]
            value = convertor.to_string(value)
            path = path.replace('{' + key + '}', value)
            path_params.pop(key)
    return path, path_params


PARAM_REGEX = re.compile('{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}'
    )


def func_mjfkrmqb(path: str) -> tuple[typing.Pattern[str], str, dict[str, Convertor]]:
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
    duplicated_params = set()
    idx = 0
    param_convertors: dict[str, Convertor] = {}
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

    def func_e6b06el3(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]:
        raise NotImplementedError()

    def func_ey72marj(self, name: str, /, **path_params: typing.Any) -> URLPath:
        raise NotImplementedError()

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
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

    def __init__(self, path: str, endpoint: typing.Any, *, methods: typing.Optional[typing.List[str]] = None, name: typing.Optional[str] = None,
        include_in_schema: bool = True, middleware: typing.Optional[typing.List[typing.Tuple[typing.Type[typing.Any], typing.Sequence[typing.Any], typing.Mapping[str, typing.Any]]]] = None) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path = path
        self.endpoint = endpoint
        self.name = func_mtudy4p9(endpoint) if name is None else name
        self.include_in_schema = include_in_schema
        endpoint_handler = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(
            endpoint_handler):
            self.app = func_u8rp4mi8(endpoint)
            if methods is None:
                methods = ['GET']
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        if methods is None:
            self.methods: typing.Optional[typing.Set[str]] = None
        else:
            self.methods = {method.upper() for method in methods}
            if 'GET' in self.methods:
                self.methods.add('HEAD')
        self.path_regex, self.path_format, self.param_convertors = (
            func_mjfkrmqb(path))

    def func_e6b06el3(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]:
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
                child_scope = {'endpoint': self.endpoint, 'path_params':
                    path_params}
                if self.methods and scope['method'] not in self.methods:
                    return Match.PARTIAL, child_scope
                else:
                    return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: typing.Any) -> URLPath:
        seen_params = set(path_params.keys())
        expected_params = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = func_0pue9e4m(self.path_format, self.
            param_convertors, path_params)
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

    def __eq__(self, other: typing.Any) -> bool:
        return (isinstance(other, Route) and self.path == other.path and 
            self.endpoint == other.endpoint and self.methods == other.methods)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        methods = sorted(self.methods or [])
        path, name = self.path, self.name
        return (
            f'{class_name}(path={path!r}, name={name!r}, methods={methods!r})')


class WebSocketRoute(BaseRoute):

    def __init__(self, path: str, endpoint: typing.Any, *, name: typing.Optional[str] = None, middleware: typing.Optional[typing.List[typing.Tuple[typing.Type[typing.Any], typing.Sequence[typing.Any], typing.Mapping[str, typing.Any]]]] = None) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path = path
        self.endpoint = endpoint
        self.name = func_mtudy4p9(endpoint) if name is None else name
        endpoint_handler = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(
            endpoint_handler):
            self.app = func_w1pbf9do(endpoint)
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        self.path_regex, self.path_format, self.param_convertors = (
            func_mjfkrmqb(path))

    def func_e6b06el3(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]:
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
                child_scope = {'endpoint': self.endpoint, 'path_params':
                    path_params}
                return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: typing.Any) -> URLPath:
        seen_params = set(path_params.keys())
        expected_params = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = func_0pue9e4m(self.path_format, self.
            param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='websocket')

    async def func_zkflwkfi(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    def __eq__(self, other: typing.Any) -> bool:
        return isinstance(other, WebSocketRoute
            ) and self.path == other.path and self.endpoint == other.endpoint

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(path={self.path!r}, name={self.name!r})'
            )


class Mount(BaseRoute):

    def __init__(self, path: str, app: typing.Optional[ASGIApp] = None, routes: typing.Optional[typing.List[BaseRoute]] = None, name: typing.Optional[str] = None, *,
        middleware: typing.Optional[typing.List[typing.Tuple[typing.Type[typing.Any], typing.Sequence[typing.Any], typing.Mapping[str, typing.Any]]]] = None) -> None:
        assert path == '' or path.startswith('/'
            ), "Routed paths must start with '/'"
        assert app is not None or routes is not None, "Either 'app=...', or 'routes=' must be specified"
        self.path = path.rstrip('/')
        if app is not None:
            self._base_app = app
        else:
            self._base_app = Router(routes=routes)
        self.app = self._base_app
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(self.app, *args, **kwargs)
        self.name = name
        self.path_regex, self.path_format, self.param_convertors = (
            func_mjfkrmqb(self.path + '/{path:path}'))

    @property
    def func_m9rtai2w(self) -> typing.List[BaseRoute]:
        return getattr(self._base_app, 'routes', [])

    def func_e6b06el3(self, scope: Scope) -> tuple[Match, dict[str, typing.Any]]:
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
                child_scope = {'path_params': path_params, 'app_root_path':
                    scope.get('app_root_path', root_path), 'root_path': 
                    root_path + matched_path, 'endpoint': self.app}
                return Match.FULL, child_scope
        return Match.NONE, {}

    def func_ey72marj(self, name: str, /, **path_params: typing.Any) -> URLPath:
        if (self.name is not None and name == self.name and 'path' in
            path_params):
            path_params['path'] = path_params['path'].lstrip('/')
            path, remaining_params = func_0pue9e4m(self.path_format, self.
                param_convertors, path_params)
            if not remaining_params:
                return URLPath(path=path)
        elif self.name is None or name.startswith(self.name + ':'):
            if self.name is None:
                remaining_name = name
            else:
                remaining_name = name[len(self.name) + 1:]
            path_kwarg = path_params.get('path')
            path_params['path'] = ''
            path_prefix, remaining_params = func_0pue9e4m(self.path_format,
                self.param_convertors, path_params)
            if path_kwarg is not None:
                remaining_params['path'] = path_kwarg
            for route in (self.r