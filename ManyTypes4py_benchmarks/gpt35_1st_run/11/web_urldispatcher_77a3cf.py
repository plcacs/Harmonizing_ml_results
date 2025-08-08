import abc
import asyncio
import base64
import functools
import hashlib
import html
import keyword
import os
import re
import sys
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Container, Dict, Final, Generator, Iterable, Iterator, List, Mapping, NoReturn, Optional, Pattern, Set, Sized, Tuple, Type, TypedDict, Union, cast
from yarl import URL
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import HTTPException, HTTPExpectationFailed, HTTPForbidden, HTTPMethodNotAllowed, HTTPNotFound
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef

__all__: Tuple[str, ...] = ('UrlDispatcher', 'UrlMappingMatchInfo', 'AbstractResource', 'Resource', 'PlainResource', 'DynamicResource', 'AbstractRoute', 'ResourceRoute', 'StaticResource', 'View')

if TYPE_CHECKING:
    from .web_app import Application
    BaseDict: Dict[str, str]
else:
    BaseDict: dict

CIRCULAR_SYMLINK_ERROR: Union[Tuple[OSError], Tuple[RuntimeError]] = (OSError,) if sys.version_info < (3, 10) and sys.platform.startswith('win32') else (RuntimeError,) if sys.version_info < (3, 13) else ()

HTTP_METHOD_RE: Pattern[str] = re.compile("^[0-9A-Za-z!#\\$%&'\\*\\+\\-\\.\\^_`\\|~]+$")
ROUTE_RE: Pattern[str] = re.compile('(\\{[_a-zA-Z][^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}')
PATH_SEP: str = re.escape('/')

_ExpectHandler: Callable[[Request], Awaitable[Optional[StreamResponse]]]
_Resolve: Tuple[Optional['UrlMappingMatchInfo'], Set[str]]

html_escape: Callable[[str], str] = functools.partial(html.escape, quote=True)

class _InfoDict(TypedDict, total=False):
    pass

class AbstractResource(Sized, Iterable['AbstractRoute']):

    def __init__(self, *, name: Optional[str] = None) -> None:
        self._name: Optional[str] = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        pass

    @abc.abstractmethod
    def url_for(self, **kwargs) -> str:
        pass

    @abc.abstractmethod
    async def resolve(self, request: Request) -> _Resolve:
        pass

    @abc.abstractmethod
    def add_prefix(self, prefix: str) -> None:
        pass

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        pass

    def freeze(self) -> None:
        pass

    @abc.abstractmethod
    def raw_match(self, path: str) -> bool:
        pass

class AbstractRoute(abc.ABC):

    def __init__(self, method: str, handler: Handler, *, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None, resource: Optional[AbstractResource] = None) -> None:
        if expect_handler is None:
            expect_handler = _default_expect_handler
        assert asyncio.iscoroutinefunction(expect_handler), f'Coroutine is expected, got {expect_handler!r}'
        method = method.upper()
        if not HTTP_METHOD_RE.match(method):
            raise ValueError(f'{method} is not allowed HTTP method')
        if asyncio.iscoroutinefunction(handler):
            pass
        elif isinstance(handler, type) and issubclass(handler, AbstractView):
            pass
        else:
            raise TypeError('Only async functions are allowed as web-handlers , got {!r}'.format(handler))
        self._method: str = method
        self._handler: Handler = handler
        self._expect_handler: Callable[[Request], Awaitable[Optional[StreamResponse]]] = expect_handler
        self._resource: Optional[AbstractResource] = resource

    @property
    def method(self) -> str:
        return self._method

    @property
    def handler(self) -> Handler:
        return self._handler

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        pass

    @property
    def resource(self) -> Optional[AbstractResource]:
        return self._resource

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        pass

    @abc.abstractmethod
    def url_for(self, *args, **kwargs) -> str:
        pass

    async def handle_expect_header(self, request: Request) -> StreamResponse:
        return await self._expect_handler(request)

class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):
    __slots__: Tuple[str, ...] = ('_route', '_apps', '_current_app', '_frozen')

    def __init__(self, match_dict: Dict[str, str], route: AbstractRoute) -> None:
        super().__init__(match_dict)
        self._route: AbstractRoute = route
        self._apps: List['Application'] = []
        self._current_app: Optional['Application'] = None
        self._frozen: bool = False

    @property
    def handler(self) -> Handler:
        return self._route.handler

    @property
    def route(self) -> AbstractRoute:
        return self._route

    @property
    def expect_handler(self) -> Callable[[Request], Awaitable[Optional[StreamResponse]]]:
        return self._route.handle_expect_header

    @property
    def http_exception(self) -> Optional[HTTPException]:
        return None

    def get_info(self) -> _InfoDict:
        return self._route.get_info()

    @property
    def apps(self) -> Tuple['Application', ...]:
        return tuple(self._apps)

    def add_app(self, app: 'Application') -> None:
        if self._frozen:
            raise RuntimeError('Cannot change apps stack after .freeze() call')
        if self._current_app is None:
            self._current_app = app
        self._apps.insert(0, app)

    @property
    def current_app(self) -> 'Application':
        app = self._current_app
        assert app is not None
        return app

    @current_app.setter
    def current_app(self, app: 'Application') -> None:
        if DEBUG:
            if app not in self._apps:
                raise RuntimeError('Expected one of the following apps {!r}, got {!r}'.format(self._apps, app))
        self._current_app = app

    def freeze(self) -> None:
        self._frozen = True

    def __repr__(self) -> str:
        return f'<MatchInfo {super().__repr__()}: {self._route}>'

class MatchInfoError(UrlMappingMatchInfo):
    __slots__: Tuple[str, ...] = ('_exception',)

    def __init__(self, http_exception: HTTPException) -> None:
        self._exception: HTTPException = http_exception
        super().__init__({}, SystemRoute(self._exception))

    @property
    def http_exception(self) -> HTTPException:
        return self._exception

    def __repr__(self) -> str:
        return '<MatchInfoError {}: {}>'.format(self._exception.status, self._exception.reason)

async def _default_expect_handler(request: Request) -> StreamResponse:
    """Default handler for Expect header.

    Just send "100 Continue" to client.
    raise HTTPExpectationFailed if value of header is not "100-continue"
    """
    expect = request.headers.get(hdrs.EXPECT, '')
    if request.version == HttpVersion11:
        if expect.lower() == '100-continue':
            await request.writer.write(b'HTTP/1.1 100 Continue\r\n\r\n')
            request.writer.output_size = 0
        else:
            raise HTTPExpectationFailed(text='Unknown Expect: %s' % expect)
