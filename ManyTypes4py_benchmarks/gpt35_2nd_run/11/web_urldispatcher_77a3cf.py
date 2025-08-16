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

CIRCULAR_SYMLINK_ERROR: Tuple[Union[OSError, RuntimeError], ...] = (OSError,) if sys.version_info < (3, 10) and sys.platform.startswith('win32') else (RuntimeError,) if sys.version_info < (3, 13) else ()

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
        ...

    @abc.abstractmethod
    def url_for(self, **kwargs) -> str:
        ...

    @abc.abstractmethod
    async def resolve(self, request: Request) -> _Resolve:
        ...

    @abc.abstractmethod
    def add_prefix(self, prefix: str) -> None:
        ...

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        ...

    def freeze(self) -> None:
        pass

    @abc.abstractmethod
    def raw_match(self, path: str) -> bool:
        ...

class AbstractRoute(abc.ABC):

    def __init__(self, method: str, handler: Handler, *, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None, resource: Optional[AbstractResource] = None) -> None:
        ...

    @property
    def method(self) -> str:
        ...

    @property
    def handler(self) -> Handler:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        ...

    @property
    def resource(self) -> Optional[AbstractResource]:
        ...

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        ...

    @abc.abstractmethod
    def url_for(self, *args, **kwargs) -> str:
        ...

    async def handle_expect_header(self, request: Request) -> StreamResponse:
        ...

class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):
    __slots__: Tuple[str, ...] = ('_route', '_apps', '_current_app', '_frozen')

    def __init__(self, match_dict: Dict[str, str], route: AbstractRoute) -> None:
        ...

    @property
    def handler(self) -> Handler:
        ...

    @property
    def route(self) -> AbstractRoute:
        ...

    @property
    def expect_handler(self) -> Callable[[Request], Awaitable[Optional[StreamResponse]]]:
        ...

    @property
    def http_exception(self) -> None:
        ...

    def get_info(self) -> _InfoDict:
        ...

    @property
    def apps(self) -> Tuple['Application', ...]:
        ...

    def add_app(self, app: 'Application') -> None:
        ...

    @property
    def current_app(self) -> 'Application':
        ...

    @current_app.setter
    def current_app(self, app: 'Application') -> None:
        ...

    def freeze(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

class MatchInfoError(UrlMappingMatchInfo):
    __slots__: Tuple[str, ...] = ('_exception',)

    def __init__(self, http_exception: HTTPException) -> None:
        ...

    @property
    def http_exception(self) -> HTTPException:
        ...

    def __repr__(self) -> str:
        ...

async def _default_expect_handler(request: Request) -> Optional[StreamResponse]:
    ...

class Resource(AbstractResource):

    def __init__(self, *, name: Optional[str] = None) -> None:
        ...

    def add_route(self, method: str, handler: Handler, *, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None) -> ResourceRoute:
        ...

    def register_route(self, route: ResourceRoute) -> None:
        ...

    async def resolve(self, request: Request) -> _Resolve:
        ...

    @abc.abstractmethod
    def _match(self, path: str) -> Optional[Dict[str, str]]:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[AbstractRoute]:
        ...

class PlainResource(Resource):

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def freeze(self) -> None:
        ...

    def add_prefix(self, prefix: str) -> None:
        ...

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        ...

    def raw_match(self, path: str) -> bool:
        ...

    def get_info(self) -> _InfoDict:
        ...

    def url_for(self) -> URL:
        ...

    def __repr__(self) -> str:
        ...

class DynamicResource(Resource):

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def add_prefix(self, prefix: str) -> None:
        ...

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        ...

    def raw_match(self, path: str) -> bool:
        ...

    def get_info(self) -> _InfoDict:
        ...

    def url_for(self, **parts) -> URL:
        ...

    def __repr__(self) -> str:
        ...

class PrefixResource(AbstractResource):

    def __init__(self, prefix: str, *, name: Optional[str] = None) -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def add_prefix(self, prefix: str) -> None:
        ...

    def raw_match(self, prefix: str) -> bool:
        ...

class StaticResource(PrefixResource):

    def __init__(self, prefix: str, directory: str, *, name: Optional[str] = None, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None, chunk_size: int = 256 * 1024, show_index: bool = False, follow_symlinks: bool = False, append_version: bool = False) -> None:
        ...

    def url_for(self, *, filename: str, append_version: Optional[bool] = None) -> URL:
        ...

    def get_info(self) -> _InfoDict:
        ...

    def set_options_route(self, handler: Handler) -> None:
        ...

    async def resolve(self, request: Request) -> _Resolve:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[ResourceRoute]:
        ...

    async def _handle(self, request: Request) -> Union[Response, FileResponse]:
        ...

    def _resolve_path_to_response(self, unresolved_path: Path) -> Union[Response, FileResponse]:
        ...

    def _directory_as_html(self, dir_path: Path) -> str:
        ...

    def __repr__(self) -> str:
        ...

class PrefixedSubAppResource(PrefixResource):

    def __init__(self, prefix: str, app: 'Application') -> None:
        ...

    def add_prefix(self, prefix: str) -> None:
        ...

    def url_for(self, *args, **kwargs) -> NoReturn:
        ...

    def get_info(self) -> _InfoDict:
        ...

    async def resolve(self, request: Request) -> _Resolve:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[ResourceRoute]:
        ...

    def __repr__(self) -> str:
        ...

class AbstractRuleMatching(abc.ABC):

    @abc.abstractmethod
    async def match(self, request: Request) -> bool:
        ...

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        ...

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        ...

class Domain(AbstractRuleMatching):

    def __init__(self, domain: str) -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def validation(self, domain: str) -> str:
        ...

    async def match(self, request: Request) -> bool:
        ...

    def match_domain(self, host: str) -> bool:
        ...

    def get_info(self) -> _InfoDict:
        ...

class MaskDomain(Domain):

    def __init__(self, domain: str) -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def match_domain(self, host: str) -> bool:
        ...

class MatchedSubAppResource(PrefixedSubAppResource):

    def __init__(self, rule: AbstractRuleMatching, app: 'Application') -> None:
        ...

    @property
    def canonical(self) -> str:
        ...

    def get_info(self) -> _InfoDict:
        ...

    async def resolve(self, request: Request) -> _Resolve:
        ...

    def __repr__(self) -> str:
        ...

class ResourceRoute(AbstractRoute):

    def __init__(self, method: str, handler: Handler, resource: AbstractResource, *, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    def url_for(self, *args, **kwargs) -> str:
        ...

    def get_info(self) -> _InfoDict:
        ...

class SystemRoute(AbstractRoute):

    def __init__(self, http_exception: HTTPException) -> None:
        ...

    def url_for(self, *args, **kwargs) -> NoReturn:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    def get_info(self) -> _InfoDict:
        ...

    async def _handle(self, request: Request) -> NoReturn:
        ...

    @property
    def status(self) -> int:
        ...

    @property
    def reason(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class View(AbstractView):

    async def _iter(self) -> Any:
        ...

    def __await__(self) -> Any:
        ...

    def _raise_allowed_methods(self) -> NoReturn:
        ...

class ResourcesView(Sized, Iterable[AbstractResource], Container[AbstractResource]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[AbstractResource]:
        ...

    def __contains__(self, resource: AbstractResource) -> bool:
        ...

class RoutesView(Sized, Iterable[AbstractRoute], Container[AbstractRoute]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[AbstractRoute]:
        ...

    def __contains__(self, route: AbstractRoute) -> bool:
        ...

class UrlDispatcher(AbstractRouter, Mapping[str, AbstractResource]):

    def __init__(self) -> None:
        ...

    async def resolve(self, request: Request) -> UrlMappingMatchInfo:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, resource: AbstractResource) -> bool:
        ...

    def __getitem__(self, name: str) -> AbstractResource:
        ...

    def resources(self) -> ResourcesView:
        ...

    def routes(self) -> RoutesView:
        ...

    def named_resources(self) -> MappingProxyType:
        ...

    def register_resource(self, resource: AbstractResource) -> None:
        ...

    def _get_resource_index_key(self, resource: AbstractResource) -> str:
        ...

    def index_resource(self, resource: AbstractResource) -> None:
        ...

    def unindex_resource(self, resource: AbstractResource) -> None:
        ...

    def add_resource(self, path: str, *, name: Optional[str] = None) -> AbstractResource:
        ...

    def add_route(self, method: str, path: str, handler: Handler, *, name: Optional[str] = None, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None) -> ResourceRoute:
        ...

    def add_static(self, prefix: str, path: str, *, name: Optional[str] = None, expect_handler: Optional[Callable[[Request], Awaitable[Optional[StreamResponse]]]] = None, chunk_size: int = 256 * 1024, show_index: bool = False, follow_symlinks: bool = False, append_version: bool = False) -> StaticResource:
        ...

    def add_head(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_options(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_get(self, path: str, handler: Handler, *, name: Optional[str] = None, allow_head: bool = True, **kwargs) -> ResourceRoute:
        ...

    def add_post(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_put(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_patch(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_delete(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def add_view(self, path: str, handler: Handler, **kwargs) -> ResourceRoute:
        ...

    def freeze(self) -> None:
        ...

    def add_routes(self, routes: List[AbstractRouteDef]) -> List[AbstractRoute]:
        ...

def _quote_path(value: str) -> str:
    ...

def _unquote_path_safe(value: str) -> str:
    ...

def _requote_path(value: str) -> str:
    ...
