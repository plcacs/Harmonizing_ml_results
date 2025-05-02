import abc
import typing
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, NamedTuple, Optional, Sequence, Type, Union
from aiohttp.client import ClientSession as HttpClientT
from mode import Seconds, ServiceT
from mypy_extensions import Arg, KwArg, VarArg
from yarl import URL
if typing.TYPE_CHECKING:
    from faust.types.app import AppT as _AppT
    from faust.web.base import Request, Response, Web
    from faust.web.views import View
else:

    class _AppT:
        ...

    class Request:
        ...

    class Response:
        ...

    class Web:
        ...

    class View:
        ...
__all__ = ['Request', 'Response', 'ResourceOptions', 'View', 'ViewHandlerMethod', 'ViewHandlerFun', 'ViewDecorator', 'PageArg', 'HttpClientT', 'Web', 'CacheBackendT', 'CacheT', 'BlueprintT']
ViewHandlerMethod = Callable[[Arg(Request), VarArg(Any), KwArg(Any)], Awaitable[Response]]
ViewHandler2ArgsFun = Callable[[Arg(View), Arg(Request)], Union[Coroutine[Any, Any, Response], Awaitable[Response]]]
ViewHandlerVarArgsFun = Callable[[Arg(View), Arg(Request), VarArg(Any), KwArg(Any)], Union[Coroutine[Any, Any, Response], Awaitable[Response]]]
ViewHandlerFun = Union[ViewHandler2ArgsFun, ViewHandlerVarArgsFun]
ViewGetHandler = ViewHandlerFun
ViewDecorator = Callable[[ViewHandlerFun], ViewHandlerFun]
RoutedViewGetHandler = ViewDecorator
PageArg = Union[Type[View], ViewHandlerFun]
RouteDecoratorRet = Callable[[PageArg], PageArg]
CORSListOption = Union[str, Sequence[str]]

class ResourceOptions(NamedTuple):
    """CORS Options for specific route, or defaults."""
    allow_credentials = False
    expose_headers = ()
    allow_headers = ()
    max_age = None
    allow_methods = ()

class CacheBackendT(ServiceT):

    @abc.abstractmethod
    def __init__(self, app, url='memory://', **kwargs):
        ...

    @abc.abstractmethod
    async def get(self, key):
        ...

    @abc.abstractmethod
    async def set(self, key, value, timeout=None):
        ...

    @abc.abstractmethod
    async def delete(self, key):
        ...

class CacheT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, timeout=None, key_prefix=None, backend=None, **kwargs):
        ...

    @abc.abstractmethod
    def view(self, timeout=None, include_headers=False, key_prefix=None, **kwargs):
        ...

class BlueprintT(abc.ABC):

    @abc.abstractmethod
    def cache(self, timeout=None, include_headers=False, key_prefix=None, backend=None):
        ...

    @abc.abstractmethod
    def route(self, uri, *, name=None, base=View):
        ...

    @abc.abstractmethod
    def static(self, uri, file_or_directory, *, name=None):
        ...

    @abc.abstractmethod
    def register(self, app, *, url_prefix=None):
        ...

    @abc.abstractmethod
    def init_webserver(self, web):
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web):
        ...