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
    def __init__(self, app, url='memory://', **kwargs) -> None:
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
    def __init__(self, timeout: Union[None, str, mode.Seconds, int]=None, key_prefix: Union[None, str, mode.Seconds, int]=None, backend: Union[None, str, mode.Seconds, int]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def view(self, timeout: Union[None, bool, str, mode.Seconds]=None, include_headers: bool=False, key_prefix: Union[None, bool, str, mode.Seconds]=None, **kwargs) -> None:
        ...

class BlueprintT(abc.ABC):

    @abc.abstractmethod
    def cache(self, timeout: Union[None, bool, str, mode.Seconds]=None, include_headers: bool=False, key_prefix: Union[None, bool, str, mode.Seconds]=None, backend: Union[None, bool, str, mode.Seconds]=None) -> None:
        ...

    @abc.abstractmethod
    def route(self, uri: Union[str, None, typing.Type], *, name: Union[None, str, typing.Type]=None, base: View=View) -> None:
        ...

    @abc.abstractmethod
    def static(self, uri: Union[str, None, pathlib.Path], file_or_directory: Union[str, None, pathlib.Path], *, name: Union[None, str, pathlib.Path]=None) -> None:
        ...

    @abc.abstractmethod
    def register(self, app: Union[str, None, faustypes.app.AppT], *, url_prefix: Union[None, str, faustypes.app.AppT]=None) -> None:
        ...

    @abc.abstractmethod
    def init_webserver(self, web: Union[typing.Callable, aiohttp.web.Application]) -> None:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Union[faustypes.web.Web, str, abilian.app.Application]) -> None:
        ...