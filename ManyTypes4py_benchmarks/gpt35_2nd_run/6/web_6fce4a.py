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
__all__: Sequence[str] = ['Request', 'Response', 'ResourceOptions', 'View', 'ViewHandlerMethod', 'ViewHandlerFun', 'ViewDecorator', 'PageArg', 'HttpClientT', 'Web', 'CacheBackendT', 'CacheT', 'BlueprintT']
ViewHandlerMethod: Type[Callable[[Arg(Request), VarArg(Any), KwArg(Any)], Awaitable[Response]]] = Callable
ViewHandler2ArgsFun: Type[Callable[[Arg(View), Arg(Request)], Union[Coroutine[Any, Any, Response], Awaitable[Response]]]] = Callable
ViewHandlerVarArgsFun: Type[Callable[[Arg(View), Arg(Request), VarArg(Any), KwArg(Any)], Union[Coroutine[Any, Any, Response], Awaitable[Response]]] = Callable
ViewHandlerFun: Type[Union[ViewHandler2ArgsFun, ViewHandlerVarArgsFun]] = Callable
ViewGetHandler: Type[ViewHandlerFun] = Callable
ViewDecorator: Type[Callable[[ViewHandlerFun], ViewHandlerFun]] = Callable
RoutedViewGetHandler: Type[ViewDecorator] = Callable
PageArg: Type[Union[Type[View], ViewHandlerFun]] = Union
RouteDecoratorRet: Type[Callable[[PageArg], PageArg]] = Callable
CORSListOption: Type[Union[str, Sequence[str]]] = Union

class ResourceOptions(NamedTuple):
    """CORS Options for specific route, or defaults."""
    allow_credentials: bool = False
    expose_headers: Sequence[str] = ()
    allow_headers: Sequence[str] = ()
    max_age: Optional[int] = None
    allow_methods: Sequence[str] = ()

class CacheBackendT(ServiceT):

    @abc.abstractmethod
    def __init__(self, app: Any, url: str = 'memory://', **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def get(self, key: Any) -> Any:
        ...

    @abc.abstractmethod
    async def set(self, key: Any, value: Any, timeout: Optional[int] = None) -> None:
        ...

    @abc.abstractmethod
    async def delete(self, key: Any) -> None:
        ...

class CacheT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, timeout: Optional[int] = None, key_prefix: Optional[str] = None, backend: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def view(self, timeout: Optional[int] = None, include_headers: bool = False, key_prefix: Optional[str] = None, **kwargs: Any) -> None:
        ...

class BlueprintT(abc.ABC):

    @abc.abstractmethod
    def cache(self, timeout: Optional[int] = None, include_headers: bool = False, key_prefix: Optional[str] = None, backend: Any = None) -> None:
        ...

    @abc.abstractmethod
    def route(self, uri: str, *, name: Optional[str] = None, base: Type[View] = View) -> None:
        ...

    @abc.abstractmethod
    def static(self, uri: str, file_or_directory: str, *, name: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def register(self, app: Any, *, url_prefix: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def init_webserver(self, web: Any) -> None:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Any) -> None:
        ...
