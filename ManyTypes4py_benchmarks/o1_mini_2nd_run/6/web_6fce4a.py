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

__all__ = [
    'Request', 'Response', 'ResourceOptions', 'View', 'ViewHandlerMethod',
    'ViewHandlerFun', 'ViewDecorator', 'PageArg', 'HttpClientT', 'Web',
    'CacheBackendT', 'CacheT', 'BlueprintT'
]

ViewHandlerMethod = Callable[[Arg['Request'], VarArg[Any], KwArg[Any]], Awaitable['Response']]

ViewHandler2ArgsFun = Callable[
    [Arg['View'], Arg['Request']],
    Union[Coroutine[Any, Any, 'Response'], Awaitable['Response']]
]

ViewHandlerVarArgsFun = Callable[
    [Arg['View'], Arg['Request'], VarArg[Any], KwArg[Any]],
    Union[Coroutine[Any, Any, 'Response'], Awaitable['Response']]
]

ViewHandlerFun = Union[ViewHandler2ArgsFun, ViewHandlerVarArgsFun]

ViewGetHandler = ViewHandlerFun

ViewDecorator = Callable[[ViewHandlerFun], ViewHandlerFun]

RoutedViewGetHandler = ViewDecorator

PageArg = Union[Type['View'], ViewHandlerFun]

RouteDecoratorRet = Callable[[PageArg], PageArg]

CORSListOption = Union[str, Sequence[str]]

class ResourceOptions(NamedTuple):
    """CORS Options for specific route, or defaults."""
    allow_credentials: bool = False
    expose_headers: Sequence[str] = ()
    allow_headers: Sequence[str] = ()
    max_age: Optional[int] = None
    allow_methods: Sequence[str] = ()

class CacheBackendT(ServiceT, abc.ABC):

    @abc.abstractmethod
    def __init__(self, app: '_AppT', url: str = 'memory://', **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def get(self, key: str) -> Any:
        ...

    @abc.abstractmethod
    async def set(self, key: str, value: Any, timeout: Optional[Seconds] = None) -> None:
        ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        ...

class CacheT(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
        timeout: Optional[Seconds] = None,
        key_prefix: Optional[str] = None,
        backend: Optional[CacheBackendT] = None,
        **kwargs: Any
    ) -> None:
        ...

    @abc.abstractmethod
    def view(
        self,
        timeout: Optional[Seconds] = None,
        include_headers: bool = False,
        key_prefix: Optional[str] = None,
        **kwargs: Any
    ) -> Callable[[ViewHandlerFun], ViewHandlerFun]:
        ...

class BlueprintT(abc.ABC):

    @abc.abstractmethod
    def cache(
        self,
        timeout: Optional[Seconds] = None,
        include_headers: bool = False,
        key_prefix: Optional[str] = None,
        backend: Optional[CacheBackendT] = None
    ) -> Callable[[ViewHandlerFun], ViewHandlerFun]:
        ...

    @abc.abstractmethod
    def route(
        self,
        uri: str,
        *,
        name: Optional[str] = None,
        base: Type['View'] = 'View'
    ) -> Callable[[PageArg], PageArg]:
        ...

    @abc.abstractmethod
    def static(
        self,
        uri: str,
        file_or_directory: Union[str, Path],
        *,
        name: Optional[str] = None
    ) -> None:
        ...

    @abc.abstractmethod
    def register(
        self,
        app: '_AppT',
        *,
        url_prefix: Optional[str] = None
    ) -> None:
        ...

    @abc.abstractmethod
    def init_webserver(self, web: 'Web') -> None:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: 'Web') -> None:
        ...
