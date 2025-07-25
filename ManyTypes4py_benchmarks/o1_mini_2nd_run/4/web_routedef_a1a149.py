import abc
import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    overload,
)
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike

if TYPE_CHECKING:
    from .web_request import Request
    from .web_response import StreamResponse
    from .web_urldispatcher import AbstractRoute, UrlDispatcher
else:
    Request = Optional["Request"]
    StreamResponse = Optional["StreamResponse"]
    UrlDispatcher = Optional["UrlDispatcher"]
    AbstractRoute = Optional["AbstractRoute"]

__all__ = (
    "AbstractRouteDef",
    "RouteDef",
    "StaticDef",
    "RouteTableDef",
    "head",
    "options",
    "get",
    "post",
    "patch",
    "put",
    "delete",
    "route",
    "view",
    "static",
)

class AbstractRouteDef(abc.ABC):
    @abc.abstractmethod
    def register(self, router: "UrlDispatcher") -> List["AbstractRoute"]:
        """Register itself into the given router."""

_HandlerType = Union[Type[AbstractView], Handler]

@dataclasses.dataclass(frozen=True, repr=False)
class RouteDef(AbstractRouteDef):
    method: str
    path: PathLike
    handler: _HandlerType
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f", {name}={value!r}")
        return "<RouteDef {method} {path} -> {handler.__name__!r}{info}>".format(
            method=self.method, path=self.path, handler=self.handler, info="".join(info)
        )

    def register(self, router: "UrlDispatcher") -> List["AbstractRoute"]:
        if self.method in hdrs.METH_ALL:
            reg_method = f"add_{self.method.lower()}"
            reg = getattr(router, reg_method)  # type: Callable[..., "AbstractRoute"]
            return [reg(self.path, self.handler, **self.kwargs)]
        else:
            return [
                router.add_route(
                    self.method, self.path, self.handler, **self.kwargs
                )
            ]

@dataclasses.dataclass(frozen=True, repr=False)
class StaticDef(AbstractRouteDef):
    prefix: PathLike
    path: PathLike
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f", {name}={value!r}")
        return "<StaticDef {prefix} -> {path}{info}>".format(
            prefix=self.prefix, path=self.path, info="".join(info)
        )

    def register(self, router: "UrlDispatcher") -> List["AbstractRoute"]:
        resource = router.add_static(self.prefix, self.path, **self.kwargs)
        routes_info = resource.get_info().get("routes", {})
        return list(routes_info.values())

def route(
    method: str,
    path: PathLike,
    handler: _HandlerType,
    **kwargs: Any
) -> AbstractRouteDef:
    return RouteDef(method, path, handler, kwargs)

def head(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_HEAD, path, handler, **kwargs)

def options(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_OPTIONS, path, handler, **kwargs)

def get(
    path: PathLike,
    handler: _HandlerType,
    *,
    name: Optional[str] = None,
    allow_head: bool = True,
    **kwargs: Any
) -> AbstractRouteDef:
    return route(
        hdrs.METH_GET,
        path,
        handler,
        name=name,
        allow_head=allow_head,
        **kwargs
    )

def post(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_POST, path, handler, **kwargs)

def put(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_PUT, path, handler, **kwargs)

def patch(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_PATCH, path, handler, **kwargs)

def delete(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_DELETE, path, handler, **kwargs)

def view(path: PathLike, handler: _HandlerType, **kwargs: Any) -> AbstractRouteDef:
    return route(hdrs.METH_ANY, path, handler, **kwargs)

def static(prefix: PathLike, path: PathLike, **kwargs: Any) -> StaticDef:
    return StaticDef(prefix, path, kwargs)

_Deco = Callable[[_HandlerType], _HandlerType]

class RouteTableDef(Sequence[AbstractRouteDef]):
    """Route definition table"""

    def __init__(self) -> None:
        self._items: List[AbstractRouteDef] = []

    def __repr__(self) -> str:
        return f"<RouteTableDef count={len(self._items)}>"

    @overload
    def __getitem__(self, index: int) -> AbstractRouteDef:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[AbstractRouteDef]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[AbstractRouteDef, List[AbstractRouteDef]]:
        return self._items[index]

    def __iter__(self) -> Iterator[AbstractRouteDef]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def route(
        self, method: str, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        def inner(handler: _HandlerType) -> _HandlerType:
            self._items.append(RouteDef(method, path, handler, kwargs))
            return handler

        return inner

    def head(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_HEAD, path, **kwargs)

    def get(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_GET, path, **kwargs)

    def post(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_POST, path, **kwargs)

    def put(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_PUT, path, **kwargs)

    def patch(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_PATCH, path, **kwargs)

    def delete(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_DELETE, path, **kwargs)

    def options(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_OPTIONS, path, **kwargs)

    def view(
        self, path: PathLike, **kwargs: Any
    ) -> Callable[[_HandlerType], _HandlerType]:
        return self.route(hdrs.METH_ANY, path, **kwargs)

    def static(self, prefix: PathLike, path: PathLike, **kwargs: Any) -> None:
        self._items.append(StaticDef(prefix, path, kwargs))
