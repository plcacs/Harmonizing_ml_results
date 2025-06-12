import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Type, Union, overload
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike
if TYPE_CHECKING:
    from .web_request import Request
    from .web_response import StreamResponse
    from .web_urldispatcher import AbstractRoute, UrlDispatcher
else:
    Request = StreamResponse = UrlDispatcher = AbstractRoute = None
__all__ = ('AbstractRouteDef', 'RouteDef', 'StaticDef', 'RouteTableDef', 'head', 'options', 'get', 'post', 'patch', 'put', 'delete', 'route', 'view', 'static')

class AbstractRouteDef(abc.ABC):

    @abc.abstractmethod
    def register(self, router):
        """Register itself into the given router."""
_HandlerType = Union[Type[AbstractView], Handler]

@dataclasses.dataclass(frozen=True, repr=False)
class RouteDef(AbstractRouteDef):

    def __repr__(self):
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f', {name}={value!r}')
        return '<RouteDef {method} {path} -> {handler.__name__!r}{info}>'.format(method=self.method, path=self.path, handler=self.handler, info=''.join(info))

    def register(self, router):
        if self.method in hdrs.METH_ALL:
            reg = getattr(router, 'add_' + self.method.lower())
            return [reg(self.path, self.handler, **self.kwargs)]
        else:
            return [router.add_route(self.method, self.path, self.handler, **self.kwargs)]

@dataclasses.dataclass(frozen=True, repr=False)
class StaticDef(AbstractRouteDef):

    def __repr__(self):
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f', {name}={value!r}')
        return '<StaticDef {prefix} -> {path}{info}>'.format(prefix=self.prefix, path=self.path, info=''.join(info))

    def register(self, router):
        resource = router.add_static(self.prefix, self.path, **self.kwargs)
        routes = resource.get_info().get('routes', {})
        return list(routes.values())

def route(method, path, handler, **kwargs):
    return RouteDef(method, path, handler, kwargs)

def head(path, handler, **kwargs):
    return route(hdrs.METH_HEAD, path, handler, **kwargs)

def options(path, handler, **kwargs):
    return route(hdrs.METH_OPTIONS, path, handler, **kwargs)

def get(path, handler, *, name=None, allow_head=True, **kwargs):
    return route(hdrs.METH_GET, path, handler, name=name, allow_head=allow_head, **kwargs)

def post(path, handler, **kwargs):
    return route(hdrs.METH_POST, path, handler, **kwargs)

def put(path, handler, **kwargs):
    return route(hdrs.METH_PUT, path, handler, **kwargs)

def patch(path, handler, **kwargs):
    return route(hdrs.METH_PATCH, path, handler, **kwargs)

def delete(path, handler, **kwargs):
    return route(hdrs.METH_DELETE, path, handler, **kwargs)

def view(path, handler, **kwargs):
    return route(hdrs.METH_ANY, path, handler, **kwargs)

def static(prefix, path, **kwargs):
    return StaticDef(prefix, path, kwargs)
_Deco = Callable[[_HandlerType], _HandlerType]

class RouteTableDef(Sequence[AbstractRouteDef]):
    """Route definition table"""

    def __init__(self):
        self._items = []

    def __repr__(self):
        return f'<RouteTableDef count={len(self._items)}>'

    @overload
    def __getitem__(self, index):
        ...

    @overload
    def __getitem__(self, index):
        ...

    def __getitem__(self, index):
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def route(self, method, path, **kwargs):

        def inner(handler):
            self._items.append(RouteDef(method, path, handler, kwargs))
            return handler
        return inner

    def head(self, path, **kwargs):
        return self.route(hdrs.METH_HEAD, path, **kwargs)

    def get(self, path, **kwargs):
        return self.route(hdrs.METH_GET, path, **kwargs)

    def post(self, path, **kwargs):
        return self.route(hdrs.METH_POST, path, **kwargs)

    def put(self, path, **kwargs):
        return self.route(hdrs.METH_PUT, path, **kwargs)

    def patch(self, path, **kwargs):
        return self.route(hdrs.METH_PATCH, path, **kwargs)

    def delete(self, path, **kwargs):
        return self.route(hdrs.METH_DELETE, path, **kwargs)

    def options(self, path, **kwargs):
        return self.route(hdrs.METH_OPTIONS, path, **kwargs)

    def view(self, path, **kwargs):
        return self.route(hdrs.METH_ANY, path, **kwargs)

    def static(self, prefix, path, **kwargs):
        self._items.append(StaticDef(prefix, path, kwargs))