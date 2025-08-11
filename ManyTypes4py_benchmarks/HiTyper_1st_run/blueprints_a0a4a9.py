"""Blueprints define reusable web apps.

They are lazy and need to be registered to an app to be activated:

.. sourcecode:: python

    from faust import web

    blueprint = web.Blueprint('users')
    cache = blueprint.cache(timeout=300.0)

    @blueprint.route('/', name='list')
    class UserListView(web.View):

        @cache.view()
        async def get(self, request: web.Request) -> web.Response:
            return web.json(...)

    @blueprint.route('/{user_id}/', name='detail')
    class UserDetailView(web.View):

        @cache.view(timeout=10.0)
        async def get(self,
                      request: web.Request,
                      user_id: str) -> web.Response:
            return web.json(...)

At this point the views are realized and can be used
from Python code, but the cached ``get`` method handlers
cannot be called yet.

To actually use the view from a web server, we need to register
the blueprint to an app:

.. sourcecode:: python

    app = faust.App(
        'name',
        broker='kafka://',
        cache='redis://',
    )

    user_blueprint.register(app, url_prefix='/user/')

At this point the web server will have fully-realized views
with actually cached method handlers.

The blueprint is registered with a prefix, so the URL for the
``UserListView`` is now ``/user/``, and the URL for the ``UserDetailView``
is ``/user/{user_id}/``.

Blueprints can be registered to multiple apps at the same time.
"""
from pathlib import Path
from typing import List, Mapping, NamedTuple, Optional, Type, Union
from mode.utils.times import Seconds
from faust.types import AppT
from faust.types.web import BlueprintT, CacheBackendT, CacheT, PageArg, ResourceOptions, RouteDecoratorRet, View, Web
from .cache import Cache
__all__ = ['Blueprint']

class FutureRoute(NamedTuple):
    """Describes web route to be registered later."""

class FutureStaticRoute(NamedTuple):
    """Describes static route to be registered later."""

class Blueprint(BlueprintT):
    """Define reusable web application."""
    view_name_separator = ':'

    def __init__(self, name: Union[str, None], *, url_prefix: Union[None, str, bytes]=None) -> None:
        self.name = name
        self.url_prefix = url_prefix
        self.routes = []
        self.static_routes = []

    def cache(self, timeout: Union[None, mode.utils.times.Seconds, bool]=None, include_headers: bool=False, key_prefix: Union[None, str, bool, typing.Any]=None, backend: Union[None, mode.utils.times.Seconds, bool]=None) -> Cache:
        """Cache API."""
        if key_prefix is None:
            key_prefix = self.name
        return Cache(timeout, include_headers, key_prefix, backend)

    def route(self, uri: Union[str, typing.Type, None], *, name: Union[None, str, typing.Type]=None, cors_options: Union[None, str, typing.Type]=None, base: Any=View):
        """Create route by decorating handler or view class."""

        def _inner(handler: Any):
            route = FutureRoute(uri=uri, name=name or handler.__name__, handler=handler, base=base, cors_options=cors_options or {})
            self.routes.append(route)
            return handler
        return _inner

    def static(self, uri: Union[str, None], file_or_directory: Union[str, None], *, name: Union[None, str, pathlib.Path, list[str]]=None) -> None:
        """Add static route."""
        _name = name or 'static'
        if not _name.startswith(self.name + '.'):
            _name = f'{self.name}.{name}'
        fut = FutureStaticRoute(uri, Path(file_or_directory), _name)
        self.static_routes.append(fut)

    def register(self, app: Union[faustypes.AppT, str, Web], *, url_prefix: Union[None, str]=None) -> None:
        """Register blueprint with app."""
        url_prefix = url_prefix or self.url_prefix
        for route in self.routes:
            self._apply_route(app, route, url_prefix)
        for static_route in self.static_routes:
            self._apply_static_route(app.web, static_route, url_prefix)

    def _apply_route(self, app: Union[faustypes.AppT, str, None], route: Union[str, None, faustypes.AppT], url_prefix: Union[str, None]) -> None:
        uri = self._url_with_prefix(route.uri, url_prefix)
        app.page(path=uri[1:] if uri.startswith('//') else uri, name=self._view_name(route.name), cors_options=route.cors_options)(route.handler)

    def _view_name(self, name: Union[list[str], str, None]) -> Union[str, set[str]]:
        return self.view_name_separator.join([self.name, name])

    def init_webserver(self, web: typing.Callable[..., None]) -> None:
        """Init blueprint for web server start."""
        self.on_webserver_init(web)

    def on_webserver_init(self, web: Union[faustypes.web.Web, str, typing.Type]) -> None:
        """Call when web server starts."""
        ...

    def _url_with_prefix(self, url: str, prefix: str=None) -> str:
        if prefix:
            return prefix.rstrip('/') + '/' + url.lstrip('/')
        return url

    def _apply_static_route(self, web: Union[faustypes.web.Web, str], route: str, url_prefix: str) -> None:
        uri = self._url_with_prefix(route.uri, url_prefix)
        web.add_static(uri, route.file_or_directory)

    def __repr__(self) -> typing.Text:
        return f'<{type(self).__name__}: {self.name}>'