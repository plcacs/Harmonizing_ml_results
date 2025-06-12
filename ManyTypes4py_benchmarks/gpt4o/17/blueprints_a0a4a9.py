from pathlib import Path
from typing import Callable, List, Optional, Type
from mode.utils.times import Seconds
from faust.types import AppT
from faust.types.web import BlueprintT, CacheBackendT, CacheT, PageArg, ResourceOptions, RouteDecoratorRet, View, Web
from .cache import Cache

__all__ = ['Blueprint']

class FutureRoute(NamedTuple):
    """Describes web route to be registered later."""
    uri: str
    name: str
    handler: Callable
    base: Type[View]
    cors_options: Mapping

class FutureStaticRoute(NamedTuple):
    """Describes static route to be registered later."""
    uri: str
    file_or_directory: Path
    name: str

class Blueprint(BlueprintT):
    """Define reusable web application."""
    view_name_separator: str = ':'

    def __init__(self, name: str, *, url_prefix: Optional[str] = None) -> None:
        self.name: str = name
        self.url_prefix: Optional[str] = url_prefix
        self.routes: List[FutureRoute] = []
        self.static_routes: List[FutureStaticRoute] = []

    def cache(self, timeout: Optional[Seconds] = None, include_headers: bool = False, key_prefix: Optional[str] = None, backend: Optional[CacheBackendT] = None) -> CacheT:
        """Cache API."""
        if key_prefix is None:
            key_prefix = self.name
        return Cache(timeout, include_headers, key_prefix, backend)

    def route(self, uri: str, *, name: Optional[str] = None, cors_options: Optional[Mapping] = None, base: Type[View] = View) -> RouteDecoratorRet:
        """Create route by decorating handler or view class."""

        def _inner(handler: Callable) -> Callable:
            route = FutureRoute(uri=uri, name=name or handler.__name__, handler=handler, base=base, cors_options=cors_options or {})
            self.routes.append(route)
            return handler
        return _inner

    def static(self, uri: str, file_or_directory: Union[str, Path], *, name: Optional[str] = None) -> None:
        """Add static route."""
        _name = name or 'static'
        if not _name.startswith(self.name + '.'):
            _name = f'{self.name}.{name}'
        fut = FutureStaticRoute(uri, Path(file_or_directory), _name)
        self.static_routes.append(fut)

    def register(self, app: AppT, *, url_prefix: Optional[str] = None) -> None:
        """Register blueprint with app."""
        url_prefix = url_prefix or self.url_prefix
        for route in self.routes:
            self._apply_route(app, route, url_prefix)
        for static_route in self.static_routes:
            self._apply_static_route(app.web, static_route, url_prefix)

    def _apply_route(self, app: AppT, route: FutureRoute, url_prefix: Optional[str]) -> None:
        uri = self._url_with_prefix(route.uri, url_prefix)
        app.page(path=uri[1:] if uri.startswith('//') else uri, name=self._view_name(route.name), cors_options=route.cors_options)(route.handler)

    def _view_name(self, name: str) -> str:
        return self.view_name_separator.join([self.name, name])

    def init_webserver(self, web: Web) -> None:
        """Init blueprint for web server start."""
        self.on_webserver_init(web)

    def on_webserver_init(self, web: Web) -> None:
        """Call when web server starts."""
        ...

    def _url_with_prefix(self, url: str, prefix: Optional[str] = None) -> str:
        if prefix:
            return prefix.rstrip('/') + '/' + url.lstrip('/')
        return url

    def _apply_static_route(self, web: Web, route: FutureStaticRoute, url_prefix: Optional[str]) -> None:
        uri = self._url_with_prefix(route.uri, url_prefix)
        web.add_static(uri, route.file_or_directory)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.name}>'
