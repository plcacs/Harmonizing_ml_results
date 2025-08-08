from pathlib import Path
from typing import List, Mapping, NamedTuple, Optional, Type, Union
from mode.utils.times import Seconds
from faust.types import AppT
from faust.types.web import BlueprintT, CacheBackendT, CacheT, PageArg, ResourceOptions, RouteDecoratorRet, View, Web

class FutureRoute(NamedTuple):
    uri: str
    name: Optional[str]

class FutureStaticRoute(NamedTuple):
    uri: str
    file_or_directory: Path
    name: str

class Blueprint(BlueprintT):
    view_name_separator: str = ':'

    def __init__(self, name: str, url_prefix: Optional[str] = None) -> None:
        self.name: str = name
        self.url_prefix: Optional[str] = url_prefix
        self.routes: List[FutureRoute] = []
        self.static_routes: List[FutureStaticRoute] = []

    def cache(self, timeout: Optional[float] = None, include_headers: bool = False, key_prefix: Optional[str] = None, backend: Optional[CacheBackendT] = None) -> CacheT:
        ...

    def route(self, uri: str, name: Optional[str] = None, cors_options: Optional[Mapping[str, str]] = None, base: Type[View] = View) -> RouteDecoratorRet:
        ...

    def static(self, uri: str, file_or_directory: Path, name: Optional[str] = None) -> None:
        ...

    def register(self, app: AppT, url_prefix: Optional[str] = None) -> None:
        ...

    def _apply_route(self, app: AppT, route: FutureRoute, url_prefix: Optional[str]) -> None:
        ...

    def _view_name(self, name: str) -> str:
        ...

    def init_webserver(self, web: Web) -> None:
        ...

    def on_webserver_init(self, web: Web) -> None:
        ...

    def _url_with_prefix(self, url: str, prefix: Optional[str] = None) -> str:
        ...

    def _apply_static_route(self, web: Web, route: FutureStaticRoute, url_prefix: Optional[str]) -> None:
        ...

    def __repr__(self) -> str:
        ...
