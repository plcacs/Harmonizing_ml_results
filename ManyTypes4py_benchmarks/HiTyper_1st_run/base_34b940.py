"""Base interface for Web server and views."""
import abc
import socket
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Type, Union
from urllib.parse import quote
from mode import Service
from mode.utils.compat import want_str
from mode.utils.imports import SymbolArg, symbol_by_name
from yarl import URL
from faust.types import AppT
from faust.types.web import BlueprintT, ResourceOptions, View
__all__ = ['DEFAULT_BLUEPRINTS', 'BlueprintManager', 'Request', 'Response', 'Web']
_bytes = bytes
_BPArg = SymbolArg[BlueprintT]
_BPList = Iterable[Tuple[str, _BPArg]]
DEFAULT_BLUEPRINTS = [('/router', 'faust.web.apps.router:blueprint'), ('/table', 'faust.web.apps.tables.blueprint')]
PRODUCTION_BLUEPRINTS = [('', 'faust.web.apps.production_index:blueprint')]
DEBUG_BLUEPRINTS = [('/graph', 'faust.web.apps.graph:blueprint'), ('', 'faust.web.apps.stats:blueprint')]
CONTENT_SEPARATOR = b'\r\n\r\n'
HEADER_SEPARATOR = b'\r\n'
HEADER_KEY_VALUE_SEPARATOR = b': '

class Response:
    """Web server response and status."""

    @property
    @abc.abstractmethod
    def status(self) -> None:
        """Return the response status code."""
        ...

    @property
    @abc.abstractmethod
    def body(self) -> None:
        """Return the response body as bytes."""
        ...

    @property
    @abc.abstractmethod
    def headers(self) -> None:
        """Return mapping of response HTTP headers."""
        ...

    @property
    @abc.abstractmethod
    def content_length(self) -> None:
        """Return the size of the response body."""
        ...

    @property
    @abc.abstractmethod
    def content_type(self) -> None:
        """Return the response content type."""
        ...

    @property
    @abc.abstractmethod
    def charset(self) -> None:
        """Return the response character set."""
        ...

    @property
    @abc.abstractmethod
    def chunked(self) -> None:
        """Return :const:`True` if response is chunked."""
        ...

    @property
    @abc.abstractmethod
    def compression(self) -> None:
        """Return :const:`True` if the response body is compressed."""
        ...

    @property
    @abc.abstractmethod
    def keep_alive(self) -> None:
        """Return :const:`True` if HTTP keep-alive enabled."""
        ...

    @property
    @abc.abstractmethod
    def body_length(self) -> None:
        """Size of HTTP response body."""
        ...

class BlueprintManager:
    """Manager of all blueprints."""

    def __init__(self, initial=None) -> None:
        self.applied = False
        self._enabled = list(initial) if initial else []
        self._active = {}

    def add(self, prefix: Union[str, typing.Sequence[str], bool], blueprint: Union[str, typing.Sequence[str], bool]) -> None:
        """Register blueprint with this app."""
        if self.applied:
            raise RuntimeError('Cannot add blueprints after server started')
        self._enabled.append((prefix, blueprint))

    def apply(self, web: typing.Callable) -> None:
        """Apply all blueprints."""
        if not self.applied:
            self.applied = True
            for prefix, blueprint in self._enabled:
                bp = symbol_by_name(blueprint)
                self._apply_blueprint(web, prefix, bp)

    def _apply_blueprint(self, web: Union[faustypes.AppT, str, aiohttp.web.Application], prefix: Union[str, faustypes.web.Web, faustypes.AppT], bp: Union[str, faustypes.AppT, bool]) -> None:
        self._active[bp.name] = bp
        bp.register(web.app, url_prefix=prefix)
        bp.init_webserver(web)

class Web(Service):
    """Web server and HTTP interface."""
    default_blueprints = DEFAULT_BLUEPRINTS
    production_blueprints = PRODUCTION_BLUEPRINTS
    debug_blueprints = DEBUG_BLUEPRINTS
    content_separator = CONTENT_SEPARATOR
    header_separator = HEADER_SEPARATOR
    header_key_value_separator = HEADER_KEY_VALUE_SEPARATOR

    def __init__(self, app: Any, **kwargs) -> None:
        self.app = app
        self.views = {}
        self.reverse_names = {}
        blueprints = list(self.default_blueprints)
        if self.app.conf.debug:
            blueprints.extend(self.debug_blueprints)
        else:
            blueprints.extend(self.production_blueprints)
        self.blueprints = BlueprintManager(blueprints)
        Service.__init__(self, **kwargs)

    @abc.abstractmethod
    def text(self, value: Union[str, int, typing.MutableMapping], *, content_type: Union[None, str, int, typing.MutableMapping]=None, status: int=200, reason: Union[None, str, int, typing.MutableMapping]=None, headers: Union[None, str, int, typing.MutableMapping]=None) -> None:
        """Create text response, using "text/plain" content-type."""
        ...

    @abc.abstractmethod
    def html(self, value: Union[str, int, typing.MutableMapping], *, content_type: Union[None, str, int, typing.MutableMapping]=None, status: int=200, reason: Union[None, str, int, typing.MutableMapping]=None, headers: Union[None, str, int, typing.MutableMapping]=None) -> None:
        """Create HTML response from string, ``text/html`` content-type."""
        ...

    @abc.abstractmethod
    def json(self, value: Union[str, typing.MutableMapping, int], *, content_type: Union[None, str, typing.MutableMapping, int]=None, status: int=200, reason: Union[None, str, typing.MutableMapping, int]=None, headers: Union[None, str, typing.MutableMapping, int]=None) -> None:
        """Create new JSON response.

        Accepts any JSON-serializable value and will automatically
        serialize it for you.

        The content-type is set to "application/json".
        """
        ...

    @abc.abstractmethod
    def bytes(self, value: Union[str, int, typing.MutableMapping], *, content_type: Union[None, str, int, typing.MutableMapping]=None, status: int=200, reason: Union[None, str, int, typing.MutableMapping]=None, headers: Union[None, str, int, typing.MutableMapping]=None) -> None:
        """Create new ``bytes`` response - for binary data."""
        ...

    @abc.abstractmethod
    def bytes_to_response(self, s: Union[str, bytes]) -> None:
        """Deserialize HTTP response from byte string."""
        ...

    def _bytes_to_response(self, s: Union[bytes, str]) -> tuple[typing.Union[HTTPStatus,str]]:
        status_code, _, payload = s.partition(self.content_separator)
        headers, _, body = payload.partition(self.content_separator)
        return (HTTPStatus(int(status_code)), dict((self._splitheader(h) for h in headers.splitlines())), body)

    def _splitheader(self, header: Union[str, bytes, dict]) -> tuple:
        key, value = header.split(self.header_key_value_separator, 1)
        return (want_str(key.strip()), want_str(value.strip()))

    @abc.abstractmethod
    def response_to_bytes(self, response: Union[bytes, dict]) -> None:
        """Serialize HTTP response into byte string."""
        ...

    def _response_to_bytes(self, status: Union[int, typing.Mapping, bytes], headers: Union[int, typing.Mapping, bytes], body: Union[int, typing.Mapping, bytes]) -> Union[bytes, str]:
        return self.content_separator.join([str(status).encode(), self.content_separator.join([self._headers_serialize(headers), body])])

    def _headers_serialize(self, headers: eth.abc.BlockHeaderAPI) -> Union[str, bytes, None]:
        return self.header_separator.join((self.header_key_value_separator.join([k if isinstance(k, _bytes) else k.encode('ascii'), v if isinstance(v, _bytes) else v.encode('latin-1')]) for k, v in headers.items()))

    @abc.abstractmethod
    def route(self, pattern: Union[str, typing.Callable, typing.Mapping], handler: Union[str, typing.Callable, typing.Mapping], cors_options: Union[None, str, typing.Callable, typing.Mapping]=None) -> None:
        """Add route for handler."""
        ...

    @abc.abstractmethod
    def add_static(self, prefix: Union[str, pathlib.Path], path: Union[str, pathlib.Path], **kwargs) -> None:
        """Add static route."""
        ...

    @abc.abstractmethod
    async def read_request_content(self, request):
        """Read HTTP body as bytes."""
        ...

    @abc.abstractmethod
    async def wsgi(self):
        """WSGI entry point."""
        ...

    def add_view(self, view_cls: Union[str, typing.Callable, None, bool], *, prefix: typing.Text='', cors_options: Union[None, typing.Callable, str, T]=None):
        """Add route for view."""
        view = view_cls(self.app, self)
        path = prefix.rstrip('/') + '/' + view.view_path.lstrip('/')
        self.route(path, view, cors_options)
        self.views[path] = view
        self.reverse_names[view.view_name] = path
        return view

    def url_for(self, view_name: str, **kwargs):
        """Get URL by view name.

        If the provided view name has associated URL parameters,
        those need to be passed in as kwargs, or a :exc:`TypeError`
        will be raised.
        """
        try:
            path = self.reverse_names[view_name]
        except KeyError:
            raise KeyError(f'No view with name {view_name!r} found')
        else:
            return path.format(**{k: self._quote_for_url(str(v)) for k, v in kwargs.items()})

    def _quote_for_url(self, value: str) -> Union[str, None, set[str]]:
        return quote(value, safe='')

    def init_server(self) -> None:
        """Initialize and setup web server."""
        self.blueprints.apply(self)
        self.app.on_webserver_init(self)

    @property
    def url(self) -> URL:
        """Return the canonical URL to this worker (including port)."""
        canon = self.app.conf.canonical_url
        if canon.host == socket.gethostname():
            return URL(f'http://localhost:{self.app.conf.web_port}/')
        return self.app.conf.canonical_url

class Request(abc.ABC):
    """HTTP Request."""

    @abc.abstractmethod
    def can_read_body(self) -> None:
        """Return :const:`True` if the request has a body."""
        ...

    @abc.abstractmethod
    async def read(self):
        """Read post data as bytes."""
        ...

    @abc.abstractmethod
    async def text(self):
        """Read post data as text."""
        ...

    @abc.abstractmethod
    async def json(self):
        """Read post data and deserialize as JSON."""
        ...

    @abc.abstractmethod
    async def post(self):
        """Read post data."""
        ...

    @property
    @abc.abstractmethod
    def match_info(self) -> None:
        """Return match info from URL route as a mapping."""
        ...

    @property
    @abc.abstractmethod
    def query(self) -> None:
        """Return HTTP query parameters as a mapping."""
        ...

    @property
    @abc.abstractmethod
    def cookies(self) -> None:
        """Return cookies as a mapping."""
        ...