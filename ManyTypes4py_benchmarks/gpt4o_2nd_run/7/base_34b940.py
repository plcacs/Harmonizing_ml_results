"""Base interface for Web server and views."""
import abc
import socket
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Type, Union, Dict
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

DEFAULT_BLUEPRINTS: List[Tuple[str, str]] = [
    ('/router', 'faust.web.apps.router:blueprint'),
    ('/table', 'faust.web.apps.tables.blueprint')
]
PRODUCTION_BLUEPRINTS: List[Tuple[str, str]] = [
    ('', 'faust.web.apps.production_index:blueprint')
]
DEBUG_BLUEPRINTS: List[Tuple[str, str]] = [
    ('/graph', 'faust.web.apps.graph:blueprint'),
    ('', 'faust.web.apps.stats:blueprint')
]

CONTENT_SEPARATOR: bytes = b'\r\n\r\n'
HEADER_SEPARATOR: bytes = b'\r\n'
HEADER_KEY_VALUE_SEPARATOR: bytes = b': '

class Response:
    """Web server response and status."""

    @property
    @abc.abstractmethod
    def status(self) -> int:
        """Return the response status code."""
        ...

    @property
    @abc.abstractmethod
    def body(self) -> bytes:
        """Return the response body as bytes."""
        ...

    @property
    @abc.abstractmethod
    def headers(self) -> Mapping[str, str]:
        """Return mapping of response HTTP headers."""
        ...

    @property
    @abc.abstractmethod
    def content_length(self) -> int:
        """Return the size of the response body."""
        ...

    @property
    @abc.abstractmethod
    def content_type(self) -> str:
        """Return the response content type."""
        ...

    @property
    @abc.abstractmethod
    def charset(self) -> str:
        """Return the response character set."""
        ...

    @property
    @abc.abstractmethod
    def chunked(self) -> bool:
        """Return :const:`True` if response is chunked."""
        ...

    @property
    @abc.abstractmethod
    def compression(self) -> bool:
        """Return :const:`True` if the response body is compressed."""
        ...

    @property
    @abc.abstractmethod
    def keep_alive(self) -> bool:
        """Return :const:`True` if HTTP keep-alive enabled."""
        ...

    @property
    @abc.abstractmethod
    def body_length(self) -> int:
        """Size of HTTP response body."""
        ...

class BlueprintManager:
    """Manager of all blueprints."""

    def __init__(self, initial: Optional[_BPList] = None) -> None:
        self.applied: bool = False
        self._enabled: List[Tuple[str, _BPArg]] = list(initial) if initial else []
        self._active: Dict[str, BlueprintT] = {}

    def add(self, prefix: str, blueprint: _BPArg) -> None:
        """Register blueprint with this app."""
        if self.applied:
            raise RuntimeError('Cannot add blueprints after server started')
        self._enabled.append((prefix, blueprint))

    def apply(self, web: 'Web') -> None:
        """Apply all blueprints."""
        if not self.applied:
            self.applied = True
            for prefix, blueprint in self._enabled:
                bp = symbol_by_name(blueprint)
                self._apply_blueprint(web, prefix, bp)

    def _apply_blueprint(self, web: 'Web', prefix: str, bp: BlueprintT) -> None:
        self._active[bp.name] = bp
        bp.register(web.app, url_prefix=prefix)
        bp.init_webserver(web)

class Web(Service):
    """Web server and HTTP interface."""
    default_blueprints: ClassVar[List[Tuple[str, str]]] = DEFAULT_BLUEPRINTS
    production_blueprints: ClassVar[List[Tuple[str, str]]] = PRODUCTION_BLUEPRINTS
    debug_blueprints: ClassVar[List[Tuple[str, str]]] = DEBUG_BLUEPRINTS
    content_separator: ClassVar[bytes] = CONTENT_SEPARATOR
    header_separator: ClassVar[bytes] = HEADER_SEPARATOR
    header_key_value_separator: ClassVar[bytes] = HEADER_KEY_VALUE_SEPARATOR

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        self.app: AppT = app
        self.views: Dict[str, View] = {}
        self.reverse_names: Dict[str, str] = {}
        blueprints = list(self.default_blueprints)
        if self.app.conf.debug:
            blueprints.extend(self.debug_blueprints)
        else:
            blueprints.extend(self.production_blueprints)
        self.blueprints = BlueprintManager(blueprints)
        Service.__init__(self, **kwargs)

    @abc.abstractmethod
    def text(self, value: str, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> Response:
        """Create text response, using "text/plain" content-type."""
        ...

    @abc.abstractmethod
    def html(self, value: str, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> Response:
        """Create HTML response from string, ``text/html`` content-type."""
        ...

    @abc.abstractmethod
    def json(self, value: Any, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> Response:
        """Create new JSON response.

        Accepts any JSON-serializable value and will automatically
        serialize it for you.

        The content-type is set to "application/json".
        """
        ...

    @abc.abstractmethod
    def bytes(self, value: bytes, *, content_type: Optional[str] = None, status: int = 200, reason: Optional[str] = None, headers: Optional[Mapping[str, str]] = None) -> Response:
        """Create new ``bytes`` response - for binary data."""
        ...

    @abc.abstractmethod
    def bytes_to_response(self, s: bytes) -> Response:
        """Deserialize HTTP response from byte string."""
        ...

    def _bytes_to_response(self, s: bytes) -> Tuple[HTTPStatus, Dict[str, str], bytes]:
        status_code, _, payload = s.partition(self.content_separator)
        headers, _, body = payload.partition(self.content_separator)
        return (HTTPStatus(int(status_code)), dict((self._splitheader(h) for h in headers.splitlines())), body)

    def _splitheader(self, header: bytes) -> Tuple[str, str]:
        key, value = header.split(self.header_key_value_separator, 1)
        return (want_str(key.strip()), want_str(value.strip()))

    @abc.abstractmethod
    def response_to_bytes(self, response: Response) -> bytes:
        """Serialize HTTP response into byte string."""
        ...

    def _response_to_bytes(self, status: HTTPStatus, headers: Mapping[str, str], body: bytes) -> bytes:
        return self.content_separator.join([str(status).encode(), self.content_separator.join([self._headers_serialize(headers), body])])

    def _headers_serialize(self, headers: Mapping[str, str]) -> bytes:
        return self.header_separator.join((self.header_key_value_separator.join([k if isinstance(k, _bytes) else k.encode('ascii'), v if isinstance(v, _bytes) else v.encode('latin-1')]) for k, v in headers.items()))

    @abc.abstractmethod
    def route(self, pattern: str, handler: Callable[..., Any], cors_options: Optional[Dict[str, Any]] = None) -> None:
        """Add route for handler."""
        ...

    @abc.abstractmethod
    def add_static(self, prefix: str, path: Union[str, Path], **kwargs: Any) -> None:
        """Add static route."""
        ...

    @abc.abstractmethod
    async def read_request_content(self, request: 'Request') -> bytes:
        """Read HTTP body as bytes."""
        ...

    @abc.abstractmethod
    async def wsgi(self) -> Any:
        """WSGI entry point."""
        ...

    def add_view(self, view_cls: Type[View], *, prefix: str = '', cors_options: Optional[Dict[str, Any]] = None) -> View:
        """Add route for view."""
        view = view_cls(self.app, self)
        path = prefix.rstrip('/') + '/' + view.view_path.lstrip('/')
        self.route(path, view, cors_options)
        self.views[path] = view
        self.reverse_names[view.view_name] = path
        return view

    def url_for(self, view_name: str, **kwargs: Any) -> str:
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

    def _quote_for_url(self, value: str) -> str:
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
    def can_read_body(self) -> bool:
        """Return :const:`True` if the request has a body."""
        ...

    @abc.abstractmethod
    async def read(self) -> bytes:
        """Read post data as bytes."""
        ...

    @abc.abstractmethod
    async def text(self) -> str:
        """Read post data as text."""
        ...

    @abc.abstractmethod
    async def json(self) -> Any:
        """Read post data and deserialize as JSON."""
        ...

    @abc.abstractmethod
    async def post(self) -> Mapping[str, Any]:
        """Read post data."""
        ...

    @property
    @abc.abstractmethod
    def match_info(self) -> Mapping[str, str]:
        """Return match info from URL route as a mapping."""
        ...

    @property
    @abc.abstractmethod
    def query(self) -> Mapping[str, str]:
        """Return HTTP query parameters as a mapping."""
        ...

    @property
    @abc.abstractmethod
    def cookies(self) -> Mapping[str, str]:
        """Return cookies as a mapping."""
        ...
