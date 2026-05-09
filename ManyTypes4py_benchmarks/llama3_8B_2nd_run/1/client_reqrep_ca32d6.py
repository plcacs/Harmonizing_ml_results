from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Type, Union
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import ClientConnectionError, ClientOSError, ClientResponseError, ContentTypeError, InvalidURL, ServerFingerprintMismatch
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .hdrs import CONTENT_TYPE
from .helpers import _SENTINEL, BaseTimerContext, BasicAuth, HeadersMixin, TimerNoop, basicauth_from_netrc, frozen_dataclass_decorator, is_expected_content_type, netrc_from_env, parse_mimetype, reify, set_exception, set_result
from .http import SERVER_SOFTWARE, HttpVersion, HttpVersion10, HttpVersion11, StreamWriter
from .log import client_logger
from .streams import StreamReader
from .typedefs import DEFAULT_JSON_DECODER, JSONDecoder, LooseCookies, LooseHeaders, Query, RawHeaders
if TYPE_CHECKING:
    import ssl
    from ssl import SSLContext
else:
    try:
        import ssl
        from ssl import SSLContext
    except ImportError:
        ssl = None
        SSLContext = object

class ClientRequest:
    GET_METHODS: Tuple[str, ...]
    POST_METHODS: Tuple[str, ...]
    ALL_METHODS: Tuple[str, ...]
    DEFAULT_HEADERS: Dict[str, str]
    body: bytes
    auth: Optional[BasicAuth]
    response: Optional['ClientResponse']
    __writer: Optional[AbstractStreamWriter]
    _continue: Optional[asyncio.Future]
    _skip_auto_headers: Optional[CIMultiDict]

    def __init__(self, method: str, url: URL, *, params: Optional[Dict[str, str]], headers: Optional[Dict[str, str]], skip_auto_headers: Optional[Iterable[str]], data: Optional[bytes], cookies: Optional[Dict[str, str]], auth: Optional[BasicAuth], version: HttpVersion, compress: Optional[bool], chunked: Optional[bool], expect100: bool, loop: asyncio.BaseEventLoop, response_class: Type['ClientResponse'], proxy: Optional[URL], proxy_auth: Optional[BasicAuth], proxy_headers: Optional[Dict[str, str]], timer: Optional[BaseTimerContext], session: Optional['ClientSession'], ssl: bool, server_hostname: Optional[str]) -> None

    @property
    def is_ssl(self) -> bool

    @property
    def ssl(self) -> bool

    @property
    def connection_key(self) -> ConnectionKey

    @property
    def host(self) -> str

    @property
    def port(self) -> int

    @property
    def request_info(self) -> RequestInfo

    async def send(self, conn: StreamReader) -> 'ClientResponse'

    async def close(self) -> None

    def terminate(self) -> None

    async def write_bytes(self, writer: StreamWriter, conn: StreamReader) -> None

class ClientResponse(HeadersMixin):
    version: HttpVersion
    status: int
    reason: str
    content: Optional[bytes]
    _body: Optional[bytes]
    _headers: Dict[str, str]
    _raw_headers: Dict[str, str]
    _history: Tuple['ClientResponse', ...]
    _cookies: Optional[LooseCookies]
    _continue: Optional[asyncio.Future]
    _source_traceback: Optional[traceback.FrameSummary]
    _session: Optional['ClientSession']
    _closed: bool
    _released: bool
    _in_context: bool
    __writer: Optional[AbstractStreamWriter]

    @property
    def connection(self) -> Optional[StreamReader]

    @property
    def history(self) -> Tuple['ClientResponse', ...]

    @property
    def links(self) -> MultiDictProxy

    async def start(self, connection: StreamReader) -> None

    def raise_for_status(self) -> None

    async def read(self) -> bytes

    async def text(self, encoding: Optional[str], errors: str) -> str

    async def json(self, *, encoding: Optional[str], loads: Callable[[bytes], Any], content_type: str = 'application/json') -> Any

    async def __aenter__(self) -> 'ClientResponse'

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[traceback.Traceback]) -> None
