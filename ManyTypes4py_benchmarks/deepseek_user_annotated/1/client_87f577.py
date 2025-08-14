"""HTTP Client for asyncio."""

import asyncio
import base64
import dataclasses
import hashlib
import json
import os
import sys
import traceback
import warnings
from contextlib import suppress
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Dict,
    Final,
    FrozenSet,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    final,
    cast,
)

from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from yarl import URL

from . import hdrs, http, payload
from ._websocket.reader import WebSocketDataQueue
from .abc import AbstractCookieJar
from .client_exceptions import (
    ClientConnectionError,
    ClientConnectionResetError,
    ClientConnectorCertificateError,
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientError,
    ClientHttpProxyError,
    ClientOSError,
    ClientPayloadError,
    ClientProxyConnectionError,
    ClientResponseError,
    ClientSSLError,
    ConnectionTimeoutError,
    ContentTypeError,
    InvalidURL,
    InvalidUrlClientError,
    InvalidUrlRedirectClientError,
    NonHttpUrlClientError,
    NonHttpUrlRedirectClientError,
    RedirectClientError,
    ServerConnectionError,
    ServerDisconnectedError,
    ServerFingerprintMismatch,
    ServerTimeoutError,
    SocketTimeoutError,
    TooManyRedirects,
    WSMessageTypeError,
    WSServerHandshakeError,
)
from .client_reqrep import (
    SSL_ALLOWED_TYPES,
    ClientRequest,
    ClientResponse,
    Fingerprint,
    RequestInfo,
)
from .client_ws import (
    DEFAULT_WS_CLIENT_TIMEOUT,
    ClientWebSocketResponse,
    ClientWSTimeout,
)
from .connector import (
    HTTP_AND_EMPTY_SCHEMA_SET,
    BaseConnector,
    NamedPipeConnector,
    TCPConnector,
    UnixConnector,
)
from .cookiejar import CookieJar
from .helpers import (
    _SENTINEL,
    EMPTY_BODY_METHODS,
    BasicAuth,
    TimeoutHandle,
    frozen_dataclass_decorator,
    get_env_proxy_for_url,
    sentinel,
    strip_auth_from_url,
)
from .http import WS_KEY, HttpVersion, WebSocketReader, WebSocketWriter
from .http_websocket import WSHandshakeError, ws_ext_gen, ws_ext_parse
from .tracing import Trace, TraceConfig
from .typedefs import JSONEncoder, LooseCookies, LooseHeaders, Query, StrOrURL

__all__ = (
    # client_exceptions
    "ClientConnectionError",
    "ClientConnectionResetError",
    "ClientConnectorCertificateError",
    "ClientConnectorDNSError",
    "ClientConnectorError",
    "ClientConnectorSSLError",
    "ClientError",
    "ClientHttpProxyError",
    "ClientOSError",
    "ClientPayloadError",
    "ClientProxyConnectionError",
    "ClientResponseError",
    "ClientSSLError",
    "ConnectionTimeoutError",
    "ContentTypeError",
    "InvalidURL",
    "InvalidUrlClientError",
    "RedirectClientError",
    "NonHttpUrlClientError",
    "InvalidUrlRedirectClientError",
    "NonHttpUrlRedirectClientError",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ServerTimeoutError",
    "SocketTimeoutError",
    "TooManyRedirects",
    "WSServerHandshakeError",
    # client_reqrep
    "ClientRequest",
    "ClientResponse",
    "Fingerprint",
    "RequestInfo",
    # connector
    "BaseConnector",
    "TCPConnector",
    "UnixConnector",
    "NamedPipeConnector",
    # client_ws
    "ClientWebSocketResponse",
    # client
    "ClientSession",
    "ClientTimeout",
    "ClientWSTimeout",
    "request",
    "WSMessageTypeError",
)


if TYPE_CHECKING:
    from ssl import SSLContext
else:
    SSLContext = None

if sys.version_info >= (3, 11) and TYPE_CHECKING:
    from typing import Unpack


class _RequestOptions(TypedDict, total=False):
    params: Query
    data: Any
    json: Any
    cookies: Union[LooseCookies, None]
    headers: Union[LooseHeaders, None]
    skip_auto_headers: Union[Iterable[str], None]
    auth: Union[BasicAuth, None]
    allow_redirects: bool
    max_redirects: int
    compress: Union[str, bool]
    chunked: Union[bool, None]
    expect100: bool
    raise_for_status: Union[None, bool, Callable[[ClientResponse], Awaitable[None]]]
    read_until_eof: bool
    proxy: Union[StrOrURL, None]
    proxy_auth: Union[BasicAuth, None]
    timeout: "Union[ClientTimeout, _SENTINEL, None]"
    ssl: Union[SSLContext, bool, Fingerprint]
    server_hostname: Union[str, None]
    proxy_headers: Union[LooseHeaders, None]
    trace_request_ctx: Union[Mapping[str, Any], None]
    read_bufsize: Union[int, None]
    auto_decompress: Union[bool, None]
    max_line_size: Union[int, None]
    max_field_size: Union[int, None]


@frozen_dataclass_decorator
@dataclasses.dataclass(frozen=True)
class ClientTimeout:
    total: Optional[float] = None
    connect: Optional[float] = None
    sock_read: Optional[float] = None
    sock_connect: Optional[float] = None
    ceil_threshold: float = 5


DEFAULT_TIMEOUT: Final[ClientTimeout] = ClientTimeout(total=5 * 60, sock_connect=30)

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"})

_RetType = TypeVar("_RetType", ClientResponse, ClientWebSocketResponse)
_CharsetResolver = Callable[[ClientResponse, bytes], str]


@final
class ClientSession:
    __slots__ = (
        "_base_url",
        "_base_url_origin",
        "_source_traceback",
        "_connector",
        "_loop",
        "_cookie_jar",
        "_connector_owner",
        "_default_auth",
        "_version",
        "_json_serialize",
        "_requote_redirect_url",
        "_timeout",
        "_raise_for_status",
        "_auto_decompress",
        "_trust_env",
        "_default_headers",
        "_skip_auto_headers",
        "_request_class",
        "_response_class",
        "_ws_response_class",
        "_trace_configs",
        "_read_bufsize",
        "_max_line_size",
        "_max_field_size",
        "_resolve_charset",
        "_default_proxy",
        "_default_proxy_auth",
        "_retry_connection",
    )

    def __init__(
        self,
        base_url: Optional[StrOrURL] = None,
        *,
        connector: Optional[BaseConnector] = None,
        cookies: Optional[LooseCookies] = None,
        headers: Optional[LooseHeaders] = None,
        proxy: Optional[StrOrURL] = None,
        proxy_auth: Optional[BasicAuth] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[BasicAuth] = None,
        json_serialize: JSONEncoder = json.dumps,
        request_class: Type[ClientRequest] = ClientRequest,
        response_class: Type[ClientResponse] = ClientResponse,
        ws_response_class: Type[ClientWebSocketResponse] = ClientWebSocketResponse,
        version: HttpVersion = http.HttpVersion11,
        cookie_jar: Optional[AbstractCookieJar] = None,
        connector_owner: bool = True,
        raise_for_status: Union[
            bool, Callable[[ClientResponse], Awaitable[None]]
        ] = False,
        timeout: Union[_SENTINEL, ClientTimeout, None] = sentinel,
        auto_decompress: bool = True,
        trust_env: bool = False,
        requote_redirect_url: bool = True,
        trace_configs: Optional[List[TraceConfig[object]]] = None,
        read_bufsize: int = 2**16,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        fallback_charset_resolver: _CharsetResolver = lambda r, b: "utf-8",
    ) -> None:
        self._connector: Optional[BaseConnector] = None
        if base_url is None or isinstance(base_url, URL):
            self._base_url: Optional[URL] = base_url
            self._base_url_origin = None if base_url is None else base_url.origin()
        else:
            self._base_url = URL(base_url)
            self._base_url_origin = self._base_url.origin()
            assert self._base_url.absolute, "Only absolute URLs are supported"
        if self._base_url is not None and not self._base_url.path.endswith("/"):
            raise ValueError("base_url must have a trailing '/'")

        loop = asyncio.get_running_loop()

        if timeout is sentinel or timeout is None:
            timeout = DEFAULT_TIMEOUT
        if not isinstance(timeout, ClientTimeout):
            raise ValueError(
                f"timeout parameter cannot be of {type(timeout)} type, "
                "please use 'timeout=ClientTimeout(...)'",
            )
        self._timeout = timeout

        if connector is None:
            connector = TCPConnector()
        self._connector = connector
        self._loop = loop
        if loop.get_debug():
            self._source_traceback: Optional[traceback.StackSummary] = (
                traceback.extract_stack(sys._getframe(1))
            )
        else:
            self._source_traceback = None

        if connector._loop is not loop:
            raise RuntimeError("Session and connector have to use same event loop")

        if cookie_jar is None:
            cookie_jar = CookieJar()
        self._cookie_jar = cookie_jar

        if cookies:
            self._cookie_jar.update_cookies(cookies)

        self._connector_owner = connector_owner
        self._default_auth = auth
        self._version = version
        self._json_serialize = json_serialize
        self._raise_for_status = raise_for_status
        self._auto_decompress = auto_decompress
        self._trust_env = trust_env
        self._requote_redirect_url = requote_redirect_url
        self._read_bufsize = read_bufsize
        self._max_line_size = max_line_size
        self._max_field_size = max_field_size

        if headers:
            real_headers: CIMultiDict[str] = CIMultiDict(headers)
        else:
            real_headers = CIMultiDict()
        self._default_headers: CIMultiDict[str] = real_headers
        if skip_auto_headers is not None:
            self._skip_auto_headers = frozenset(istr(i) for i in skip_auto_headers)
        else:
            self._skip_auto_headers = frozenset()

        self._request_class = request_class
        self._response_class = response_class
        self._ws_response_class = ws_response_class

        self._trace_configs = trace_configs or []
        for trace_config in self._trace_configs:
            trace_config.freeze()

        self._resolve_charset = fallback_charset_resolver

        self._default_proxy = proxy
        self._default_proxy_auth = proxy_auth
        self._retry_connection: bool = True

    def __init_subclass__(cls: Type["ClientSession"]) -> None:
        raise TypeError(
            "Inheritance class {} from ClientSession "
            "is forbidden".format(cls.__name__)
        )

    def __del__(self, _warnings: Any = warnings) -> None:
        if not self.closed:
            _warnings.warn(
                f"Unclosed client session {self!r}",
                ResourceWarning,
                source=self,
            )
            context = {"client_session": self, "message": "Unclosed client session"}
            if self._source_traceback is not None:
                context["source_traceback"] = self._source_traceback
            self._loop.call_exception_handler(context)

    if sys.version_info >= (3, 11) and TYPE_CHECKING:
        def request(
            self,
            method: str,
            url: StrOrURL,
            **kwargs: Unpack[_RequestOptions],
        ) -> "_RequestContextManager": ...
    else:
        def request(
            self, method: str, url: StrOrURL, **kwargs: Any
        ) -> "_RequestContextManager":
            return _RequestContextManager(self._request(method, url, **kwargs))

    def _build_url(self, str_or_url: StrOrURL) -> URL:
        url = URL(str_or_url)
        if self._base_url and not url.absolute:
            return self._base_url.join(url)
        return url

    async def _request(
        self,
        method: str,
        str_or_url: StrOrURL,
        *,
        params: Query = None,
        data: Any = None,
        json: Any = None,
        cookies: Optional[LooseCookies] = None,
        headers: Optional[LooseHeaders] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[BasicAuth] = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        compress: Union[str, bool] = False,
        chunked: Optional[bool] = None,
        expect100: bool = False,
        raise_for_status: Union[
            None, bool, Callable[[ClientResponse], Awaitable[None]]
        ] = None,
        read_until_eof: bool = True,
        proxy: Optional[StrOrURL] = None,
        proxy_auth: Optional[BasicAuth] = None,
        timeout: Union[ClientTimeout, _SENTINEL, None] = sentinel,
        ssl: Union[SSLContext, bool, Fingerprint] = True,
        server_hostname: Optional[str] = None,
        proxy_headers: Optional[LooseHeaders] = None,
        trace_request_ctx: Optional[Mapping[str, Any]] = None,
        read_bufsize: Optional[int] = None,
        auto_decompress: Optional[bool] = None,
        max_line_size: Optional[int] = None,
        max_field_size: Optional[int] = None,
    ) -> ClientResponse:
        if self.closed:
            raise RuntimeError("Session is closed")

        if not isinstance(ssl, SSL_ALLOWED_TYPES):
            raise TypeError(
                "ssl should be SSLContext, Fingerprint, or bool, "
                "got {!r} instead.".format(ssl)
            )

        if data is not None and json is not None:
            raise ValueError(
                "data and json parameters can not be used at the same time"
            )
        elif json is not None:
            data = payload.JsonPayload(json, dumps=self._json_serialize)

        redirects = 0
        history: List[ClientResponse] = []
        version = self._version
        params = params or {}

        headers = self._prepare_headers(headers)

        try:
            url = self._build_url(str_or_url)
        except ValueError as e:
            raise InvalidUrlClientError(str_or_url) from e

        assert self._connector is not None
        if url.scheme not in self._connector.allowed_protocol_schema_set:
            raise NonHttpUrlClientError(url)

        skip_headers: Optional[Iterable[istr]]
        if skip_auto_headers is not None:
            skip_headers = {
                istr(i) for i in skip_auto_headers
            } | self._skip_auto_headers
        elif self._skip_auto_headers:
            skip_headers = self._skip_auto_headers
        else:
            skip_headers = None

        if proxy is None:
            proxy = self._default_proxy
        if proxy_auth is None:
            proxy_auth = self._default_proxy_auth

        if proxy is None:
            proxy_headers = None
        else:
            proxy_headers = self._prepare_headers(proxy_headers)
            try:
                proxy = URL(proxy)
            except ValueError as e:
                raise InvalidURL(proxy) from e

        if timeout is sentinel or timeout is None:
            real_timeout: ClientTimeout = self._timeout
        else:
            real_timeout = timeout
        tm = TimeoutHandle(
            self._loop, real_timeout.total, ceil_threshold=real_timeout.ceil_threshold
        )
        handle = tm.start()

        if read_bufsize is None:
            read_bufsize = self._read_bufsize

        if auto_decompress is None:
            auto_decompress = self._auto_decompress

        if max_line_size is None:
            max_line_size = self._max_line_size

        if max_field_size is None:
            max_field_size = self._max_field_size

        traces = [
            Trace(
                self,
                trace_config,
                trace_config.trace_config_ctx(trace_request_ctx=trace_request_ctx),
            )
            for trace_config in self._trace_configs
        ]

        for trace in traces:
            await trace.send_request_start(method, url.update_query(params), headers)

        timer = tm.timer()
        try:
            with timer:
                retry_persistent_connection = (
                    self._retry_connection and method in IDEMPOTENT_METHODS
                )
                while True:
                    url, auth_from_url = strip_auth_from_url(url)
                    if not url.raw_host:
                        err_exc_cls = (
                            InvalidUrlRedirectClientError
                            if redirects
                            else InvalidUrl