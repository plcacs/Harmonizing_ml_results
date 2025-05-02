import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
    TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, List, Mapping,
    NamedTuple, Optional, Sequence, Set, Tuple, Type, Union, cast
)
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
    ClientConnectionError, ClientOSError, ClientResponseError,
    ContentTypeError, InvalidURL, ServerFingerprintMismatch
)
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .hdrs import CONTENT_TYPE
from .helpers import (
    _SENTINEL, BaseTimerContext, BasicAuth, HeadersMixin, TimerNoop,
    basicauth_from_netrc, frozen_dataclass_decorator, is_expected_content_type,
    netrc_from_env, parse_mimetype, reify, set_exception, set_result
)
from .http import SERVER_SOFTWARE, HttpVersion, HttpVersion10, HttpVersion11, StreamWriter
from .log import client_logger
from .streams import StreamReader
from .typedefs import DEFAULT_JSON_DECODER, JSONDecoder, LooseCookies, LooseHeaders, Query, RawHeaders

if TYPE_CHECKING:
    import ssl
    from ssl import SSLContext
    from .client import ClientSession
    from .connector import Connection
    from .tracing import Trace
else:
    try:
        import ssl
        from ssl import SSLContext
    except ImportError:
        ssl = None
        SSLContext = object  # type: ignore

__all__ = ('ClientRequest', 'ClientResponse', 'RequestInfo', 'Fingerprint')

_CONTAINS_CONTROL_CHAR_RE = re.compile("[^-!#$%&'*+.^_`|~0-9a-zA-Z]")

def _gen_default_accept_encoding() -> str:
    return 'gzip, deflate, br' if HAS_BROTLI else 'gzip, deflate'

@frozen_dataclass_decorator
class ContentDisposition:
    disposition_type: str
    parameters: Mapping[str, str]
    filename: Optional[str]

class _RequestInfo(NamedTuple):
    url: URL
    method: str
    headers: CIMultiDictProxy[str]
    real_url: URL

class RequestInfo(_RequestInfo):
    def __new__(
        cls,
        url: URL,
        method: str,
        headers: CIMultiDictProxy[str],
        real_url: Any = _SENTINEL
    ) -> _RequestInfo:
        """Create a new RequestInfo instance.

        For backwards compatibility, the real_url parameter is optional.
        """
        return tuple.__new__(cls, (url, method, headers, url if real_url is _SENTINEL else real_url))

class Fingerprint:
    HASHFUNC_BY_DIGESTLEN: Dict[int, Callable[..., Any]] = {16: md5, 20: sha1, 32: sha256}

    def __init__(self, fingerprint: bytes) -> None:
        digestlen = len(fingerprint)
        hashfunc = self.HASHFUNC_BY_DIGESTLEN.get(digestlen)
        if not hashfunc:
            raise ValueError('fingerprint has invalid length')
        elif hashfunc is md5 or hashfunc is sha1:
            raise ValueError('md5 and sha1 are insecure and not supported. Use sha256.')
        self._hashfunc = hashfunc
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> bytes:
        return self._fingerprint

    def check(self, transport: asyncio.Transport) -> None:
        if not transport.get_extra_info('sslcontext'):
            return
        sslobj = transport.get_extra_info('ssl_object')
        cert = sslobj.getpeercert(binary_form=True)
        got = self._hashfunc(cert).digest()
        if got != self._fingerprint:
            host, port, *_ = transport.get_extra_info('peername')
            raise ServerFingerprintMismatch(self._fingerprint, got, host, port)

if ssl is not None:
    SSL_ALLOWED_TYPES = (ssl.SSLContext, bool, Fingerprint)
else:
    SSL_ALLOWED_TYPES = (bool,)
_SSL_SCHEMES = frozenset(('https', 'wss'))

class ConnectionKey(NamedTuple):
    host: str
    port: Optional[int]
    is_ssl: bool
    ssl: Union[bool, SSLContext, Fingerprint, None]
    proxy: Optional[URL]
    proxy_auth: Optional[BasicAuth]
    proxy_headers_hash: Optional[int]

class ClientRequest:
    GET_METHODS: Set[str] = {hdrs.METH_GET, hdrs.METH_HEAD, hdrs.METH_OPTIONS, hdrs.METH_TRACE}
    POST_METHODS: Set[str] = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT}
    ALL_METHODS: Set[str] = GET_METHODS.union(POST_METHODS).union({hdrs.METH_DELETE})
    DEFAULT_HEADERS: Dict[str, str] = {
        hdrs.ACCEPT: '*/*',
        hdrs.ACCEPT_ENCODING: _gen_default_accept_encoding()
    }
    
    body: Union[bytes, bytearray, Tuple[Union[bytes, bytearray], ...], payload.Payload, None] = b''
    auth: Optional[BasicAuth] = None
    response: Optional['ClientResponse'] = None
    __writer: Optional[asyncio.Future[None]] = None
    _continue: Optional[asyncio.Future[bool]] = None
    _skip_auto_headers: Optional[CIMultiDict[str]] = None

    def __init__(
        self,
        method: str,
        url: URL,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[LooseHeaders] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        data: Any = None,
        cookies: Optional[LooseCookies] = None,
        auth: Optional[BasicAuth] = None,
        version: Union[str, HttpVersion] = http.HttpVersion11,
        compress: Union[bool, str] = False,
        chunked: Optional[bool] = None,
        expect100: bool = False,
        loop: asyncio.AbstractEventLoop,
        response_class: Optional[Type['ClientResponse']] = None,
        proxy: Optional[URL] = None,
        proxy_auth: Optional[BasicAuth] = None,
        timer: Optional[BaseTimerContext] = None,
        session: 'ClientSession',
        ssl: Union[bool, SSLContext, Fingerprint] = True,
        proxy_headers: Optional[LooseHeaders] = None,
        traces: Optional[List['Trace']] = None,
        trust_env: bool = False,
        server_hostname: Optional[str] = None
    ) -> None:
        if (match := _CONTAINS_CONTROL_CHAR_RE.search(method)):
            raise ValueError(f'Method cannot contain non-token characters {method!r} (found at least {match.group()!r})')
        assert type(url) is URL, url
        if proxy is not None:
            assert type(proxy) is URL, proxy
        if TYPE_CHECKING:
            assert session is not None
        self._session = session
        if params:
            url = url.extend_query(params)
        self.original_url = url
        self.url = url.with_fragment(None) if url.raw_fragment else url
        self.method = method.upper()
        self.chunked = chunked
        self.loop = loop
        self.length: Optional[int] = None
        if response_class is None:
            real_response_class = ClientResponse
        else:
            real_response_class = response_class
        self.response_class = real_response_class
        self._timer = timer if timer is not None else TimerNoop()
        self._ssl = ssl
        self.server_hostname = server_hostname
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))
        self.update_version(version)
        self.update_host(url)
        self.update_headers(headers)
        self.update_auto_headers(skip_auto_headers)
        self.update_cookies(cookies)
        self.update_content_encoding(data, compress)
        self.update_auth(auth, trust_env)
        self.update_proxy(proxy, proxy_auth, proxy_headers)
        self.update_body_from_data(data)
        if data is not None or self.method not in self.GET_METHODS:
            self.update_transfer_encoding()
        self.update_expect_continue(expect100)
        self._traces = [] if traces is None else traces

    def __reset_writer(self, _: Optional[asyncio.Future[None]] = None) -> None:
        self.__writer = None

    @property
    def skip_auto_headers(self) -> CIMultiDict[str]:
        return self._skip_auto_headers or CIMultiDict()

    @property
    def _writer(self) -> Optional[asyncio.Future[None]]:
        return self.__writer

    @_writer.setter
    def _writer(self, writer: Optional[asyncio.Future[None]]) -> None:
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        if writer is not None:
            writer.add_done_callback(self.__reset_writer)

    def is_ssl(self) -> bool:
        return self.url.scheme in _SSL_SCHEMES

    @property
    def ssl(self) -> Union[bool, SSLContext, Fingerprint]:
        return self._ssl

    @property
    def connection_key(self) -> ConnectionKey:
        if (proxy_headers := self.proxy_headers):
            h = hash(tuple(proxy_headers.items()))
        else:
            h = None
        url = self.url
        return tuple.__new__(ConnectionKey, (
            url.raw_host or '', url.port, url.scheme in _SSL_SCHEMES,
            self._ssl, self.proxy, self.proxy_auth, h
        ))

    @property
    def host(self) -> str:
        ret = self.url.raw_host
        assert ret is not None
        return ret

    @property
    def port(self) -> Optional[int]:
        return self.url.port

    @property
    def request_info(self) -> RequestInfo:
        headers = CIMultiDictProxy(self.headers)
        return tuple.__new__(RequestInfo, (self.url, self.method, headers, self.original_url))

    def update_host(self, url: URL) -> None:
        """Update destination host, port and connection type (ssl)."""
        if not url.raw_host:
            raise InvalidURL(url)
        if url.raw_user or url.raw_password:
            self.auth = helpers.BasicAuth(url.user or '', url.password or '')

    def update_version(self, version: Union[str, HttpVersion]) -> None:
        """Convert request version to two elements tuple.

        parser HTTP version '1.1' => (1, 1)
        """
        if isinstance(version, str):
            v = [part.strip() for part in version.split('.', 1)]
            try:
                version = http.HttpVersion(int(v[0]), int(v[1]))
            except ValueError:
                raise ValueError(f'Can not parse http version number: {version}') from None
        self.version = version

    def update_headers(self, headers: Optional[LooseHeaders]) -> None:
        """Update request headers."""
        self.headers = CIMultiDict()
        host = self.url.host_port_subcomponent
        assert host is not None
        self.headers[hdrs.HOST] = host
        if not headers:
            return
        if isinstance(headers, (dict, MultiDictProxy, MultiDict)):
            headers = headers.items()
        for key, value in headers:
            if key in hdrs.HOST_ALL:
                self.headers[key] = value
            else:
                self.headers.add(key, value)

    def update_auto_headers(self, skip_auto_headers: Optional[Iterable[str]]) -> None:
        if skip_auto_headers is not None:
            self._skip_auto_headers = CIMultiDict(((hdr, None) for hdr in sorted(skip_auto_headers)))
            used_headers = self.headers.copy()
            used_headers.extend(self._skip_auto_headers)
        else:
            used_headers = self.headers
        for hdr, val in self.DEFAULT_HEADERS.items():
            if hdr not in used_headers:
                self.headers[hdr] = val
        if hdrs.USER_AGENT not in used_headers:
            self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE

    def update_cookies(self, cookies: Optional[LooseCookies]) -> None:
        """Update request cookies header."""
        if not cookies:
            return
        c = SimpleCookie()
        if hdrs.COOKIE in self.headers:
            c.load(self.headers.get(hdrs.COOKIE, ''))
            del self.headers[hdrs.COOKIE]
        if isinstance(cookies, Mapping):
            iter_cookies = cookies.items()
        else:
            iter_cookies = cookies
        for name, value in iter_cookies:
            if isinstance(value, Morsel):
                mrsl_val = value.get(value.key, Morsel())
                mrsl_val.set(value.key, value.value, value.coded_value)
                c[name] = mrsl_val
            else:
                c[name] = value
        self.headers[hdrs.COOKIE] = c.output(header='', sep=';').strip()

    def update_content_encoding(self, data: Any, compress: Union[bool, str]) -> None:
        """Set request content encoding."""
        self.compress: Optional[Union[bool, str]] = None
        if not data:
            return
        if self.headers.get(hdrs.CONTENT_ENCODING):
            if compress:
                raise ValueError('compress can not be set if Content-Encoding header is set')
        elif compress:
            self.compress = compress if isinstance(compress, str) else 'deflate'
            self.headers[hdrs.CONTENT_ENCODING] = self.compress
            self.chunked = True

    def update_transfer_encoding(self) -> None:
        """Analyze transfer-encoding header."""
        te = self.headers.get(hdrs.TRANSFER_ENCODING, '').lower()
        if 'chunked' in te:
            if self.chunked:
                raise ValueError('chunked can not be set if "Transfer-Encoding: chunked" header is set')
        elif self.chunked:
            if hdrs.CONTENT_LENGTH in self.headers:
                raise ValueError('chunked can not be set if Content-Length header is set')
            self.headers[hdrs.TRANSFER_ENCODING] = 'chunked'
        elif hdrs.CONTENT_LENGTH not in self.headers:
            self.headers[hdrs.CONTENT_LENGTH] = str(len(self.body))

    def update_auth(self, auth: Optional[BasicAuth], trust_env: bool) -> None:
        """Set basic auth."""
        if auth is None:
            auth = self.auth
        if auth is None and trust_env and (self.url.host is not None):
            netrc_obj = netrc_from_env()
            with contextlib.suppress(LookupError):
                auth = basicauth_from_netrc(netrc_obj, self.url.host)
        if auth is None:
            return
        if not isinstance(auth, helpers.BasicAuth):
            raise TypeError('BasicAuth() tuple is required instead')
        self.headers[hdrs.AUTHORIZATION] = auth.encode()

    def update_body_from_data(self, body: Any) -> None:
        if body is None:
            return
        if isinstance(body, FormData):
            body = body()
        try:
            body = payload.PAYLOAD_REGISTRY.get(body, disposition=None)
        except payload.LookupError:
            boundary = None
            if CONTENT_TYPE in self.headers:
                boundary = parse_mimetype(self.headers[CONTENT_TYPE]).parameters.get('boundary')
            body = FormData(body, boundary=boundary)()
        self.body = body
        if not self.chunked and hdrs.CONTENT_LENGTH not in self.headers:
            if (size := body.size) is not None:
                self.headers[hdrs.CONTENT_LENGTH] = str(size)
            else:
                self.chunked = True
        assert body.headers
        headers = self.headers
        skip_headers = self._skip_auto_headers
        for key, value in body.headers.items():
            if key in headers or (skip_headers is not None and key in skip_headers):
                continue
            headers[key] = value

    def update_expect_continue(self, expect: bool) -> None:
        if expect:
            self.headers[hdrs.EXPECT] = '100-continue'
        elif hdrs.EXPECT in self.headers and self.headers[hdrs.EXPECT].lower() == '100-continue':
            expect = True
        if expect:
            self._continue = self.loop.create_future()

    def update_proxy(
        self,
        proxy: Optional[URL],
        proxy_auth: Optional[BasicAuth],
        proxy_headers: Optional[LooseHeaders]
    ) -> None:
        self.proxy = proxy
        if proxy is None:
            self.proxy_auth = None
            self.proxy_headers = None
            return
        if proxy_auth and (not