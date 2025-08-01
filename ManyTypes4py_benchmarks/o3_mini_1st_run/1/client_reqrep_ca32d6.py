#!/usr/bin/env python3
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
from typing import (Any, Callable, Dict, Iterable, List, Mapping, NamedTuple,
                    Optional, Tuple, Type, Union, TYPE_CHECKING, Awaitable)
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (ClientConnectionError, ClientOSError,
                                ClientResponseError, ContentTypeError,
                                InvalidURL, ServerFingerprintMismatch)
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .hdrs import CONTENT_TYPE
from .helpers import (_SENTINEL, BaseTimerContext, BasicAuth, HeadersMixin, TimerNoop,
                      basicauth_from_netrc, frozen_dataclass_decorator, is_expected_content_type,
                      netrc_from_env, parse_mimetype, reify, set_exception, set_result)
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
        ssl = None  # type: ignore
        SSLContext = object

__all__ = ('ClientRequest', 'ClientResponse', 'RequestInfo', 'Fingerprint')

_CONTAINS_CONTROL_CHAR_RE = re.compile("[^-!#$%&'*+.^_`|~0-9a-zA-Z]")


def _gen_default_accept_encoding() -> str:
    return 'gzip, deflate, br' if HAS_BROTLI else 'gzip, deflate'


@frozen_dataclass_decorator
class ContentDisposition:
    # Placeholder for content disposition attributes
    pass


class _RequestInfo(NamedTuple):
    # Placeholder for internal RequestInfo fields
    pass


class RequestInfo(_RequestInfo):
    def __new__(cls, url: URL, method: str, headers: CIMultiDictProxy, real_url: Any = _SENTINEL) -> "RequestInfo":
        """Create a new RequestInfo instance.

        For backwards compatibility, the real_url parameter is optional.
        """
        return tuple.__new__(cls, (url, method, headers, url if real_url is _SENTINEL else real_url))


class Fingerprint:
    HASHFUNC_BY_DIGESTLEN: Dict[int, Callable[[bytes], Any]] = {16: md5, 20: sha1, 32: sha256}

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

    def check(self, transport: Any) -> None:
        if not transport.get_extra_info('sslcontext'):
            return
        sslobj = transport.get_extra_info('ssl_object')
        cert = sslobj.getpeercert(binary_form=True)
        got = self._hashfunc(cert).digest()
        if got != self._fingerprint:
            host, port, *_ = transport.get_extra_info('peername')
            raise ServerFingerprintMismatch(self._fingerprint, got, host, port)


if ssl is not None:
    SSL_ALLOWED_TYPES: Tuple[Union[Type[SSLContext], Type[bool], Type[Fingerprint]], ...] = (SSLContext, bool, Fingerprint)
else:
    SSL_ALLOWED_TYPES = (bool,)


_SSL_SCHEMES = frozenset(('https', 'wss'))


class ConnectionKey(NamedTuple):
    # Placeholder for connection key fields
    pass


class ClientRequest:
    GET_METHODS: set = {hdrs.METH_GET, hdrs.METH_HEAD, hdrs.METH_OPTIONS, hdrs.METH_TRACE}
    POST_METHODS: set = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT}
    ALL_METHODS: set = GET_METHODS.union(POST_METHODS).union({hdrs.METH_DELETE})
    DEFAULT_HEADERS: Dict[str, str] = {hdrs.ACCEPT: '*/*', hdrs.ACCEPT_ENCODING: _gen_default_accept_encoding()}
    body: bytes = b''
    auth: Optional[BasicAuth] = None
    response: Optional["ClientResponse"] = None
    __writer: Optional[asyncio.Task] = None
    _continue: Optional[asyncio.Future] = None
    _skip_auto_headers: Optional[CIMultiDict] = None

    def __init__(self,
                 method: str,
                 url: URL,
                 *,
                 params: Optional[Mapping[str, Any]] = None,
                 headers: Optional[Union[Mapping[str, str], MultiDict, MultiDictProxy]] = None,
                 skip_auto_headers: Optional[Iterable[str]] = None,
                 data: Any = None,
                 cookies: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None,
                 auth: Optional[BasicAuth] = None,
                 version: HttpVersion = HttpVersion11,
                 compress: Union[bool, str] = False,
                 chunked: Optional[bool] = None,
                 expect100: bool = False,
                 *,
                 loop: asyncio.AbstractEventLoop,
                 response_class: Optional[Type[Any]] = None,
                 proxy: Optional[URL] = None,
                 proxy_auth: Optional[BasicAuth] = None,
                 timer: Optional[BaseTimerContext] = None,
                 session: Optional["ClientSession"] = None,
                 ssl: Union[bool, "SSLContext", Fingerprint] = True,
                 proxy_headers: Optional[Union[MultiDict, MultiDictProxy]] = None,
                 traces: Optional[List[Any]] = None,
                 trust_env: bool = False,
                 server_hostname: Optional[str] = None) -> None:
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
        self.original_url: URL = url
        self.url: URL = url.with_fragment(None) if url.raw_fragment else url
        self.method: str = method.upper()
        self.chunked: Optional[bool] = chunked
        self.loop: asyncio.AbstractEventLoop = loop
        self.length: Optional[int] = None
        if response_class is None:
            real_response_class = ClientResponse
        else:
            real_response_class = response_class
        self.response_class: Type[Any] = real_response_class
        self._timer: BaseTimerContext = timer if timer is not None else TimerNoop()
        self._ssl: Union[bool, "SSLContext", Fingerprint] = ssl
        self.server_hostname: Optional[str] = server_hostname
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
        self._traces: List[Any] = [] if traces is None else traces

    def __reset_writer(self, _: Any = None) -> None:
        self.__writer = None

    @property
    def skip_auto_headers(self) -> CIMultiDict:
        return self._skip_auto_headers or CIMultiDict()

    @property
    def _writer(self) -> Optional[asyncio.Task]:
        return self.__writer

    @_writer.setter
    def _writer(self, writer: asyncio.Task) -> None:
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        writer.add_done_callback(self.__reset_writer)

    def is_ssl(self) -> bool:
        return self.url.scheme in _SSL_SCHEMES

    @property
    def ssl(self) -> Union[bool, "SSLContext", Fingerprint]:
        return self._ssl

    @property
    def connection_key(self) -> ConnectionKey:
        if (proxy_headers := self.proxy_headers):
            h = hash(tuple(proxy_headers.items()))
        else:
            h = None
        url = self.url
        return tuple.__new__(ConnectionKey, (url.raw_host or '', url.port, url.scheme in _SSL_SCHEMES, self._ssl, self.proxy, self.proxy_auth, h))

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
        headers: CIMultiDictProxy = CIMultiDictProxy(self.headers)
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

    def update_headers(self, headers: Optional[Union[Mapping[str, str], MultiDictProxy, MultiDict]]) -> None:
        """Update request headers."""
        self.headers: CIMultiDict = CIMultiDict()
        host: str = self.url.host_port_subcomponent  # type: ignore
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
            used_headers: CIMultiDict = self.headers.copy()
            used_headers.extend(self._skip_auto_headers)
        else:
            used_headers = self.headers
        for hdr, val in self.DEFAULT_HEADERS.items():
            if hdr not in used_headers:
                self.headers[hdr] = val
        if hdrs.USER_AGENT not in used_headers:
            self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE

    def update_cookies(self, cookies: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]]) -> None:
        """Update request cookies header."""
        if not cookies:
            return
        c: SimpleCookie = SimpleCookie()
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
        self.compress: Optional[str] = None
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
        te: str = self.headers.get(hdrs.TRANSFER_ENCODING, '').lower()
        if 'chunked' in te:
            if self.chunked:
                raise ValueError('chunked can not be set if "Transfer-Encoding: chunked" header is set')
        elif self.chunked:
            if hdrs.CONTENT_LENGTH in self.headers:
                raise ValueError('chunked can not be set if Content-Length header is set')
            self.headers[hdrs.TRANSFER_ENCODING] = 'chunked'
        elif hdrs.CONTENT_LENGTH not in self.headers:
            self.headers[hdrs.CONTENT_LENGTH] = str(len(self.body))

    def update_auth(self, auth: Optional[BasicAuth], trust_env: bool = False) -> None:
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
            boundary: Optional[str] = None
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

    def update_expect_continue(self, expect: bool = False) -> None:
        if expect:
            self.headers[hdrs.EXPECT] = '100-continue'
        elif hdrs.EXPECT in self.headers and self.headers[hdrs.EXPECT].lower() == '100-continue':
            expect = True
        if expect:
            self._continue = self.loop.create_future()

    def update_proxy(self,
                     proxy: Optional[URL],
                     proxy_auth: Optional[BasicAuth],
                     proxy_headers: Optional[Union[MultiDict, MultiDictProxy]]) -> None:
        self.proxy = proxy
        if proxy is None:
            self.proxy_auth = None
            self.proxy_headers = None
            return
        if proxy_auth and (not isinstance(proxy_auth, helpers.BasicAuth)):
            raise ValueError('proxy_auth must be None or BasicAuth() tuple')
        self.proxy_auth = proxy_auth
        if proxy_headers is not None and (not isinstance(proxy_headers, (MultiDict, MultiDictProxy))):
            proxy_headers = CIMultiDict(proxy_headers)
        self.proxy_headers = proxy_headers

    async def write_bytes(self, writer: StreamWriter, conn: Any) -> None:
        """Support coroutines that yield bytes objects."""
        if self._continue is not None:
            await writer.drain()
            await self._continue
        protocol = conn.protocol
        assert protocol is not None
        try:
            if isinstance(self.body, payload.Payload):
                await self.body.write(writer)
            else:
                if isinstance(self.body, (bytes, bytearray)):
                    self.body = (self.body,)
                for chunk in self.body:
                    await writer.write(chunk)
        except OSError as underlying_exc:
            reraised_exc = underlying_exc
            exc_is_not_timeout = underlying_exc.errno is not None or not isinstance(underlying_exc, asyncio.TimeoutError)
            if exc_is_not_timeout:
                reraised_exc = ClientOSError(underlying_exc.errno, f'Can not write request body for {self.url!s}')
            set_exception(protocol, reraised_exc, underlying_exc)
        except asyncio.CancelledError:
            conn.close()
            raise
        except Exception as underlying_exc:
            set_exception(protocol, ClientConnectionError(f'Failed to send bytes into the underlying connection {conn!s}'), underlying_exc)
        else:
            await writer.write_eof()
            protocol.start_timeout()

    async def send(self, conn: Any) -> "ClientResponse":
        if self.method == hdrs.METH_CONNECT:
            connect_host: str = self.url.host_subcomponent  # type: ignore
            assert connect_host is not None
            path = f'{connect_host}:{self.url.port}'
        elif self.proxy and (not self.is_ssl()):
            path = str(self.url)
        else:
            path = self.url.raw_path_qs
        protocol = conn.protocol
        assert protocol is not None
        writer: StreamWriter = StreamWriter(
            protocol,
            self.loop,
            on_chunk_sent=functools.partial(self._on_chunk_request_sent, self.method, self.url) if self._traces else None,
            on_headers_sent=functools.partial(self._on_headers_request_sent, self.method, self.url) if self._traces else None
        )
        if self.compress:
            writer.enable_compression(self.compress)
        if self.chunked is not None:
            writer.enable_chunking()
        if self.method in self.POST_METHODS and (self._skip_auto_headers is None or hdrs.CONTENT_TYPE not in self._skip_auto_headers) and (hdrs.CONTENT_TYPE not in self.headers):
            self.headers[hdrs.CONTENT_TYPE] = 'application/octet-stream'
        v = self.version
        if hdrs.CONNECTION not in self.headers:
            if conn._connector.force_close:
                if v == HttpVersion11:
                    self.headers[hdrs.CONNECTION] = 'close'
            elif v == HttpVersion10:
                self.headers[hdrs.CONNECTION] = 'keep-alive'
        status_line: str = f'{self.method} {path} HTTP/{v.major}.{v.minor}'
        await writer.write_headers(status_line, self.headers)
        if self.body or self._continue is not None or protocol.writing_paused:
            coro = self.write_bytes(writer, conn)
            if sys.version_info >= (3, 12):
                task = asyncio.Task(coro, loop=self.loop, eager_start=True)
            else:
                task = self.loop.create_task(coro)
            if task.done():
                task = None
            else:
                self._writer = task
        else:
            protocol.start_timeout()
            writer.set_eof()
            task = None
        response_class = self.response_class
        assert response_class is not None
        self.response = response_class(self.method,
                                       self.original_url,
                                       writer=task,
                                       continue100=self._continue,
                                       timer=self._timer,
                                       request_info=self.request_info,
                                       traces=self._traces,
                                       loop=self.loop,
                                       session=self._session)
        return self.response

    async def close(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if sys.version_info >= (3, 11) and (task := asyncio.current_task()) and task.cancelling():
                    raise

    def terminate(self) -> None:
        if self.__writer is not None:
            if not self.loop.is_closed():
                self.__writer.cancel()
            self.__writer.remove_done_callback(self.__reset_writer)
            self.__writer = None

    async def _on_chunk_request_sent(self, method: str, url: URL, chunk: bytes) -> None:
        for trace in self._traces:
            await trace.send_request_chunk_sent(method, url, chunk)

    async def _on_headers_request_sent(self, method: str, url: URL, headers: Any) -> None:
        for trace in self._traces:
            await trace.send_request_headers(method, url, headers)


_CONNECTION_CLOSED_EXCEPTION = ClientConnectionError('Connection closed')


class ClientResponse(HeadersMixin):
    version: Optional[HttpVersion] = None
    status: Optional[int] = None
    reason: Optional[str] = None
    content: Optional[StreamReader] = None
    _body: Optional[bytes] = None
    _headers: Optional[CIMultiDict] = None
    _history: Tuple[Any, ...] = ()
    _raw_headers: Optional[RawHeaders] = None
    _connection: Optional[Any] = None
    _cookies: Optional[SimpleCookie] = None
    _continue: Optional[asyncio.Future] = None
    _source_traceback: Optional[Any] = None
    _session: Optional["ClientSession"] = None
    _closed: bool = True
    _released: bool = False
    _in_context: bool = False
    _resolve_charset: Callable[..., str] = lambda *_: 'utf-8'
    __writer: Optional[asyncio.Task] = None

    def __init__(self,
                 method: str,
                 url: URL,
                 *,
                 writer: Optional[asyncio.Task],
                 continue100: Optional[asyncio.Future],
                 timer: Optional[BaseTimerContext],
                 request_info: RequestInfo,
                 traces: List[Any],
                 loop: asyncio.AbstractEventLoop,
                 session: Optional["ClientSession"]) -> None:
        assert type(url) is URL
        self.method: str = method
        self._real_url: URL = url
        self._url: URL = url.with_fragment(None) if url.raw_fragment else url
        if writer is not None:
            self._writer = writer
        if continue100 is not None:
            self._continue = continue100
        self._request_info: RequestInfo = request_info
        self._timer: BaseTimerContext = timer if timer is not None else TimerNoop()
        self._cache: Dict[Any, Any] = {}
        self._traces: List[Any] = traces
        self._loop: asyncio.AbstractEventLoop = loop
        if session is not None:
            self._session = session
            self._resolve_charset = session._resolve_charset
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

    def __reset_writer(self, _: Any = None) -> None:
        self.__writer = None

    @property
    def _writer(self) -> Optional[asyncio.Task]:
        """The writer task for streaming data.

        _writer is only provided for backwards compatibility
        for subclasses that may need to access it.
        """
        return self.__writer

    @_writer.setter
    def _writer(self, writer: asyncio.Task) -> None:
        """Set the writer task for streaming data."""
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        if writer is None:
            return
        if writer.done():
            self.__writer = None
        else:
            writer.add_done_callback(self.__reset_writer)

    @property
    def cookies(self) -> SimpleCookie:
        if self._cookies is None:
            self._cookies = SimpleCookie()
        return self._cookies

    @cookies.setter
    def cookies(self, cookies: SimpleCookie) -> None:
        self._cookies = cookies

    @reify
    def url(self) -> URL:
        return self._url

    @reify
    def real_url(self) -> URL:
        return self._real_url

    @reify
    def host(self) -> str:
        assert self._url.host is not None
        return self._url.host

    @reify
    def headers(self) -> CIMultiDict:
        return self._headers  # type: ignore

    @reify
    def raw_headers(self) -> RawHeaders:
        return self._raw_headers  # type: ignore

    @reify
    def request_info(self) -> RequestInfo:
        return self._request_info

    @reify
    def content_disposition(self) -> Optional[ContentDisposition]:
        raw: Optional[str] = self._headers.get(hdrs.CONTENT_DISPOSITION) if self._headers else None
        if raw is None:
            return None
        disposition_type, params_dct = multipart.parse_content_disposition(raw)
        params = MappingProxyType(params_dct)
        filename = multipart.content_disposition_filename(params)
        return ContentDisposition(disposition_type, params, filename)

    def __del__(self) -> None:
        if self._closed:
            return
        if self._connection is not None:
            self._connection.release()
            self._cleanup_writer()
            if self._loop.get_debug():
                warnings.warn(f'Unclosed response {self!r}', ResourceWarning, source=self)
                context: Dict[str, Any] = {'client_response': self, 'message': 'Unclosed response'}
                if self._source_traceback:
                    context['source_traceback'] = self._source_traceback
                self._loop.call_exception_handler(context)

    def __repr__(self) -> str:
        out: io.StringIO = io.StringIO()
        ascii_encodable_url: str = str(self.url)
        if self.reason:
            ascii_encodable_reason: str = self.reason.encode('ascii', 'backslashreplace').decode('ascii')
        else:
            ascii_encodable_reason = 'None'
        print('<ClientResponse({}) [{} {}]>'.format(ascii_encodable_url, self.status, ascii_encodable_reason), file=out)
        print(self.headers, file=out)
        return out.getvalue()

    @property
    def connection(self) -> Optional[Any]:
        return self._connection

    @reify
    def history(self) -> Tuple[Any, ...]:
        """A sequence of responses, if redirects occurred."""
        return self._history

    @reify
    def links(self) -> MultiDictProxy:
        links_str: str = ', '.join(self.headers.getall('link', []))
        if not links_str:
            return MultiDictProxy(MultiDict())
        links = MultiDict()
        for val in re.split(',(?=\\s*<)', links_str):
            match = re.match('\\s*<(.*)>(.*)', val)
            if match is None:
                continue
            url, params_str = match.groups()
            params = params_str.split(';')[1:]
            link = MultiDict()
            for param in params:
                match = re.match('^\\s*(\\S*)\\s*=\\s*([\'\\"]?)(.*?)(\\2)\\s*$', param, re.M)
                if match is None:
                    continue
                key, _, value, _ = match.groups()
                link.add(key, value)
            key = link.get('rel', url)
            link.add('url', self.url.join(URL(url)))
            links.add(str(key), MultiDictProxy(link))
        return MultiDictProxy(links)

    async def start(self, connection: Any) -> "ClientResponse":
        """Start response processing."""
        self._closed = False
        self._protocol = connection.protocol
        self._connection = connection
        with self._timer:
            while True:
                try:
                    protocol = self._protocol
                    message, payload = await protocol.read()
                except http.HttpProcessingError as exc:
                    raise ClientResponseError(self.request_info, self.history, status=exc.code, message=exc.message, headers=exc.headers) from exc
                if message.code < 100 or message.code > 199 or message.code == 101:
                    break
                if self._continue is not None:
                    set_result(self._continue, True)
                    self._continue = None
        payload.on_eof(self._response_eof)
        self.version = message.version
        self.status = message.code
        self.reason = message.reason
        self._headers = message.headers
        self._raw_headers = message.raw_headers
        self.content = payload
        if (cookie_hdrs := self.headers.getall(hdrs.SET_COOKIE, ())):
            cookies = SimpleCookie()
            for hdr in cookie_hdrs:
                try:
                    cookies.load(hdr)
                except CookieError as exc:
                    client_logger.warning('Can not load response cookies: %s', exc)
            self._cookies = cookies
        return self

    def _response_eof(self) -> None:
        if self._closed:
            return
        protocol = self._connection and self._connection.protocol
        if protocol is not None and protocol.upgraded:
            return
        self._closed = True
        self._cleanup_writer()
        self._release_connection()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._released:
            self._notify_content()
        self._closed = True
        if self._loop.is_closed():
            return
        self._cleanup_writer()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def release(self) -> None:
        if not self._released:
            self._notify_content()
        self._closed = True
        self._cleanup_writer()
        self._release_connection()

    @property
    def ok(self) -> bool:
        """Returns ``True`` if ``status`` is less than ``400``, ``False`` if not.

        This is **not** a check for ``200 OK`` but a check that the response
        status is under 400.
        """
        return 400 > self.status  # type: ignore

    def raise_for_status(self) -> None:
        if not self.ok:
            assert self.reason is not None
            if not self._in_context:
                self.release()
            raise ClientResponseError(self.request_info, self.history, status=self.status, message=self.reason, headers=self.headers)

    def _release_connection(self) -> None:
        if self._connection is not None:
            if self.__writer is None:
                self._connection.release()
                self._connection = None
            else:
                self.__writer.add_done_callback(lambda f: self._release_connection())

    async def _wait_released(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if sys.version_info >= (3, 11) and (task := asyncio.current_task()) and task.cancelling():
                    raise
        self._release_connection()

    def _cleanup_writer(self) -> None:
        if self.__writer is not None:
            self.__writer.cancel()
        self._session = None

    def _notify_content(self) -> None:
        content = self.content
        if content and content.exception() is None:
            set_exception(content, _CONNECTION_CLOSED_EXCEPTION)
        self._released = True

    async def wait_for_close(self) -> None:
        if self.__writer is not None:
            try:
                await self.__writer
            except asyncio.CancelledError:
                if sys.version_info >= (3, 11) and (task := asyncio.current_task()) and task.cancelling():
                    raise
        self.release()

    async def read(self) -> bytes:
        """Read response payload."""
        if self._body is None:
            try:
                self._body = await self.content.read()  # type: ignore
                for trace in self._traces:
                    await trace.send_response_chunk_received(self.method, self.url, self._body)
            except BaseException:
                self.close()
                raise
        elif self._released:
            raise ClientConnectionError('Connection closed')
        protocol = self._connection and self._connection.protocol
        if protocol is None or not protocol.upgraded:
            await self._wait_released()
        return self._body  # type: ignore

    def get_encoding(self) -> str:
        ctype: str = self.headers.get(hdrs.CONTENT_TYPE, '').lower()
        mimetype = helpers.parse_mimetype(ctype)
        encoding: Optional[str] = mimetype.parameters.get('charset')
        if encoding:
            with contextlib.suppress(LookupError, ValueError):
                return codecs.lookup(encoding).name
        if mimetype.type == 'application' and (mimetype.subtype == 'json' or mimetype.subtype == 'rdap'):
            return 'utf-8'
        if self._body is None:
            raise RuntimeError('Cannot compute fallback encoding of a not yet read body')
        return self._resolve_charset(self, self._body)  # type: ignore

    async def text(self, encoding: Optional[str] = None, errors: str = 'strict') -> str:
        """Read response payload and decode."""
        await self.read()
        if encoding is None:
            encoding = self.get_encoding()
        return self._body.decode(encoding, errors=errors)  # type: ignore

    async def json(self, *, encoding: Optional[str] = None, loads: Callable[[str], Any] = DEFAULT_JSON_DECODER, content_type: str = 'application/json') -> Any:
        """Read and decodes JSON response."""
        await self.read()
        if content_type:
            if not is_expected_content_type(self.content_type, content_type):  # type: ignore
                raise ContentTypeError(self.request_info, self.history, status=self.status, message='Attempt to decode JSON with unexpected mimetype: %s' % self.content_type, headers=self.headers)
        if encoding is None:
            encoding = self.get_encoding()
        return loads(self._body.decode(encoding))  # type: ignore

    async def __aenter__(self) -> "ClientResponse":
        self._in_context = True
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        self._in_context = False
        self.release()
        await self.wait_for_close()