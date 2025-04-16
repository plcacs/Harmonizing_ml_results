```python
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
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL

from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
    ClientConnectionError,
    ClientOSError,
    ClientResponseError,
    ContentTypeError,
    InvalidURL,
    ServerFingerprintMismatch,
)
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .hdrs import CONTENT_TYPE
from .helpers import (
    _SENTINEL,
    BaseTimerContext,
    BasicAuth,
    HeadersMixin,
    TimerNoop,
    basicauth_from_netrc,
    frozen_dataclass_decorator,
    is_expected_content_type,
    netrc_from_env,
    parse_mimetype,
    reify,
    set_exception,
    set_result,
)
from .http import (
    SERVER_SOFTWARE,
    HttpVersion,
    HttpVersion10,
    HttpVersion11,
    StreamWriter,
)
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
    DEFAULT_JSON_DECODER,
    JSONDecoder,
    LooseCookies,
    LooseHeaders,
    Query,
    RawHeaders,
)

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
    except ImportError:  # pragma: no cover
        ssl = None  # type: ignore[assignment]
        SSLContext = object  # type: ignore[misc,assignment]


__all__ = ("ClientRequest", "ClientResponse", "RequestInfo", "Fingerprint")


if TYPE_CHECKING:
    from .client import ClientSession
    from .connector import Connection
    from .tracing import Trace


_CONTAINS_CONTROL_CHAR_RE = re.compile(r"[^-!#$%&'*+.^_`|~0-9a-zA-Z]")


def _gen_default_accept_encoding() -> str:
    return "gzip, deflate, br" if HAS_BROTLI else "gzip, deflate"


@frozen_dataclass_decorator
class ContentDisposition:
    type: Optional[str]
    parameters: MappingProxyType[str, str]
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
        real_url: URL = _SENTINEL,  # type: ignore[assignment]
    ) -> "RequestInfo":
        return tuple.__new__(
            cls, (url, method, headers, url if real_url is _SENTINEL else real_url)
        )


class Fingerprint:
    HASHFUNC_BY_DIGESTLEN: Dict[int, Callable[..., Any]] = {
        16: md5,
        20: sha1,
        32: sha256,
    }

    def __init__(self, fingerprint: bytes) -> None:
        digestlen = len(fingerprint)
        hashfunc = self.HASHFUNC_BY_DIGESTLEN.get(digestlen)
        if not hashfunc:
            raise ValueError("fingerprint has invalid length")
        elif hashfunc is md5 or hashfunc is sha1:
            raise ValueError("md5 and sha1 are insecure and not supported. Use sha256.")
        self._hashfunc = hashfunc
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> bytes:
        return self._fingerprint

    def check(self, transport: asyncio.Transport) -> None:
        if not transport.get_extra_info("sslcontext"):
            return
        sslobj = transport.get_extra_info("ssl_object")
        cert = sslobj.getpeercert(binary_form=True)  # type: ignore[union-attr]
        got = self._hashfunc(cert).digest()
        if got != self._fingerprint:
            host, port, *_ = transport.get_extra_info("peername")
            raise ServerFingerprintMismatch(self._fingerprint, got, host, port)


_SSL_SCHEMES = frozenset(("https", "wss"))


class ConnectionKey(NamedTuple):
    host: str
    port: Optional[int]
    is_ssl: bool
    ssl: Union["SSLContext", bool, Fingerprint]
    proxy: Optional[URL]
    proxy_auth: Optional[BasicAuth]
    proxy_headers_hash: Optional[int]


class ClientRequest:
    GET_METHODS = {
        hdrs.METH_GET,
        hdrs.METH_HEAD,
        hdrs.METH_OPTIONS,
        hdrs.METH_TRACE,
    }
    POST_METHODS = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT}
    ALL_METHODS = GET_METHODS.union(POST_METHODS).union({hdrs.METH_DELETE})

    DEFAULT_HEADERS = {
        hdrs.ACCEPT: "*/*",
        hdrs.ACCEPT_ENCODING: _gen_default_accept_encoding(),
    }

    body: Any = b""
    auth: Optional[BasicAuth] = None
    response: Optional["ClientResponse"] = None

    __writer: Optional[asyncio.Task[None]] = None
    _continue: Optional[asyncio.Future[bool]] = None

    _skip_auto_headers: Optional[CIMultiDict[Any]] = None

    def __init__(
        self,
        method: str,
        url: URL,
        *,
        params: Optional[Query] = None,
        headers: Optional[LooseHeaders] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        data: Any = None,
        cookies: Optional[LooseCookies] = None,
        auth: Optional[BasicAuth] = None,
        version: http.HttpVersion = http.HttpVersion11,
        compress: Union[str, bool] = False,
        chunked: Optional[bool] = None,
        expect100: bool = False,
        loop: asyncio.AbstractEventLoop,
        response_class: Optional[Type["ClientResponse"]] = None,
        proxy: Optional[URL] = None,
        proxy_auth: Optional[BasicAuth] = None,
        timer: Optional[BaseTimerContext] = None,
        session: Optional["ClientSession"] = None,
        ssl: Union["SSLContext", bool, Fingerprint] = True,
        proxy_headers: Optional[LooseHeaders] = None,
        traces: Optional[List["Trace"]] = None,
        trust_env: bool = False,
        server_hostname: Optional[str] = None,
    ) -> None:
        self._session = session
        self.original_url = url
        self.url = url
        self.method = method.upper()
        self.chunked = chunked
        self.loop = loop
        self.length = None
        self.response_class = response_class or ClientResponse
        self._timer = timer if timer is not None else TimerNoop()
        self._ssl = ssl
        self.server_hostname = server_hostname
        self.update_version(version)
        self.update_headers(headers)
        self.update_auto_headers(skip_auto_headers)
        self.update_cookies(cookies)
        self.update_content_encoding(data, compress)
        self.update_auth(auth, trust_env)
        self.update_proxy(proxy, proxy_auth, proxy_headers)
        self.update_body_from_data(data)
        self.update_transfer_encoding()
        self.update_expect_continue(expect100)
        self._traces = traces or []

    def __reset_writer(self, _: Any = None) -> None:
        self.__writer = None

    @property
    def skip_auto_headers(self) -> CIMultiDict[Any]:
        return self._skip_auto_headers or CIMultiDict()

    @property
    def _writer(self) -> Optional[asyncio.Task[None]]:
        return self.__writer

    @_writer.setter
    def _writer(self, writer: asyncio.Task[None]) -> None:
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        writer.add_done_callback(self.__reset_writer)

    def is_ssl(self) -> bool:
        return self.url.scheme in _SSL_SCHEMES

    @property
    def ssl(self) -> Union["SSLContext", bool, Fingerprint]:
        return self._ssl

    @property
    def connection_key(self) -> ConnectionKey:
        proxy_headers = self.proxy_headers
        h = hash(tuple(proxy_headers.items())) if proxy_headers else None
        return ConnectionKey(
            self.url.raw_host or "",
            self.url.port,
            self.is_ssl(),
            self._ssl,
            self.proxy,
            self.proxy_auth,
            h,
        )

    @property
    def host(self) -> str:
        return cast(str, self.url.raw_host)

    @property
    def port(self) -> Optional[int]:
        return self.url.port

    @property
    def request_info(self) -> RequestInfo:
        return RequestInfo(self.url, self.method, CIMultiDictProxy(self.headers), self.original_url)

    def update_host(self, url: URL) -> None:
        if not url.raw_host:
            raise InvalidURL(url)
        if url.raw_user or url.raw_password:
            self.auth = BasicAuth(url.user or "", url.password or "")

    def update_version(self, version: Union[http.HttpVersion, str]) -> None:
        if isinstance(version, str):
            v = version.split(".", 1)
            self.version = http.HttpVersion(int(v[0]), int(v[1]))
        else:
            self.version = version

    def update_headers(self, headers: Optional[LooseHeaders]) -> None:
        self.headers: CIMultiDict[str] = CIMultiDict()
        host = self.url.host_port_subcomponent
        self.headers[hdrs.HOST] = host
        if headers:
            for key, value in headers.items() if isinstance(headers, dict) else headers or []:
                self.headers.add(key, value)

    def update_auto_headers(self, skip_auto_headers: Optional[Iterable[str]]) -> None:
        self._skip_auto_headers = CIMultiDict((hdr, None) for hdr in skip_auto_headers) if skip_auto_headers else None
        used_headers = self.headers.copy()
        if self._skip_auto_headers:
            used_headers.extend(self._skip_auto_headers)
        for hdr, val in self.DEFAULT_HEADERS.items():
            if hdr not in used_headers:
                self.headers[hdr] = val
        if hdrs.USER_AGENT not in used_headers:
            self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE

    def update_cookies(self, cookies: Optional[LooseCookies]) -> None:
        c = SimpleCookie()
        if hdrs.COOKIE in self.headers:
            c.load(self.headers[hdrs.COOKIE])
            del self.headers[hdrs.COOKIE]
        if cookies:
            for name, value in cookies.items() if isinstance(cookies, dict) else cookies:
                c[name] = value
        self.headers[hdrs.COOKIE] = c.output(header="", sep=";").strip()

    def update_content_encoding(self, data: Any, compress: Union[bool, str]) -> None:
        self.compress = None
        if data:
            if self.headers.get(hdrs.CONTENT_ENCODING):
                if compress:
                    raise ValueError("Content-Encoding header is set")
            elif compress:
                self.compress = compress if isinstance(compress, str) else "deflate"
                self.headers[hdrs.CONTENT_ENCODING] = self.compress
                self.chunked = True

    def update_transfer_encoding(self) -> None:
        te = self.headers.get(hdrs.TRANSFER_ENCODING, "").lower()
        if "chunked" not in te and self.chunked:
            self.headers[hdrs.TRANSFER_ENCODING] = "chunked"
        elif not self.chunked and hdrs.CONTENT_LENGTH not in self.headers:
            self.headers[hdrs.CONTENT_LENGTH] = str(len(self.body))

    def update_auth(self, auth: Optional[BasicAuth], trust_env: bool) -> None:
        if auth is None and trust_env:
            netrc_obj = netrc_from_env()
            auth = basicauth_from_netrc(netrc_obj, self.url.host) if self.url.host else None
        if auth:
            self.headers[hdrs.AUTHORIZATION] = auth.encode()

    def update_body_from_data(self, body: Any) -> None:
        if body is None:
            return
        if isinstance(body, FormData):
            body = body()
        try:
            body = payload.PAYLOAD_REGISTRY.get(body)
        except payload.LookupError:
            boundary = parse_mimetype(self.headers.get(CONTENT_TYPE, "")).parameters.get("boundary")
            body = FormData(body, boundary=boundary)()
        self.body = body
        if self.chunked or (self.body.size is None and hdrs.CONTENT_LENGTH not in self.headers):
            self.chunked = True

    def update_expect_continue(self, expect: bool) -> None:
        if expect:
            self.headers[hdrs.EXPECT] = "100-continue"
            self._continue = self.loop.create_future()

    def update_proxy(
        self,
        proxy: Optional[URL],
        proxy_auth: Optional[BasicAuth],
        proxy_headers: Optional[LooseHeaders],
    ) -> None:
        self.proxy = proxy
        self.proxy_auth = proxy_auth
        self.proxy_headers = CIMultiDict(proxy_headers) if proxy_headers else None

    async def write_bytes(self, writer: AbstractStreamWriter, conn: "Connection") -> None:
        if self._continue:
            await writer.drain()
            await self._continue
        protocol = conn.protocol
        try:
            if isinstance(self.body, payload.Payload):
                await self.body.write(writer)
            else:
                for chunk in self.body:
                    await writer.write(chunk)
        except OSError as exc:
            set_exception(protocol, ClientOSError(exc.errno, f"Write error for {self.url}"), exc)
        except asyncio.CancelledError:
            conn.close()
            raise
        except Exception as exc:
            set_exception(protocol, ClientConnectionError(f"Connection error {conn}"), exc)
        else:
            await writer.write_eof()
            protocol.start_timeout()

    async def send(self, conn: "Connection") -> "ClientResponse":
        path = str(self.url) if self.proxy and not self.is_ssl() else self.url.raw_path_qs
        protocol = conn.protocol
        writer = StreamWriter(
            protocol,
            self.loop,
            on_chunk_sent=functools.partial(self._on_chunk_request_sent, self.method, self.url) if self._traces else None,
            on_headers_sent=functools.partial(self._on_headers_request_sent, self.method, self.url) if self._traces else None,
        )
        if self.compress:
            writer.enable_compression(self.compress)
        if self.chunked:
            writer.enable_chunking()
        if self.method in self.POST_METHODS and hdrs.CONTENT_TYPE not in self.headers:
            self.headers[hdrs.CONTENT_TYPE] = "application/octet-stream"
        if hdrs.CONNECTION not in self.headers:
            if conn._connector.force_close:
                self.headers[hdrs.CONNECTION] = "close"
        status_line = f"{self.method} {path} HTTP/{self.version.major}.{self.version.minor}"
        await writer.write_headers(status_line, self.headers)
        task = None
        if self.body or self._continue or (protocol and protocol.writing_paused):
            coro = self.write_bytes(writer, conn)
            task = asyncio.create_task(coro, eager_start=True) if sys.version_info >= (3, 12) else self.loop.create_task(coro)
            self._writer = task
        else:
            protocol.start_timeout()
            writer.set_eof()
        self.response = ClientResponse(
            self.method,
            self.original_url,
            writer=task,
            continue100=self._continue,
            timer=self._timer,
            request_info=self.request_info,
            traces=self._traces,
            loop=self.loop,
            session=self._session,
        )
        return self.response

    async def close(self) -> None:
        if self.__writer:
            await self.__writer

    def terminate(self) -> None:
        if self.__writer:
            self.__writer.cancel()
            self.__writer = None

    async def _on_chunk_request_sent(self, method: str, url: URL, chunk: bytes) -> None:
        for trace in self._traces:
            await trace.send_request_chunk_sent(method, url, chunk)

    async def _on_headers_request_sent(self, method: str, url: URL, headers: CIMultiDict