#!/usr/bin/env python3
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
    Any,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
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
from .client_reqrep import SSL_ALLOWED_TYPES, ClientRequest, ClientResponse, Fingerprint, RequestInfo
from .client_ws import DEFAULT_WS_CLIENT_TIMEOUT, ClientWebSocketResponse, ClientWSTimeout
from .connector import HTTP_AND_EMPTY_SCHEMA_SET, BaseConnector, NamedPipeConnector, TCPConnector, UnixConnector
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
    'ClientConnectionError',
    'ClientConnectionResetError',
    'ClientConnectorCertificateError',
    'ClientConnectorDNSError',
    'ClientConnectorError',
    'ClientConnectorSSLError',
    'ClientError',
    'ClientHttpProxyError',
    'ClientOSError',
    'ClientPayloadError',
    'ClientProxyConnectionError',
    'ClientResponseError',
    'ClientSSLError',
    'ConnectionTimeoutError',
    'ContentTypeError',
    'InvalidURL',
    'InvalidUrlClientError',
    'RedirectClientError',
    'NonHttpUrlClientError',
    'InvalidUrlRedirectClientError',
    'NonHttpUrlRedirectClientError',
    'ServerConnectionError',
    'ServerDisconnectedError',
    'ServerFingerprintMismatch',
    'ServerTimeoutError',
    'SocketTimeoutError',
    'TooManyRedirects',
    'WSServerHandshakeError',
    'ClientRequest',
    'ClientResponse',
    'Fingerprint',
    'RequestInfo',
    'BaseConnector',
    'TCPConnector',
    'UnixConnector',
    'NamedPipeConnector',
    'ClientWebSocketResponse',
    'ClientSession',
    'ClientTimeout',
    'ClientWSTimeout',
    'request',
    'WSMessageTypeError',
)

if sys.version_info >= (3, 11):
    from typing import Unpack  # type: ignore

_T_RetType = TypeVar("_T_RetType", bound=Union["ClientResponse", "ClientWebSocketResponse"])
_CharsetResolver = Callable[["ClientResponse", bytes], str]

class _RequestOptions(Dict[str, Any]):
    pass

@frozen_dataclass_decorator
class ClientTimeout:
    total: Optional[float] = None
    connect: Optional[float] = None
    sock_read: Optional[float] = None
    sock_connect: Optional[float] = None
    ceil_threshold: int = 5

DEFAULT_TIMEOUT: Final[ClientTimeout] = ClientTimeout(total=5 * 60, sock_connect=30)
IDEMPOTENT_METHODS: Final[FrozenSet[str]] = frozenset({'GET', 'HEAD', 'OPTIONS', 'TRACE', 'PUT', 'DELETE'})
_RetType = TypeVar('_RetType', bound=Union[ClientResponse, ClientWebSocketResponse])

@final
class ClientSession:
    """First-class interface for making HTTP requests."""
    __slots__ = (
        '_base_url',
        '_base_url_origin',
        '_source_traceback',
        '_connector',
        '_loop',
        '_cookie_jar',
        '_connector_owner',
        '_default_auth',
        '_version',
        '_json_serialize',
        '_requote_redirect_url',
        '_timeout',
        '_raise_for_status',
        '_auto_decompress',
        '_trust_env',
        '_default_headers',
        '_skip_auto_headers',
        '_request_class',
        '_response_class',
        '_ws_response_class',
        '_trace_configs',
        '_read_bufsize',
        '_max_line_size',
        '_max_field_size',
        '_resolve_charset',
        '_default_proxy',
        '_default_proxy_auth',
        '_retry_connection',
    )

    def __init__(
        self,
        base_url: Optional[Union[str, URL]] = None,
        *,
        connector: Optional[BaseConnector] = None,
        cookies: Optional[LooseCookies] = None,
        headers: Optional[LooseHeaders] = None,
        proxy: Optional[Union[str, URL]] = None,
        proxy_auth: Optional[Any] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[Any] = None,
        json_serialize: JSONEncoder = json.dumps,
        request_class: Type[ClientRequest] = ClientRequest,
        response_class: Type[ClientResponse] = ClientResponse,
        ws_response_class: Type[ClientWebSocketResponse] = ClientWebSocketResponse,
        version: HttpVersion = http.HttpVersion11,
        cookie_jar: Optional[AbstractCookieJar] = None,
        connector_owner: bool = True,
        raise_for_status: Union[bool, None] = False,
        timeout: Union[ClientTimeout, Any] = sentinel,
        auto_decompress: bool = True,
        trust_env: bool = False,
        requote_redirect_url: bool = True,
        trace_configs: Optional[List[TraceConfig]] = None,
        read_bufsize: int = 2 ** 16,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        fallback_charset_resolver: _CharsetResolver = lambda r, b: 'utf-8'
    ) -> None:
        self._connector: Optional[BaseConnector] = None
        if base_url is None or isinstance(base_url, URL):
            self._base_url = base_url
            self._base_url_origin = None if base_url is None else base_url.origin()
        else:
            self._base_url = URL(base_url)
            self._base_url_origin = self._base_url.origin()
            assert self._base_url.absolute, 'Only absolute URLs are supported'
        if self._base_url is not None and (not self._base_url.path.endswith('/')):
            raise ValueError("base_url must have a trailing '/'")
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        if timeout is sentinel or timeout is None:
            timeout = DEFAULT_TIMEOUT
        if not isinstance(timeout, ClientTimeout):
            raise ValueError(f"timeout parameter cannot be of {type(timeout)} type, please use 'timeout=ClientTimeout(...)'")
        self._timeout: ClientTimeout = timeout
        if connector is None:
            connector = TCPConnector()  # type: ignore
        self._connector = connector
        self._loop = loop
        if loop.get_debug():
            self._source_traceback: Optional[List[traceback.FrameSummary]] = traceback.extract_stack(sys._getframe(1))
        else:
            self._source_traceback = None
        if connector._loop is not loop:
            raise RuntimeError('Session and connector have to use same event loop')
        if cookie_jar is None:
            cookie_jar = CookieJar()
        self._cookie_jar = cookie_jar
        if cookies:
            self._cookie_jar.update_cookies(cookies)
        self._connector_owner: bool = connector_owner
        self._default_auth = auth
        self._version: HttpVersion = version
        self._json_serialize: JSONEncoder = json_serialize
        self._raise_for_status: Union[bool, None, Callable[[ClientResponse], Awaitable[None]]] = raise_for_status
        self._auto_decompress: bool = auto_decompress
        self._trust_env: bool = trust_env
        self._requote_redirect_url: bool = requote_redirect_url
        self._read_bufsize: int = read_bufsize
        self._max_line_size: int = max_line_size
        self._max_field_size: int = max_field_size
        if headers:
            real_headers: CIMultiDict[str] = CIMultiDict(headers)
        else:
            real_headers = CIMultiDict()
        self._default_headers: CIMultiDict[str] = real_headers
        if skip_auto_headers is not None:
            self._skip_auto_headers: FrozenSet[istr] = frozenset((istr(i) for i in skip_auto_headers))
        else:
            self._skip_auto_headers = frozenset()
        self._request_class: Type[ClientRequest] = request_class
        self._response_class: Type[ClientResponse] = response_class
        self._ws_response_class: Type[ClientWebSocketResponse] = ws_response_class
        self._trace_configs: List[TraceConfig] = trace_configs or []
        for trace_config in self._trace_configs:
            trace_config.freeze()
        self._resolve_charset: _CharsetResolver = fallback_charset_resolver
        self._default_proxy = proxy
        self._default_proxy_auth = proxy_auth
        self._retry_connection: bool = True

    def __init_subclass__(cls: Type[Any]) -> None:
        raise TypeError('Inheritance class {} from ClientSession is forbidden'.format(cls.__name__))

    def __del__(self, _warnings=warnings) -> None:
        if not self.closed:
            _warnings.warn(f'Unclosed client session {self!r}', ResourceWarning, source=self)
            context: Dict[str, Any] = {'client_session': self, 'message': 'Unclosed client session'}
            if self._source_traceback is not None:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)

    if sys.version_info >= (3, 11):
        from typing import override  # type: ignore

        def request(self, method: str, url: Union[str, URL], **kwargs: Any) -> Any:
            ...
    else:
        def request(self, method: str, url: Union[str, URL], **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP request."""
            return _RequestContextManager(self._request(method, url, **kwargs))

    def _build_url(self, str_or_url: Union[str, URL]) -> URL:
        url: URL = URL(str_or_url)
        if self._base_url and (not url.absolute):
            return self._base_url.join(url)
        return url

    async def _request(
        self,
        method: str,
        str_or_url: Union[str, URL],
        *,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        cookies: Optional[Mapping[str, str]] = None,
        headers: Optional[LooseHeaders] = None,
        skip_auto_headers: Optional[Iterable[str]] = None,
        auth: Optional[Any] = None,
        allow_redirects: bool = True,
        max_redirects: int = 10,
        compress: Union[bool, int] = False,
        chunked: Optional[int] = None,
        expect100: bool = False,
        raise_for_status: Optional[Union[bool, Callable[[ClientResponse], Awaitable[None]]]] = None,
        read_until_eof: bool = True,
        proxy: Optional[Union[str, URL]] = None,
        proxy_auth: Optional[Any] = None,
        timeout: Union[ClientTimeout, Any] = sentinel,
        ssl: Union[SSL_ALLOWED_TYPES, bool] = True,
        server_hostname: Optional[str] = None,
        proxy_headers: Optional[LooseHeaders] = None,
        trace_request_ctx: Optional[Any] = None,
        read_bufsize: Optional[int] = None,
        auto_decompress: Optional[bool] = None,
        max_line_size: Optional[int] = None,
        max_field_size: Optional[int] = None,
    ) -> ClientResponse:
        if self.closed:
            raise RuntimeError('Session is closed')
        if not isinstance(ssl, SSL_ALLOWED_TYPES):
            raise TypeError('ssl should be SSLContext, Fingerprint, or bool, got {!r} instead.'.format(ssl))
        if data is not None and json is not None:
            raise ValueError('data and json parameters can not be used at the same time')
        elif json is not None:
            data = payload.JsonPayload(json, dumps=self._json_serialize)
        redirects: int = 0
        history: List[ClientResponse] = []
        version: HttpVersion = self._version
        params = params or {}
        headers = self._prepare_headers(headers)
        try:
            url: URL = self._build_url(str_or_url)
        except ValueError as e:
            raise InvalidUrlClientError(str_or_url) from e
        assert self._connector is not None
        if url.scheme not in self._connector.allowed_protocol_schema_set:
            raise NonHttpUrlClientError(url)
        if skip_auto_headers is not None:
            skip_headers: Optional[Set[istr]] = {istr(i) for i in skip_auto_headers} | self._skip_auto_headers
        elif self._skip_auto_headers:
            skip_headers = self._skip_auto_headers  # type: ignore
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
        tm: TimeoutHandle = TimeoutHandle(self._loop, real_timeout.total, ceil_threshold=real_timeout.ceil_threshold)
        handle = tm.start()
        if read_bufsize is None:
            read_bufsize = self._read_bufsize
        if auto_decompress is None:
            auto_decompress = self._auto_decompress
        if max_line_size is None:
            max_line_size = self._max_line_size
        if max_field_size is None:
            max_field_size = self._max_field_size
        traces: List[Trace] = [Trace(self, trace_config, trace_config.trace_config_ctx(trace_request_ctx=trace_request_ctx)) for trace_config in self._trace_configs]
        for trace in traces:
            await trace.send_request_start(method, url.update_query(params), headers)
        timer = tm.timer()
        try:
            with timer:
                retry_persistent_connection: bool = self._retry_connection and method in IDEMPOTENT_METHODS
                while True:
                    url, auth_from_url = strip_auth_from_url(url)
                    if not url.raw_host:
                        err_exc_cls: Type[Exception] = InvalidUrlRedirectClientError if redirects else InvalidUrlClientError
                        raise err_exc_cls(url)
                    if not history and (auth and auth_from_url):
                        raise ValueError('Cannot combine AUTH argument with credentials encoded in URL')
                    if auth is None or (history and auth_from_url is not None):
                        auth = auth_from_url
                    if auth is None and self._default_auth and (not self._base_url or self._base_url_origin == url.origin()):
                        auth = self._default_auth
                    if auth is not None and hdrs.AUTHORIZATION in headers:
                        raise ValueError('Cannot combine AUTHORIZATION header with AUTH argument or credentials encoded in URL')
                    all_cookies = self._cookie_jar.filter_cookies(url)
                    if cookies is not None:
                        tmp_cookie_jar = CookieJar(quote_cookie=self._cookie_jar.quote_cookie)
                        tmp_cookie_jar.update_cookies(cookies)
                        req_cookies = tmp_cookie_jar.filter_cookies(url)
                        if req_cookies:
                            all_cookies.load(req_cookies)
                    if proxy is not None:
                        proxy = URL(proxy)
                    elif self._trust_env:
                        with suppress(LookupError):
                            proxy, proxy_auth = get_env_proxy_for_url(url)
                    req: ClientRequest = self._request_class(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        skip_auto_headers=skip_headers,
                        data=data,
                        cookies=all_cookies,
                        auth=auth,
                        version=version,
                        compress=compress,
                        chunked=chunked,
                        expect100=expect100,
                        loop=self._loop,
                        response_class=self._response_class,
                        proxy=proxy,
                        proxy_auth=proxy_auth,
                        timer=timer,
                        session=self,
                        ssl=ssl,
                        server_hostname=server_hostname,
                        proxy_headers=proxy_headers,
                        traces=traces,
                        trust_env=self.trust_env,
                    )
                    try:
                        conn = await self._connector.connect(req, traces=traces, timeout=real_timeout)
                    except asyncio.TimeoutError as exc:
                        raise ConnectionTimeoutError(f'Connection timeout to host {url}') from exc
                    assert conn.transport is not None
                    assert conn.protocol is not None
                    conn.protocol.set_response_params(
                        timer=timer,
                        skip_payload=method in EMPTY_BODY_METHODS,
                        read_until_eof=read_until_eof,
                        auto_decompress=auto_decompress,
                        read_timeout=real_timeout.sock_read,
                        read_bufsize=read_bufsize,
                        timeout_ceil_threshold=self._connector._timeout_ceil_threshold,  # type: ignore
                        max_line_size=max_line_size,
                        max_field_size=max_field_size,
                    )
                    try:
                        try:
                            resp: ClientResponse = await req.send(conn)
                            try:
                                await resp.start(conn)
                            except BaseException:
                                resp.close()
                                raise
                        except BaseException:
                            conn.close()
                            raise
                    except (ClientOSError, ServerDisconnectedError):
                        if retry_persistent_connection:
                            retry_persistent_connection = False
                            continue
                        raise
                    except ClientError:
                        raise
                    except OSError as exc:
                        if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                            raise
                        raise ClientOSError(*exc.args) from exc
                    if (cookies := resp._cookies):
                        self._cookie_jar.update_cookies(cookies, resp.url)
                    if resp.status in (301, 302, 303, 307, 308) and allow_redirects:
                        for trace in traces:
                            await trace.send_request_redirect(method, url.update_query(params), headers, resp)
                        redirects += 1
                        history.append(resp)
                        if max_redirects and redirects >= max_redirects:
                            resp.close()
                            raise TooManyRedirects(history[0].request_info, tuple(history))
                        if resp.status == 303 and resp.method != hdrs.METH_HEAD or (resp.status in (301, 302) and resp.method == hdrs.METH_POST):
                            method = hdrs.METH_GET
                            data = None
                            if headers.get(hdrs.CONTENT_LENGTH):
                                headers.pop(hdrs.CONTENT_LENGTH)
                        r_url: Optional[str] = resp.headers.get(hdrs.LOCATION) or resp.headers.get(hdrs.URI)
                        if r_url is None:
                            break
                        else:
                            resp.release()
                        try:
                            parsed_redirect_url: URL = URL(r_url, encoded=not self._requote_redirect_url)
                        except ValueError as e:
                            raise InvalidUrlRedirectClientError(r_url, 'Server attempted redirecting to a location that does not look like a URL') from e
                        scheme: str = parsed_redirect_url.scheme
                        if scheme not in HTTP_AND_EMPTY_SCHEMA_SET:
                            resp.close()
                            raise NonHttpUrlRedirectClientError(r_url)
                        elif not scheme:
                            parsed_redirect_url = url.join(parsed_redirect_url)
                        is_same_host_https_redirect: bool = url.host == parsed_redirect_url.host and parsed_redirect_url.scheme == 'https' and (url.scheme == 'http')
                        try:
                            redirect_origin: URL = parsed_redirect_url.origin()
                        except ValueError as origin_val_err:
                            raise InvalidUrlRedirectClientError(parsed_redirect_url, 'Invalid redirect URL origin') from origin_val_err
                        if not is_same_host_https_redirect and url.origin() != redirect_origin:
                            auth = None
                            headers.pop(hdrs.AUTHORIZATION, None)
                        url = parsed_redirect_url
                        params = {}
                        resp.release()
                        continue
                    break
            if raise_for_status is None:
                raise_for_status = self._raise_for_status
            if raise_for_status is None:
                pass
            elif callable(raise_for_status):
                await raise_for_status(resp)
            elif raise_for_status:
                resp.raise_for_status()
            if handle is not None:
                if resp.connection is not None:
                    resp.connection.add_callback(handle.cancel)
                else:
                    handle.cancel()
            resp._history = tuple(history)
            for trace in traces:
                await trace.send_request_end(method, url.update_query(params), headers, resp)
            return resp
        except BaseException as e:
            tm.close()
            if handle:
                handle.cancel()
                handle = None
            for trace in traces:
                await trace.send_request_exception(method, url.update_query(params), headers, e)
            raise

    def ws_connect(
        self,
        url: Union[str, URL],
        *,
        method: str = hdrs.METH_GET,
        protocols: Iterable[str] = (),
        timeout: Union[ClientWSTimeout, float, Any] = sentinel,
        receive_timeout: Optional[float] = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: Optional[float] = None,
        auth: Optional[Any] = None,
        origin: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[LooseHeaders] = None,
        proxy: Optional[Union[str, URL]] = None,
        proxy_auth: Optional[Any] = None,
        ssl: Union[SSL_ALLOWED_TYPES, bool] = True,
        server_hostname: Optional[str] = None,
        proxy_headers: Optional[LooseHeaders] = None,
        compress: int = 0,
        max_msg_size: int = 4 * 1024 * 1024,
    ) -> "_WSRequestContextManager":
        """Initiate websocket connection."""
        return _WSRequestContextManager(
            self._ws_connect(
                url,
                method=method,
                protocols=protocols,
                timeout=timeout,
                receive_timeout=receive_timeout,
                autoclose=autoclose,
                autoping=autoping,
                heartbeat=heartbeat,
                auth=auth,
                origin=origin,
                params=params,
                headers=headers,
                proxy=proxy,
                proxy_auth=proxy_auth,
                ssl=ssl,
                server_hostname=server_hostname,
                proxy_headers=proxy_headers,
                compress=compress,
                max_msg_size=max_msg_size,
            )
        )

    async def _ws_connect(
        self,
        url: Union[str, URL],
        *,
        method: str = hdrs.METH_GET,
        protocols: Iterable[str] = (),
        timeout: Union[ClientWSTimeout, float, Any] = sentinel,
        receive_timeout: Optional[float] = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: Optional[float] = None,
        auth: Optional[Any] = None,
        origin: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[LooseHeaders] = None,
        proxy: Optional[Union[str, URL]] = None,
        proxy_auth: Optional[Any] = None,
        ssl: Union[SSL_ALLOWED_TYPES, bool] = True,
        server_hostname: Optional[str] = None,
        proxy_headers: Optional[LooseHeaders] = None,
        compress: int = 0,
        max_msg_size: int = 4 * 1024 * 1024,
    ) -> ClientWebSocketResponse:
        if timeout is not sentinel:
            if isinstance(timeout, ClientWSTimeout):
                ws_timeout: ClientWSTimeout = timeout
            else:
                warnings.warn("parameter 'timeout' of type 'float' is deprecated, please use 'timeout=ClientWSTimeout(ws_close=...)'", DeprecationWarning, stacklevel=2)
                ws_timeout = ClientWSTimeout(ws_close=timeout)
        else:
            ws_timeout = DEFAULT_WS_CLIENT_TIMEOUT
        if receive_timeout is not None:
            warnings.warn("float parameter 'receive_timeout' is deprecated, please use parameter 'timeout=ClientWSTimeout(ws_receive=...)'", DeprecationWarning, stacklevel=2)
            ws_timeout = dataclasses.replace(ws_timeout, ws_receive=receive_timeout)
        if headers is None:
            real_headers: CIMultiDict[str] = CIMultiDict()
        else:
            real_headers = CIMultiDict(headers)
        default_headers: Dict[str, str] = {hdrs.UPGRADE: 'websocket', hdrs.CONNECTION: 'Upgrade', hdrs.SEC_WEBSOCKET_VERSION: '13'}
        for key, value in default_headers.items():
            real_headers.setdefault(key, value)
        sec_key: bytes = base64.b64encode(os.urandom(16))
        real_headers[hdrs.SEC_WEBSOCKET_KEY] = sec_key.decode()
        if protocols:
            real_headers[hdrs.SEC_WEBSOCKET_PROTOCOL] = ','.join(protocols)
        if origin is not None:
            real_headers[hdrs.ORIGIN] = origin
        if compress:
            extstr: str = ws_ext_gen(compress=compress)
            real_headers[hdrs.SEC_WEBSOCKET_EXTENSIONS] = extstr
        if not isinstance(ssl, SSL_ALLOWED_TYPES):
            raise TypeError('ssl should be SSLContext, Fingerprint, or bool, got {!r} instead.'.format(ssl))
        resp: ClientResponse = await self.request(method, url, params=params, headers=real_headers, read_until_eof=False, auth=auth, proxy=proxy, proxy_auth=proxy_auth, ssl=ssl, server_hostname=server_hostname, proxy_headers=proxy_headers)
        try:
            if resp.status != 101:
                raise WSServerHandshakeError(resp.request_info, resp.history, message='Invalid response status', status=resp.status, headers=resp.headers)
            if resp.headers.get(hdrs.UPGRADE, '').lower() != 'websocket':
                raise WSServerHandshakeError(resp.request_info, resp.history, message='Invalid upgrade header', status=resp.status, headers=resp.headers)
            if resp.headers.get(hdrs.CONNECTION, '').lower() != 'upgrade':
                raise WSServerHandshakeError(resp.request_info, resp.history, message='Invalid connection header', status=resp.status, headers=resp.headers)
            r_key: str = resp.headers.get(hdrs.SEC_WEBSOCKET_ACCEPT, '')
            match: str = base64.b64encode(hashlib.sha1(sec_key + WS_KEY).digest()).decode()
            if r_key != match:
                raise WSServerHandshakeError(resp.request_info, resp.history, message='Invalid challenge response', status=resp.status, headers=resp.headers)
            protocol: Optional[str] = None
            if protocols and hdrs.SEC_WEBSOCKET_PROTOCOL in resp.headers:
                resp_protocols = [proto.strip() for proto in resp.headers[hdrs.SEC_WEBSOCKET_PROTOCOL].split(',')]
                for proto in resp_protocols:
                    if proto in protocols:
                        protocol = proto
                        break
            notakeover: bool = False
            if compress:
                compress_hdrs: Optional[str] = resp.headers.get(hdrs.SEC_WEBSOCKET_EXTENSIONS)
                if compress_hdrs:
                    try:
                        compress, notakeover = ws_ext_parse(compress_hdrs)
                    except WSHandshakeError as exc:
                        raise WSServerHandshakeError(resp.request_info, resp.history, message=exc.args[0], status=resp.status, headers=resp.headers) from exc
                else:
                    compress = 0
                    notakeover = False
            conn = resp.connection
            assert conn is not None
            conn_proto = conn.protocol
            assert conn_proto is not None
            if ws_timeout.ws_receive is None:
                conn_proto.read_timeout = None
            elif conn_proto.read_timeout is not None:
                conn_proto.read_timeout = max(ws_timeout.ws_receive, conn_proto.read_timeout)
            transport = conn.transport
            assert transport is not None
            reader = WebSocketDataQueue(conn_proto, 2 ** 16, loop=self._loop)
            conn_proto.set_parser(WebSocketReader(reader, max_msg_size), reader)
            writer = WebSocketWriter(conn_proto, transport, use_mask=True, compress=compress, notakeover=notakeover)
        except BaseException:
            resp.close()
            raise
        else:
            return self._ws_response_class(reader, writer, protocol, resp, ws_timeout, autoclose, autoping, self._loop, heartbeat=heartbeat, compress=compress, client_notakeover=notakeover)

    def _prepare_headers(self, headers: Optional[LooseHeaders]) -> CIMultiDict[str]:
        """Add default headers and transform it to CIMultiDict"""
        result: CIMultiDict[str] = CIMultiDict(self._default_headers)
        if headers:
            if not isinstance(headers, (MultiDictProxy, MultiDict)):
                headers = CIMultiDict(headers)
            added_names: Set[str] = set()
            for key, value in headers.items():
                if key in added_names:
                    result.add(key, value)
                else:
                    result[key] = value
                    added_names.add(key)
        return result

    if sys.version_info >= (3, 11):

        def get(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def options(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def head(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def post(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def put(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def patch(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...

        def delete(self, url: Union[str, URL], **kwargs: Any) -> Any:
            ...
    else:
        def get(self, url: Union[str, URL], *, allow_redirects: bool = True, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP GET request."""
            return _RequestContextManager(self._request(hdrs.METH_GET, url, allow_redirects=allow_redirects, **kwargs))

        def options(self, url: Union[str, URL], *, allow_redirects: bool = True, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP OPTIONS request."""
            return _RequestContextManager(self._request(hdrs.METH_OPTIONS, url, allow_redirects=allow_redirects, **kwargs))

        def head(self, url: Union[str, URL], *, allow_redirects: bool = False, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP HEAD request."""
            return _RequestContextManager(self._request(hdrs.METH_HEAD, url, allow_redirects=allow_redirects, **kwargs))

        def post(self, url: Union[str, URL], *, data: Optional[Any] = None, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP POST request."""
            return _RequestContextManager(self._request(hdrs.METH_POST, url, data=data, **kwargs))

        def put(self, url: Union[str, URL], *, data: Optional[Any] = None, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP PUT request."""
            return _RequestContextManager(self._request(hdrs.METH_PUT, url, data=data, **kwargs))

        def patch(self, url: Union[str, URL], *, data: Optional[Any] = None, **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP PATCH request."""
            return _RequestContextManager(self._request(hdrs.METH_PATCH, url, data=data, **kwargs))

        def delete(self, url: Union[str, URL], **kwargs: Any) -> "_RequestContextManager":
            """Perform HTTP DELETE request."""
            return _RequestContextManager(self._request(hdrs.METH_DELETE, url, **kwargs))

    async def close(self) -> None:
        """Close underlying connector.

        Release all acquired resources.
        """
        if not self.closed:
            if self._connector is not None and self._connector_owner:
                await self._connector.close()
            self._connector = None

    @property
    def closed(self) -> bool:
        """Is client session closed.

        A readonly property.
        """
        return self._connector is None or self._connector.closed

    @property
    def connector(self) -> Optional[BaseConnector]:
        """Connector instance used for the session."""
        return self._connector

    @property
    def cookie_jar(self) -> AbstractCookieJar:
        """The session cookies."""
        return self._cookie_jar

    @property
    def version(self) -> HttpVersion:
        """The session HTTP protocol version."""
        return self._version

    @property
    def requote_redirect_url(self) -> bool:
        """Do URL requoting on redirection handling."""
        return self._requote_redirect_url

    @property
    def timeout(self) -> ClientTimeout:
        """Timeout for the session."""
        return self._timeout

    @property
    def headers(self) -> CIMultiDict[str]:
        """The default headers of the client session."""
        return self._default_headers

    @property
    def skip_auto_headers(self) -> FrozenSet[istr]:
        """Headers for which autogeneration should be skipped"""
        return self._skip_auto_headers

    @property
    def auth(self) -> Any:
        """An object that represents HTTP Basic Authorization"""
        return self._default_auth

    @property
    def json_serialize(self) -> JSONEncoder:
        """Json serializer callable"""
        return self._json_serialize

    @property
    def connector_owner(self) -> bool:
        """Should connector be closed on session closing"""
        return self._connector_owner

    @property
    def raise_for_status(self) -> Union[bool, None, Callable[[ClientResponse], Awaitable[None]]]:
        """Should `ClientResponse.raise_for_status()` be called for each response."""
        return self._raise_for_status

    @property
    def auto_decompress(self) -> bool:
        """Should the body response be automatically decompressed."""
        return self._auto_decompress

    @property
    def trust_env(self) -> bool:
        """
        Should proxies information from environment or netrc be trusted.

        Information is from HTTP_PROXY / HTTPS_PROXY environment variables
        or ~/.netrc file if present.
        """
        return self._trust_env

    @property
    def trace_configs(self) -> List[TraceConfig]:
        """A list of TraceConfig instances used for client tracing"""
        return self._trace_configs

    def detach(self) -> None:
        """Detach connector from session without closing the former.

        Session is switched to closed state anyway.
        """
        self._connector = None

    async def __aenter__(self) -> "ClientSession":
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        await self.close()

class _BaseRequestContextManager(Coroutine[Any, Any, _RetType], Generic[_RetType]):
    __slots__ = ('_coro', '_resp')

    def __init__(self, coro: Coroutine[Any, Any, _RetType]) -> None:
        self._coro: Coroutine[Any, Any, _RetType] = coro
        self._resp: Optional[_RetType] = None

    def send(self, arg: Any) -> Any:
        return self._coro.send(arg)

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        return self._coro.throw(*args, **kwargs)

    def close(self) -> Any:
        return self._coro.close()

    def __await__(self) -> Iterator[Any]:
        return self._coro.__await__()

    def __iter__(self) -> Iterator[Any]:
        return self.__await__()

    async def __aenter__(self) -> _RetType:
        self._resp = await self._coro
        return await self._resp.__aenter__()  # type: ignore

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> Any:
        await self._resp.__aexit__(exc_type, exc, tb)  # type: ignore

_RequestContextManager = _BaseRequestContextManager[ClientResponse]
_WSRequestContextManager = _BaseRequestContextManager[ClientWebSocketResponse]

class _SessionRequestContextManager:
    __slots__ = ('_coro', '_resp', '_session')

    def __init__(self, coro: Coroutine[Any, Any, ClientResponse], session: ClientSession) -> None:
        self._coro: Coroutine[Any, Any, ClientResponse] = coro
        self._resp: Optional[ClientResponse] = None
        self._session: ClientSession = session

    async def __aenter__(self) -> ClientResponse:
        try:
            self._resp = await self._coro
        except BaseException:
            await self._session.close()
            raise
        else:
            return self._resp

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        assert self._resp is not None
        self._resp.close()
        await self._session.close()

if sys.version_info >= (3, 11):
    from typing import override  # type: ignore

    def request(method: str, url: Union[str, URL], *, version: HttpVersion = http.HttpVersion11, connector: Optional[BaseConnector] = None, **kwargs: Any) -> Any:
        ...
else:
    def request(method: str, url: Union[str, URL], *, version: HttpVersion = http.HttpVersion11, connector: Optional[BaseConnector] = None, **kwargs: Any) -> _SessionRequestContextManager:
        """Constructs and sends a request.

        Returns response object.
        method - HTTP method
        url - request url
        params - (optional) Dictionary or bytes to be sent in the query
        string of the new request
        data - (optional) Dictionary, bytes, or file-like object to
        send in the body of the request
        json - (optional) Any json compatible python object
        headers - (optional) Dictionary of HTTP Headers to send with
        the request
        cookies - (optional) Dict object to send with the request
        auth - (optional) BasicAuth named tuple represent HTTP Basic Auth
        auth - aiohttp.helpers.BasicAuth
        allow_redirects - (optional) If set to False, do not follow
        redirects
        version - Request HTTP version.
        compress - Set to True if request has to be compressed
        with deflate encoding.
        chunked - Set to chunk size for chunked transfer encoding.
        expect100 - Expect 100-continue response from server.
        connector - BaseConnector sub-class instance to support
        connection pooling.
        read_until_eof - Read response until eof if response
        does not have Content-Length header.
        loop - Optional event loop.
        timeout - Optional ClientTimeout settings structure, 5min
        total timeout by default.
        Usage::
        >>> import aiohttp
        >>> async with aiohttp.request('GET', 'http://python.org/') as resp:
        ...    print(resp)
        ...    data = await resp.read()
        <ClientResponse(https://www.python.org/) [200 OK]>
        """
        connector_owner: bool = False
        if connector is None:
            connector_owner = True
            connector = TCPConnector(force_close=True)  # type: ignore
        session: ClientSession = ClientSession(
            cookies=kwargs.pop('cookies', None),
            version=version,
            timeout=kwargs.pop('timeout', sentinel),
            connector=connector,
            connector_owner=connector_owner
        )
        return _SessionRequestContextManager(session._request(method, url, **kwargs), session)
