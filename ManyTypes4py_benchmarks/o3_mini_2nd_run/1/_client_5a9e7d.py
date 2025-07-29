from __future__ import annotations
import datetime
import enum
import logging
import time
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG, Limits, Proxy, Timeout
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import InvalidURL, RemoteProtocolError, TooManyRedirects, request_context
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._types import AsyncByteStream, AuthTypes, CertTypes, CookieTypes, HeaderTypes, ProxyTypes, QueryParamTypes, RequestContent, RequestData, RequestExtensions, RequestFiles, SyncByteStream, TimeoutTypes
from ._urls import URL, QueryParams
from ._utils import URLPattern, get_environment_proxies
if typing.TYPE_CHECKING:
    import ssl
    from typing import Any, Callable, Iterator, AsyncIterator, Optional, List, Dict, Union, AsyncGenerator

__all__ = ['USE_CLIENT_DEFAULT', 'AsyncClient', 'Client']
T = typing.TypeVar('T', bound='Client')
U = typing.TypeVar('U', bound='AsyncClient')


def _is_https_redirect(url: URL, location: URL) -> bool:
    """
    Return 'True' if 'location' is a HTTPS upgrade of 'url'
    """
    if url.host != location.host:
        return False
    return url.scheme == 'http' and _port_or_default(url) == 80 and (location.scheme == 'https') and (_port_or_default(location) == 443)


def _port_or_default(url: URL) -> Optional[int]:
    if url.port is not None:
        return url.port
    return {'http': 80, 'https': 443}.get(url.scheme)


def _same_origin(url: URL, other: URL) -> bool:
    """
    Return 'True' if the given URLs share the same origin.
    """
    return url.scheme == other.scheme and url.host == other.host and (_port_or_default(url) == _port_or_default(other))


class UseClientDefault:
    """
    For some parameters such as `auth=...` and `timeout=...` we need to be able
    to indicate the default "unset" state, in a way that is distinctly different
    to using `None`.

    The default "unset" state indicates that whatever default is set on the
    client should be used. This is different to setting `None`, which
    explicitly disables the parameter, possibly overriding a client default.

    For example we use `timeout=USE_CLIENT_DEFAULT` in the `request()` signature.
    Omitting the `timeout` parameter will send a request using whatever default
    timeout has been configured on the client. Including `timeout=None` will
    ensure no timeout is used.

    Note that user code shouldn't need to use the `USE_CLIENT_DEFAULT` constant,
    but it is used internally when a parameter is not included.
    """
    pass


USE_CLIENT_DEFAULT = UseClientDefault()
logger = logging.getLogger('httpx')
USER_AGENT = f'python-httpx/{__version__}'
ACCEPT_ENCODING = ', '.join([key for key in SUPPORTED_DECODERS.keys() if key != 'identity'])


class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3


class BoundSyncStream(SyncByteStream):
    """
    A byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None:
        self._stream: SyncByteStream = stream
        self._response: Response = response
        self._start: float = start

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            yield chunk

    def close(self) -> None:
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        self._stream.close()


class BoundAsyncStream(AsyncByteStream):
    """
    An async byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None:
        self._stream: AsyncByteStream = stream
        self._response: Response = response
        self._start: float = start

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        await self._stream.aclose()


EventHook = typing.Callable[..., typing.Any]


class BaseClient:
    def __init__(
        self,
        *,
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing.List[EventHook]]] = None,
        base_url: str = '',
        trust_env: bool = True,
        default_encoding: str = 'utf-8'
    ) -> None:
        event_hooks = {} if event_hooks is None else event_hooks
        self._base_url: URL = self._enforce_trailing_slash(URL(base_url))
        self._auth: typing.Optional[Auth] = self._build_auth(auth)
        self._params: QueryParams = QueryParams(params)
        self.headers = headers  # This uses the setter.
        self._cookies: Cookies = Cookies(cookies)
        self._timeout: Timeout = Timeout(timeout)
        self.follow_redirects: bool = follow_redirects
        self.max_redirects: int = max_redirects
        self._event_hooks: typing.Dict[str, typing.List[EventHook]] = {
            'request': list(event_hooks.get('request', [])),
            'response': list(event_hooks.get('response', []))
        }
        self._trust_env: bool = trust_env
        self._default_encoding: str = default_encoding
        self._state: ClientState = ClientState.UNOPENED

    @property
    def is_closed(self) -> bool:
        """
        Check if the client being closed
        """
        return self._state == ClientState.CLOSED

    @property
    def trust_env(self) -> bool:
        return self._trust_env

    def _enforce_trailing_slash(self, url: URL) -> URL:
        if url.raw_path.endswith(b'/'):
            return url
        return url.copy_with(raw_path=url.raw_path + b'/')

    def _get_proxy_map(self, proxy: typing.Optional[typing.Union[str, URL, Proxy]], allow_env_proxies: bool) -> typing.Dict[str, typing.Optional[Proxy]]:
        if proxy is None:
            if allow_env_proxies:
                return {key: None if url is None else Proxy(url=url) for key, url in get_environment_proxies().items()}
            return {}
        else:
            proxy_inst: Proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy  # type: ignore
            return {'all://': proxy_inst}

    @property
    def timeout(self) -> Timeout:
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None:
        self._timeout = Timeout(timeout)

    @property
    def event_hooks(self) -> typing.Dict[str, typing.List[EventHook]]:
        return self._event_hooks

    @event_hooks.setter
    def event_hooks(self, event_hooks: typing.Dict[str, typing.List[EventHook]]) -> None:
        self._event_hooks = {'request': list(event_hooks.get('request', [])), 'response': list(event_hooks.get('response', []))}

    @property
    def auth(self) -> typing.Optional[Auth]:
        """
        Authentication class used when none is passed at the request-level.
        """
        return self._auth

    @auth.setter
    def auth(self, auth: typing.Optional[AuthTypes]) -> None:
        self._auth = self._build_auth(auth)

    @property
    def base_url(self) -> URL:
        """
        Base URL to use when sending requests with relative URLs.
        """
        return self._base_url

    @base_url.setter
    def base_url(self, url: str) -> None:
        self._base_url = self._enforce_trailing_slash(URL(url))

    @property
    def headers(self) -> Headers:
        """
        HTTP headers to include when sending requests.
        """
        return self._headers

    @headers.setter
    def headers(self, headers: typing.Optional[HeaderTypes]) -> None:
        client_headers: Headers = Headers({b'Accept': b'*/*',
                                           b'Accept-Encoding': ACCEPT_ENCODING.encode('ascii'),
                                           b'Connection': b'keep-alive',
                                           b'User-Agent': USER_AGENT.encode('ascii')})
        client_headers.update(headers)
        self._headers = client_headers

    @property
    def cookies(self) -> Cookies:
        """
        Cookie values to include when sending requests.
        """
        return self._cookies

    @cookies.setter
    def cookies(self, cookies: CookieTypes) -> None:
        self._cookies = Cookies(cookies)

    @property
    def params(self) -> QueryParams:
        """
        Query parameters to include in the URL when sending requests.
        """
        return self._params

    @params.setter
    def params(self, params: QueryParamTypes) -> None:
        self._params = QueryParams(params)

    def build_request(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Request:
        """
        Build and return a request instance.
        """
        url = self._merge_url(url)
        headers = self._merge_headers(headers)
        cookies = self._merge_cookies(cookies)
        params = self._merge_queryparams(params)
        extensions = {} if extensions is None else extensions
        if 'timeout' not in extensions:
            timeout_inst: Timeout = self.timeout if isinstance(timeout, UseClientDefault) else Timeout(timeout)  # type: ignore
            extensions = dict(**extensions, timeout=timeout_inst.as_dict())
        return Request(method, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, extensions=extensions)

    def _merge_url(self, url: typing.Union[str, URL]) -> URL:
        merge_url = URL(url)
        if merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b'/')
            return self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url

    def _merge_cookies(self, cookies: typing.Optional[CookieTypes] = None) -> typing.Optional[Cookies]:
        if cookies or self.cookies:
            merged_cookies: Cookies = Cookies(self.cookies)
            merged_cookies.update(cookies)
            return merged_cookies
        return cookies

    def _merge_headers(self, headers: typing.Optional[HeaderTypes] = None) -> Headers:
        merged_headers: Headers = Headers(self.headers)
        merged_headers.update(headers)
        return merged_headers

    def _merge_queryparams(self, params: typing.Optional[QueryParamTypes] = None) -> typing.Optional[QueryParamTypes]:
        if params or self.params:
            merged_queryparams: QueryParams = QueryParams(self.params)
            return merged_queryparams.merge(params)
        return params

    def _build_auth(self, auth: typing.Optional[AuthTypes]) -> typing.Optional[Auth]:
        if auth is None:
            return None
        elif isinstance(auth, tuple):
            return BasicAuth(username=auth[0], password=auth[1])
        elif isinstance(auth, Auth):
            return auth
        elif callable(auth):
            return FunctionAuth(func=auth)
        else:
            raise TypeError(f'Invalid "auth" argument: {auth!r}')

    def _build_request_auth(self, request: Request, auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT) -> Auth:
        auth_inst: typing.Optional[Auth] = self._auth if isinstance(auth, UseClientDefault) else self._build_auth(auth)
        if auth_inst is not None:
            return auth_inst
        username, password = (request.url.username, request.url.password)
        if username or password:
            return BasicAuth(username=username, password=password)
        return Auth()

    def _build_redirect_request(self, request: Request, response: Response) -> Request:
        method: str = self._redirect_method(request, response)
        url: URL = self._redirect_url(request, response)
        headers: Headers = self._redirect_headers(request, url, method)
        stream: typing.Optional[SyncByteStream] = self._redirect_stream(request, method)
        cookies: Cookies = Cookies(self.cookies)
        return Request(method=method, url=url, headers=headers, cookies=cookies, stream=stream, extensions=request.extensions)

    def _redirect_method(self, request: Request, response: Response) -> str:
        method: str = request.method
        if response.status_code == codes.SEE_OTHER and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.FOUND and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.MOVED_PERMANENTLY and method == 'POST':
            method = 'GET'
        return method

    def _redirect_url(self, request: Request, response: Response) -> URL:
        location: str = response.headers['Location']
        try:
            url = URL(location)
        except InvalidURL as exc:
            raise RemoteProtocolError(f'Invalid URL in location header: {exc}.', request=request) from None
        if url.scheme and (not url.host):
            url = url.copy_with(host=request.url.host)
        if url.is_relative_url:
            url = request.url.join(url)
        if request.url.fragment and (not url.fragment):
            url = url.copy_with(fragment=request.url.fragment)
        return url

    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers:
        headers: Headers = Headers(request.headers)
        if not _same_origin(url, request.url):
            if not _is_https_redirect(request.url, url):
                headers.pop('Authorization', None)
            headers['Host'] = url.netloc.decode('ascii')
        if method != request.method and method == 'GET':
            headers.pop('Content-Length', None)
            headers.pop('Transfer-Encoding', None)
        headers.pop('Cookie', None)
        return headers

    def _redirect_stream(self, request: Request, method: str) -> typing.Optional[SyncByteStream]:
        if method != request.method and method == 'GET':
            return None
        return request.stream

    def _set_timeout(self, request: Request) -> None:
        if 'timeout' not in request.extensions:
            timeout_inst: Timeout = self.timeout if isinstance(self.timeout, UseClientDefault) else Timeout(self.timeout)  # type: ignore
            request.extensions = dict(**request.extensions, timeout=timeout_inst.as_dict())


class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: typing.Optional[ProxyTypes] = None,
        mounts: typing.Optional[typing.Dict[str, BaseTransport]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing.List[EventHook]]] = None,
        base_url: str = '',
        transport: typing.Optional[BaseTransport] = None,
        default_encoding: str = 'utf-8'
    ) -> None:
        super().__init__(auth=auth, params=params, headers=headers, cookies=cookies, timeout=timeout, follow_redirects=follow_redirects, max_redirects=max_redirects, event_hooks=event_hooks, base_url=base_url, trust_env=trust_env, default_encoding=default_encoding)
        if http2:
            try:
                import h2  # type: ignore
            except ImportError:
                raise ImportError("Using http2=True, but the 'h2' package is not installed. Make sure to install httpx using `pip install httpx[http2]`.") from None
        allow_env_proxies: bool = trust_env and transport is None
        proxy_map: typing.Dict[str, typing.Optional[Proxy]] = self._get_proxy_map(proxy, allow_env_proxies)
        self._transport: BaseTransport = self._init_transport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, transport=transport)
        self._mounts: typing.Dict[URLPattern, typing.Optional[BaseTransport]] = {
            URLPattern(key): None if proxy_inst is None else self._init_proxy_transport(proxy_inst, verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)
            for key, proxy_inst in proxy_map.items()
        }
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(
        self,
        *,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: typing.Optional[BaseTransport] = None
    ) -> BaseTransport:
        if transport is not None:
            return transport
        return HTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)

    def _init_proxy_transport(
        self,
        proxy: Proxy,
        *,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS
    ) -> BaseTransport:
        return HTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, proxy=proxy)

    def _transport_for_url(self, url: URL) -> BaseTransport:
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport
        return self._transport

    def request(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        if cookies is not None:
            message: str = 'Setting per-request cookies=<...> is being deprecated, because the expected behaviour on cookie persistence is ambiguous. Set cookies directly on the client instance instead.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        request_inst: Request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        return self.send(request_inst, auth=auth, follow_redirects=follow_redirects)

    @contextmanager
    def stream(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Iterator[Response]:
        request_inst: Request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        response: Response = self.send(request=request_inst, auth=auth, follow_redirects=follow_redirects, stream=True)
        try:
            yield response
        finally:
            response.close()

    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT
    ) -> Response:
        if self._state == ClientState.CLOSED:
            raise RuntimeError('Cannot send a request, as the client has been closed.')
        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects  # type: ignore
        self._set_timeout(request)
        auth_inst: Auth = self._build_request_auth(request, auth)
        response: Response = self._send_handling_auth(request, auth=auth_inst, follow_redirects=follow_redirects, history=[])
        try:
            if not stream:
                response.read()
            return response
        except BaseException as exc:
            response.close()
            raise exc

    def _send_handling_auth(
        self,
        request: Request,
        *,
        auth: Auth,
        follow_redirects: bool,
        history: typing.List[Response]
    ) -> Response:
        auth_flow = auth.sync_auth_flow(request)
        try:
            request = next(auth_flow)
            while True:
                response: Response = self._send_handling_redirects(request, follow_redirects=follow_redirects, history=history)
                try:
                    try:
                        next_request: Request = auth_flow.send(response)
                    except StopIteration:
                        return response
                    response.history = list(history)
                    response.read()
                    request = next_request
                    history.append(response)
                except BaseException as exc:
                    response.close()
                    raise exc
        finally:
            auth_flow.close()

    def _send_handling_redirects(
        self,
        request: Request,
        *,
        follow_redirects: bool,
        history: typing.List[Response]
    ) -> Response:
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects('Exceeded maximum allowed redirects.', request=request)
            for hook in self._event_hooks['request']:
                hook(request)
            response: Response = self._send_single_request(request)
            try:
                for hook in self._event_hooks['response']:
                    hook(response)
                response.history = list(history)
                if not response.has_redirect_location:
                    return response
                request = self._build_redirect_request(request, response)
                history = history + [response]
                if follow_redirects:
                    response.read()
                else:
                    response.next_request = request
                    return response
            except BaseException as exc:
                response.close()
                raise exc

    def _send_single_request(self, request: Request) -> Response:
        transport: BaseTransport = self._transport_for_url(request.url)
        start: float = time.perf_counter()
        if not isinstance(request.stream, SyncByteStream):
            raise RuntimeError('Attempted to send an async request with a sync Client instance.')
        with request_context(request=request):
            response: Response = transport.handle_request(request)
        assert isinstance(response.stream, SyncByteStream)
        response.request = request
        response.stream = BoundSyncStream(response.stream, response=response, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding
        logger.info('HTTP Request: %s %s "%s %d %s"', request.method, request.url, response.http_version, response.status_code, response.reason_phrase)
        return response

    def get(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def options(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def head(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def post(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def put(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def patch(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def delete(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return self.request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def close(self) -> None:
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED
            self._transport.close()
            for transport in self._mounts.values():
                if transport is not None:
                    transport.close()

    def __enter__(self) -> Client:
        if self._state != ClientState.UNOPENED:
            msg: str = {ClientState.OPENED: 'Cannot open a client instance more than once.', ClientState.CLOSED: 'Cannot reopen a client instance, once it has been closed.'}[self._state]
            raise RuntimeError(msg)
        self._state = ClientState.OPENED
        self._transport.__enter__()
        for transport in self._mounts.values():
            if transport is not None:
                transport.__enter__()
        return self

    def __exit__(self, exc_type: typing.Optional[type], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None:
        self._state = ClientState.CLOSED
        self._transport.__exit__(exc_type, exc_value, traceback)
        for transport in self._mounts.values():
            if transport is not None:
                transport.__exit__(exc_type, exc_value, traceback)


class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: typing.Optional[AuthTypes] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        http1: bool = True,
        http2: bool = False,
        proxy: typing.Optional[ProxyTypes] = None,
        mounts: typing.Optional[typing.Dict[str, AsyncBaseTransport]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: typing.Optional[typing.Dict[str, typing.List[EventHook]]] = None,
        base_url: str = '',
        transport: typing.Optional[AsyncBaseTransport] = None,
        trust_env: bool = True,
        default_encoding: str = 'utf-8'
    ) -> None:
        super().__init__(auth=auth, params=params, headers=headers, cookies=cookies, timeout=timeout, follow_redirects=follow_redirects, max_redirects=max_redirects, event_hooks=event_hooks, base_url=base_url, trust_env=trust_env, default_encoding=default_encoding)
        if http2:
            try:
                import h2  # type: ignore
            except ImportError:
                raise ImportError("Using http2=True, but the 'h2' package is not installed. Make sure to install httpx using `pip install httpx[http2]`.") from None
        allow_env_proxies: bool = trust_env and transport is None
        proxy_map: typing.Dict[str, typing.Optional[Proxy]] = self._get_proxy_map(proxy, allow_env_proxies)
        self._transport: AsyncBaseTransport = self._init_transport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, transport=transport)
        self._mounts: typing.Dict[URLPattern, typing.Optional[AsyncBaseTransport]] = {
            URLPattern(key): None if proxy_inst is None else self._init_proxy_transport(proxy_inst, verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)
            for key, proxy_inst in proxy_map.items()
        }
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(
        self,
        *,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: typing.Optional[AsyncBaseTransport] = None
    ) -> AsyncBaseTransport:
        if transport is not None:
            return transport
        return AsyncHTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)

    def _init_proxy_transport(
        self,
        proxy: Proxy,
        *,
        verify: bool = True,
        cert: typing.Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS
    ) -> AsyncBaseTransport:
        return AsyncHTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, proxy=proxy)

    def _transport_for_url(self, url: URL) -> AsyncBaseTransport:
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport
        return self._transport

    async def request(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        if cookies is not None:
            message: str = 'Setting per-request cookies=<...> is being deprecated, because the expected behaviour on cookie persistence is ambiguous. Set cookies directly on the client instance instead.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        request_inst: Request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        return await self.send(request_inst, auth=auth, follow_redirects=follow_redirects)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> typing.AsyncGenerator[Response, None]:
        request_inst: Request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        response: Response = await self.send(request=request_inst, auth=auth, follow_redirects=follow_redirects, stream=True)
        try:
            yield response
        finally:
            await response.aclose()

    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT
    ) -> Response:
        if self._state == ClientState.CLOSED:
            raise RuntimeError('Cannot send a request, as the client has been closed.')
        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects  # type: ignore
        self._set_timeout(request)
        auth_inst: Auth = self._build_request_auth(request, auth)
        response: Response = await self._send_handling_auth(request, auth=auth_inst, follow_redirects=follow_redirects, history=[])
        try:
            if not stream:
                await response.aread()
            return response
        except BaseException as exc:
            await response.aclose()
            raise exc

    async def _send_handling_auth(
        self,
        request: Request,
        *,
        auth: Auth,
        follow_redirects: bool,
        history: typing.List[Response]
    ) -> Response:
        auth_flow = auth.async_auth_flow(request)
        try:
            request = await auth_flow.__anext__()
            while True:
                response: Response = await self._send_handling_redirects(request, follow_redirects=follow_redirects, history=history)
                try:
                    try:
                        next_request: Request = await auth_flow.asend(response)
                    except StopAsyncIteration:
                        return response
                    response.history = list(history)
                    await response.aread()
                    request = next_request
                    history.append(response)
                except BaseException as exc:
                    await response.aclose()
                    raise exc
        finally:
            await auth_flow.aclose()

    async def _send_handling_redirects(
        self,
        request: Request,
        *,
        follow_redirects: bool,
        history: typing.List[Response]
    ) -> Response:
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects('Exceeded maximum allowed redirects.', request=request)
            for hook in self._event_hooks['request']:
                await hook(request)
            response: Response = await self._send_single_request(request)
            try:
                for hook in self._event_hooks['response']:
                    await hook(response)
                response.history = list(history)
                if not response.has_redirect_location:
                    return response
                request = self._build_redirect_request(request, response)
                history = history + [response]
                if follow_redirects:
                    await response.aread()
                else:
                    response.next_request = request
                    return response
            except BaseException as exc:
                await response.aclose()
                raise exc

    async def _send_single_request(self, request: Request) -> Response:
        transport: AsyncBaseTransport = self._transport_for_url(request.url)
        start: float = time.perf_counter()
        if not isinstance(request.stream, AsyncByteStream):
            raise RuntimeError('Attempted to send an sync request with an AsyncClient instance.')
        with request_context(request=request):
            response: Response = await transport.handle_async_request(request)
        assert isinstance(response.stream, AsyncByteStream)
        response.request = request
        response.stream = BoundAsyncStream(response.stream, response=response, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding
        logger.info('HTTP Request: %s %s "%s %d %s"', request.method, request.url, response.http_version, response.status_code, response.reason_phrase)
        return response

    async def get(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def options(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def head(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def post(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def put(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def patch(
        self,
        url: typing.Union[str, URL],
        *,
        content: typing.Optional[RequestContent] = None,
        data: typing.Optional[RequestData] = None,
        files: typing.Optional[RequestFiles] = None,
        json: typing.Optional[typing.Any] = None,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def delete(
        self,
        url: typing.Union[str, URL],
        *,
        params: typing.Optional[QueryParamTypes] = None,
        headers: typing.Optional[HeaderTypes] = None,
        cookies: typing.Optional[CookieTypes] = None,
        auth: typing.Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: typing.Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: typing.Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: typing.Optional[RequestExtensions] = None
    ) -> Response:
        return await self.request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def aclose(self) -> None:
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED
            await self._transport.aclose()
            for proxy in self._mounts.values():
                if proxy is not None:
                    await proxy.aclose()

    async def __aenter__(self) -> AsyncClient:
        if self._state != ClientState.UNOPENED:
            msg: str = {ClientState.OPENED: 'Cannot open a client instance more than once.', ClientState.CLOSED: 'Cannot reopen a client instance, once it has been closed.'}[self._state]
            raise RuntimeError(msg)
        self._state = ClientState.OPENED
        await self._transport.__aenter__()
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aenter__()
        return self

    async def __aexit__(self, exc_type: typing.Optional[type], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None:
        self._state = ClientState.CLOSED
        await self._transport.__aexit__(exc_type, exc_value, traceback)
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aexit__(exc_type, exc_value, traceback)