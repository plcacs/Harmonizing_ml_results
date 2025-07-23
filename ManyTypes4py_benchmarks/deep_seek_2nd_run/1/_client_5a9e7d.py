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
    from typing import Any, AsyncGenerator, AsyncIterator, Dict, Generator, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union

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

def _port_or_default(url: URL) -> int:
    if url.port is not None:
        return url.port
    return {'http': 80, 'https': 443}.get(url.scheme, 443)

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
    """
USE_CLIENT_DEFAULT = UseClientDefault()
logger: logging.Logger = logging.getLogger('httpx')
USER_AGENT: str = f'python-httpx/{__version__}'
ACCEPT_ENCODING: str = ', '.join([key for key in SUPPORTED_DECODERS.keys() if key != 'identity'])

class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3

class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None:
        self._stream = stream
        self._response = response
        self._start = start

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            yield chunk

    def close(self) -> None:
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        self._stream.close()

class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None:
        self._stream = stream
        self._response = response
        self._start = start

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
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: str = '',
        trust_env: bool = True,
        default_encoding: str = 'utf-8'
    ) -> None:
        event_hooks = {} if event_hooks is None else event_hooks
        self._base_url = self._enforce_trailing_slash(URL(base_url))
        self._auth = self._build_auth(auth)
        self._params = QueryParams(params)
        self.headers = Headers(headers)
        self._cookies = Cookies(cookies)
        self._timeout = Timeout(timeout)
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self._event_hooks = {
            'request': list(event_hooks.get('request', [])),
            'response': list(event_hooks.get('response', []))
        }
        self._trust_env = trust_env
        self._default_encoding = default_encoding
        self._state = ClientState.UNOPENED

    @property
    def is_closed(self) -> bool:
        return self._state == ClientState.CLOSED

    @property
    def trust_env(self) -> bool:
        return self._trust_env

    def _enforce_trailing_slash(self, url: URL) -> URL:
        if url.raw_path.endswith(b'/'):
            return url
        return url.copy_with(raw_path=url.raw_path + b'/')

    def _get_proxy_map(self, proxy: Optional[ProxyTypes], allow_env_proxies: bool) -> Dict[str, Optional[Proxy]]:
        if proxy is None:
            if allow_env_proxies:
                return {key: None if url is None else Proxy(url=url) for key, url in get_environment_proxies().items()}
            return {}
        else:
            proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
            return {'all://': proxy}

    @property
    def timeout(self) -> Timeout:
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None:
        self._timeout = Timeout(timeout)

    @property
    def event_hooks(self) -> Dict[str, List[EventHook]]:
        return self._event_hooks

    @event_hooks.setter
    def event_hooks(self, event_hooks: Dict[str, List[EventHook]]) -> None:
        self._event_hooks = {
            'request': list(event_hooks.get('request', [])),
            'response': list(event_hooks.get('response', []))
        }

    @property
    def auth(self) -> Optional[Auth]:
        return self._auth

    @auth.setter
    def auth(self, auth: Optional[AuthTypes]) -> None:
        self._auth = self._build_auth(auth)

    @property
    def base_url(self) -> URL:
        return self._base_url

    @base_url.setter
    def base_url(self, url: str) -> None:
        self._base_url = self._enforce_trailing_slash(URL(url))

    @property
    def headers(self) -> Headers:
        return self._headers

    @headers.setter
    def headers(self, headers: Optional[HeaderTypes]) -> None:
        client_headers = Headers({
            b'Accept': b'*/*',
            b'Accept-Encoding': ACCEPT_ENCODING.encode('ascii'),
            b'Connection': b'keep-alive',
            b'User-Agent': USER_AGENT.encode('ascii')
        })
        client_headers.update(headers)
        self._headers = client_headers

    @property
    def cookies(self) -> Cookies:
        return self._cookies

    @cookies.setter
    def cookies(self, cookies: Optional[CookieTypes]) -> None:
        self._cookies = Cookies(cookies)

    @property
    def params(self) -> QueryParams:
        return self._params

    @params.setter
    def params(self, params: Optional[QueryParamTypes]) -> None:
        self._params = QueryParams(params)

    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None
    ) -> Request:
        url = self._merge_url(url)
        headers = self._merge_headers(headers)
        cookies = self._merge_cookies(cookies)
        params = self._merge_queryparams(params)
        extensions = {} if extensions is None else extensions
        if 'timeout' not in extensions:
            timeout = self.timeout if isinstance(timeout, UseClientDefault) else Timeout(timeout)
            extensions = dict(**extensions, timeout=timeout.as_dict())
        return Request(
            method, url,
            content=content, data=data, files=files, json=json,
            params=params, headers=headers, cookies=cookies,
            extensions=extensions
        )

    def _merge_url(self, url: str) -> URL:
        merge_url = URL(url)
        if merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b'/')
            return self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url

    def _merge_cookies(self, cookies: Optional[CookieTypes] = None) -> Optional[Cookies]:
        if cookies or self.cookies:
            merged_cookies = Cookies(self.cookies)
            merged_cookies.update(cookies)
            return merged_cookies
        return cookies

    def _merge_headers(self, headers: Optional[HeaderTypes] = None) -> Headers:
        merged_headers = Headers(self.headers)
        merged_headers.update(headers)
        return merged_headers

    def _merge_queryparams(self, params: Optional[QueryParamTypes] = None) -> Optional[QueryParams]:
        if params or self.params:
            merged_queryparams = QueryParams(self.params)
            return merged_queryparams.merge(params)
        return params

    def _build_auth(self, auth: Optional[AuthTypes]) -> Optional[Auth]:
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

    def _build_request_auth(self, request: Request, auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT) -> Auth:
        auth = self._auth if isinstance(auth, UseClientDefault) else self._build_auth(auth)
        if auth is not None:
            return auth
        username, password = (request.url.username, request.url.password)
        if username or password:
            return BasicAuth(username=username, password=password)
        return Auth()

    def _build_redirect_request(self, request: Request, response: Response) -> Request:
        method = self._redirect_method(request, response)
        url = self._redirect_url(request, response)
        headers = self._redirect_headers(request, url, method)
        stream = self._redirect_stream(request, method)
        cookies = Cookies(self.cookies)
        return Request(
            method=method,
            url=url,
            headers=headers,
            cookies=cookies,
            stream=stream,
            extensions=request.extensions
        )

    def _redirect_method(self, request: Request, response: Response) -> str:
        method = request.method
        if response.status_code == codes.SEE_OTHER and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.FOUND and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.MOVED_PERMANENTLY and method == 'POST':
            method = 'GET'
        return method

    def _redirect_url(self, request: Request, response: Response) -> URL:
        location = response.headers['Location']
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
        headers = Headers(request.headers)
        if not _same_origin(url, request.url):
            if not _is_https_redirect(request.url, url):
                headers.pop('Authorization', None)
            headers['Host'] = url.netloc.decode('ascii')
        if method != request.method and method == 'GET':
            headers.pop('Content-Length', None)
            headers.pop('Transfer-Encoding', None)
        headers.pop('Cookie', None)
        return headers

    def _redirect_stream(self, request: Request, method: str) -> Optional[SyncByteStream]:
        if method != request.method and method == 'GET':
            return None
        return request.stream

    def _set_timeout(self, request: Request) -> None:
        if 'timeout' not in request.extensions:
            timeout = self.timeout if isinstance(self.timeout, UseClientDefault) else Timeout(self.timeout)
            request.extensions = dict(**request.extensions, timeout=timeout.as_dict())

class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: Union[bool, str, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: Optional[ProxyTypes] = None,
        mounts: Optional[Dict[str, BaseTransport]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: str = '',
        transport: Optional[BaseTransport] = None,
        default_encoding: str = 'utf-8'
    ) -> None:
        super().__init__(
            auth=auth, params=params, headers=headers, cookies=cookies,
            timeout=timeout, follow_redirects=follow_redirects,
            max_redirects=max_redirects, event_hooks=event_hooks,
            base_url=base_url, trust_env=trust_env,
            default_encoding=default_encoding
        )
        if http2:
            try:
                import h2
            except ImportError:
                raise ImportError("Using http2=True, but the 'h2' package is not installed. Make sure to install httpx using `pip install httpx[http2]`.") from None
        allow_env_proxies = trust_env and transport is None
        proxy_map = self._get_proxy_map(proxy, allow_env_proxies)
        self._transport = self._init_transport(
            verify=verify, cert=cert, trust_env=trust_env,
            http1=http1, http2=http2, limits=limits,
            transport=transport
        )
        self._mounts = {
            URLPattern(key): None if proxy is None else self._init_proxy_transport(
                proxy, verify=verify, cert=cert,
                trust_env=trust_env, http1=http1,
                http2=http2, limits=limits
            )
            for key, proxy in proxy_map.items()
        }
        if mounts is not None:
            self._mounts.update({
                URLPattern(key): transport
                for key, transport in mounts.items()
            })
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(
        self,
        verify: Union[bool, str, ssl.SSLContext] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        transport: Optional[BaseTransport] = None
    ) -> BaseTransport:
        if transport is not None:
            return transport
        return HTTPTransport(
            verify=verify, cert=cert, trust_env=trust_env,
            http1=http1, http2=http2, limits=limits
        )

    def _init_proxy_transport(
        self,
        proxy: Proxy,
        verify: Union[bool, str, ssl.SSLContext] = True,
        cert: Optional[CertTypes