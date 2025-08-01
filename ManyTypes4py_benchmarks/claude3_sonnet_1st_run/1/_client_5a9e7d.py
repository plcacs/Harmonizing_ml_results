from __future__ import annotations
import datetime
import enum
import logging
import time
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, AsyncIterator, cast
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
__all__ = ['USE_CLIENT_DEFAULT', 'AsyncClient', 'Client']
T = TypeVar('T', bound='Client')
U = TypeVar('U', bound='AsyncClient')

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
    return {'http': 80, 'https': 443}.get(url.scheme, 0)

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
    """
    An async byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """

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
EventHook = Callable[..., Any]

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
        self._event_hooks: Dict[str, List[EventHook]] = {'request': list(event_hooks.get('request', [])), 'response': list(event_hooks.get('response', []))}
        self._trust_env = trust_env
        self._default_encoding = default_encoding
        self._state = ClientState.UNOPENED

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
        self._event_hooks = {'request': list(event_hooks.get('request', [])), 'response': list(event_hooks.get('response', []))}

    @property
    def auth(self) -> Optional[Auth]:
        """
        Authentication class used when none is passed at the request-level.

        See also [Authentication][0].

        [0]: /quickstart/#authentication
        """
        return self._auth

    @auth.setter
    def auth(self, auth: Optional[AuthTypes]) -> None:
        self._auth = self._build_auth(auth)

    @property
    def base_url(self) -> URL:
        """
        Base URL to use when sending requests with relative URLs.
        """
        return self._base_url

    @base_url.setter
    def base_url(self, url: Union[URL, str]) -> None:
        self._base_url = self._enforce_trailing_slash(URL(url))

    @property
    def headers(self) -> Headers:
        """
        HTTP headers to include when sending requests.
        """
        return self._headers

    @headers.setter
    def headers(self, headers: HeaderTypes) -> None:
        client_headers = Headers({b'Accept': b'*/*', b'Accept-Encoding': ACCEPT_ENCODING.encode('ascii'), b'Connection': b'keep-alive', b'User-Agent': USER_AGENT.encode('ascii')})
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
        url: Union[URL, str], 
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
        """
        Build and return a request instance.

        * The `params`, `headers` and `cookies` arguments
        are merged with any values set on the client.
        * The `url` argument is merged with any `base_url` set on the client.

        See also: [Request instances][0]

        [0]: /advanced/clients/#request-instances
        """
        url = self._merge_url(url)
        headers = self._merge_headers(headers)
        cookies = self._merge_cookies(cookies)
        params = self._merge_queryparams(params)
        extensions = {} if extensions is None else extensions
        if 'timeout' not in extensions:
            timeout = self.timeout if isinstance(timeout, UseClientDefault) else Timeout(timeout)
            extensions = dict(**extensions, timeout=timeout.as_dict())
        return Request(method, url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, extensions=extensions)

    def _merge_url(self, url: Union[URL, str]) -> URL:
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        merge_url = URL(url)
        if merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b'/')
            return self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url

    def _merge_cookies(self, cookies: Optional[CookieTypes] = None) -> Optional[Cookies]:
        """
        Merge a cookies argument together with any cookies on the client,
        to create the cookies used for the outgoing request.
        """
        if cookies or self.cookies:
            merged_cookies = Cookies(self.cookies)
            merged_cookies.update(cookies)
            return merged_cookies
        return cookies

    def _merge_headers(self, headers: Optional[HeaderTypes] = None) -> Headers:
        """
        Merge a headers argument together with any headers on the client,
        to create the headers used for the outgoing request.
        """
        merged_headers = Headers(self.headers)
        merged_headers.update(headers)
        return merged_headers

    def _merge_queryparams(self, params: Optional[QueryParamTypes] = None) -> Optional[QueryParams]:
        """
        Merge a queryparams argument together with any queryparams on the client,
        to create the queryparams used for the outgoing request.
        """
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
        """
        Given a request and a redirect response, return a new request that
        should be used to effect the redirect.
        """
        method = self._redirect_method(request, response)
        url = self._redirect_url(request, response)
        headers = self._redirect_headers(request, url, method)
        stream = self._redirect_stream(request, method)
        cookies = Cookies(self.cookies)
        return Request(method=method, url=url, headers=headers, cookies=cookies, stream=stream, extensions=request.extensions)

    def _redirect_method(self, request: Request, response: Response) -> str:
        """
        When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = request.method
        if response.status_code == codes.SEE_OTHER and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.FOUND and method != 'HEAD':
            method = 'GET'
        if response.status_code == codes.MOVED_PERMANENTLY and method == 'POST':
            method = 'GET'
        return method

    def _redirect_url(self, request: Request, response: Response) -> URL:
        """
        Return the URL for the redirect to follow.
        """
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
        """
        Return the headers that should be used for the redirect request.
        """
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

    def _redirect_stream(self, request: Request, method: str) -> Optional[Union[SyncByteStream, AsyncByteStream]]:
        """
        Return the body that should be used for the redirect request.
        """
        if method != request.method and method == 'GET':
            return None
        return request.stream

    def _set_timeout(self, request: Request) -> None:
        if 'timeout' not in request.extensions:
            timeout = self.timeout if isinstance(self.timeout, UseClientDefault) else Timeout(self.timeout)
            request.extensions = dict(**request.extensions, timeout=timeout.as_dict())

class Client(BaseClient):
    """
    An HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.

    It can be shared between threads.

    Usage:

    