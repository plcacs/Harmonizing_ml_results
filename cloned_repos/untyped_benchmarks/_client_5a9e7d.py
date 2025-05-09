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
__all__ = ['USE_CLIENT_DEFAULT', 'AsyncClient', 'Client']
T = typing.TypeVar('T', bound='Client')
U = typing.TypeVar('U', bound='AsyncClient')

def _is_https_redirect(url, location):
    """
    Return 'True' if 'location' is a HTTPS upgrade of 'url'
    """
    if url.host != location.host:
        return False
    return url.scheme == 'http' and _port_or_default(url) == 80 and (location.scheme == 'https') and (_port_or_default(location) == 443)

def _port_or_default(url):
    if url.port is not None:
        return url.port
    return {'http': 80, 'https': 443}.get(url.scheme)

def _same_origin(url, other):
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

    def __init__(self, stream, response, start):
        self._stream = stream
        self._response = response
        self._start = start

    def __iter__(self):
        for chunk in self._stream:
            yield chunk

    def close(self):
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        self._stream.close()

class BoundAsyncStream(AsyncByteStream):
    """
    An async byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """

    def __init__(self, stream, response, start):
        self._stream = stream
        self._response = response
        self._start = start

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self):
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        await self._stream.aclose()
EventHook = typing.Callable[..., typing.Any]

class BaseClient:

    def __init__(self, *, auth=None, params=None, headers=None, cookies=None, timeout=DEFAULT_TIMEOUT_CONFIG, follow_redirects=False, max_redirects=DEFAULT_MAX_REDIRECTS, event_hooks=None, base_url='', trust_env=True, default_encoding='utf-8'):
        event_hooks = {} if event_hooks is None else event_hooks
        self._base_url = self._enforce_trailing_slash(URL(base_url))
        self._auth = self._build_auth(auth)
        self._params = QueryParams(params)
        self.headers = Headers(headers)
        self._cookies = Cookies(cookies)
        self._timeout = Timeout(timeout)
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self._event_hooks = {'request': list(event_hooks.get('request', [])), 'response': list(event_hooks.get('response', []))}
        self._trust_env = trust_env
        self._default_encoding = default_encoding
        self._state = ClientState.UNOPENED

    @property
    def is_closed(self):
        """
        Check if the client being closed
        """
        return self._state == ClientState.CLOSED

    @property
    def trust_env(self):
        return self._trust_env

    def _enforce_trailing_slash(self, url):
        if url.raw_path.endswith(b'/'):
            return url
        return url.copy_with(raw_path=url.raw_path + b'/')

    def _get_proxy_map(self, proxy, allow_env_proxies):
        if proxy is None:
            if allow_env_proxies:
                return {key: None if url is None else Proxy(url=url) for key, url in get_environment_proxies().items()}
            return {}
        else:
            proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
            return {'all://': proxy}

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        self._timeout = Timeout(timeout)

    @property
    def event_hooks(self):
        return self._event_hooks

    @event_hooks.setter
    def event_hooks(self, event_hooks):
        self._event_hooks = {'request': list(event_hooks.get('request', [])), 'response': list(event_hooks.get('response', []))}

    @property
    def auth(self):
        """
        Authentication class used when none is passed at the request-level.

        See also [Authentication][0].

        [0]: /quickstart/#authentication
        """
        return self._auth

    @auth.setter
    def auth(self, auth):
        self._auth = self._build_auth(auth)

    @property
    def base_url(self):
        """
        Base URL to use when sending requests with relative URLs.
        """
        return self._base_url

    @base_url.setter
    def base_url(self, url):
        self._base_url = self._enforce_trailing_slash(URL(url))

    @property
    def headers(self):
        """
        HTTP headers to include when sending requests.
        """
        return self._headers

    @headers.setter
    def headers(self, headers):
        client_headers = Headers({b'Accept': b'*/*', b'Accept-Encoding': ACCEPT_ENCODING.encode('ascii'), b'Connection': b'keep-alive', b'User-Agent': USER_AGENT.encode('ascii')})
        client_headers.update(headers)
        self._headers = client_headers

    @property
    def cookies(self):
        """
        Cookie values to include when sending requests.
        """
        return self._cookies

    @cookies.setter
    def cookies(self, cookies):
        self._cookies = Cookies(cookies)

    @property
    def params(self):
        """
        Query parameters to include in the URL when sending requests.
        """
        return self._params

    @params.setter
    def params(self, params):
        self._params = QueryParams(params)

    def build_request(self, method, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, timeout=USE_CLIENT_DEFAULT, extensions=None):
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

    def _merge_url(self, url):
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        merge_url = URL(url)
        if merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b'/')
            return self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url

    def _merge_cookies(self, cookies=None):
        """
        Merge a cookies argument together with any cookies on the client,
        to create the cookies used for the outgoing request.
        """
        if cookies or self.cookies:
            merged_cookies = Cookies(self.cookies)
            merged_cookies.update(cookies)
            return merged_cookies
        return cookies

    def _merge_headers(self, headers=None):
        """
        Merge a headers argument together with any headers on the client,
        to create the headers used for the outgoing request.
        """
        merged_headers = Headers(self.headers)
        merged_headers.update(headers)
        return merged_headers

    def _merge_queryparams(self, params=None):
        """
        Merge a queryparams argument together with any queryparams on the client,
        to create the queryparams used for the outgoing request.
        """
        if params or self.params:
            merged_queryparams = QueryParams(self.params)
            return merged_queryparams.merge(params)
        return params

    def _build_auth(self, auth):
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

    def _build_request_auth(self, request, auth=USE_CLIENT_DEFAULT):
        auth = self._auth if isinstance(auth, UseClientDefault) else self._build_auth(auth)
        if auth is not None:
            return auth
        username, password = (request.url.username, request.url.password)
        if username or password:
            return BasicAuth(username=username, password=password)
        return Auth()

    def _build_redirect_request(self, request, response):
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

    def _redirect_method(self, request, response):
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

    def _redirect_url(self, request, response):
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

    def _redirect_headers(self, request, url, method):
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

    def _redirect_stream(self, request, method):
        """
        Return the body that should be used for the redirect request.
        """
        if method != request.method and method == 'GET':
            return None
        return request.stream

    def _set_timeout(self, request):
        if 'timeout' not in request.extensions:
            timeout = self.timeout if isinstance(self.timeout, UseClientDefault) else Timeout(self.timeout)
            request.extensions = dict(**request.extensions, timeout=timeout.as_dict())

class Client(BaseClient):
    """
    An HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.

    It can be shared between threads.

    Usage:

    ```python
    >>> client = httpx.Client()
    >>> response = client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(self, *, auth=None, params=None, headers=None, cookies=None, verify=True, cert=None, trust_env=True, http1=True, http2=False, proxy=None, mounts=None, timeout=DEFAULT_TIMEOUT_CONFIG, follow_redirects=False, limits=DEFAULT_LIMITS, max_redirects=DEFAULT_MAX_REDIRECTS, event_hooks=None, base_url='', transport=None, default_encoding='utf-8'):
        super().__init__(auth=auth, params=params, headers=headers, cookies=cookies, timeout=timeout, follow_redirects=follow_redirects, max_redirects=max_redirects, event_hooks=event_hooks, base_url=base_url, trust_env=trust_env, default_encoding=default_encoding)
        if http2:
            try:
                import h2
            except ImportError:
                raise ImportError("Using http2=True, but the 'h2' package is not installed. Make sure to install httpx using `pip install httpx[http2]`.") from None
        allow_env_proxies = trust_env and transport is None
        proxy_map = self._get_proxy_map(proxy, allow_env_proxies)
        self._transport = self._init_transport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, transport=transport)
        self._mounts = {URLPattern(key): None if proxy is None else self._init_proxy_transport(proxy, verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits) for key, proxy in proxy_map.items()}
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(self, verify=True, cert=None, trust_env=True, http1=True, http2=False, limits=DEFAULT_LIMITS, transport=None):
        if transport is not None:
            return transport
        return HTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)

    def _init_proxy_transport(self, proxy, verify=True, cert=None, trust_env=True, http1=True, http2=False, limits=DEFAULT_LIMITS):
        return HTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, proxy=proxy)

    def _transport_for_url(self, url):
        """
        Returns the transport instance that should be used for a given URL.
        This will either be the standard connection pool, or a proxy.
        """
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport
        return self._transport

    def request(self, method, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Build and send a request.

        Equivalent to:

        ```python
        request = client.build_request(...)
        response = client.send(request, ...)
        ```

        See `Client.build_request()`, `Client.send()` and
        [Merging of configuration][0] for how the various parameters
        are merged with client-level configuration.

        [0]: /advanced/clients/#merging-of-configuration
        """
        if cookies is not None:
            message = 'Setting per-request cookies=<...> is being deprecated, because the expected behaviour on cookie persistence is ambiguous. Set cookies directly on the client instance instead.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        return self.send(request, auth=auth, follow_redirects=follow_redirects)

    @contextmanager
    def stream(self, method, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Alternative to `httpx.request()` that streams the response body
        instead of loading it into memory at once.

        **Parameters**: See `httpx.request`.

        See also: [Streaming Responses][0]

        [0]: /quickstart#streaming-responses
        """
        request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        response = self.send(request=request, auth=auth, follow_redirects=follow_redirects, stream=True)
        try:
            yield response
        finally:
            response.close()

    def send(self, request, *, stream=False, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT):
        """
        Send a request.

        The request is sent as-is, unmodified.

        Typically you'll want to build one with `Client.build_request()`
        so that any client-level configuration is merged into the request,
        but passing an explicit `httpx.Request()` is supported as well.

        See also: [Request instances][0]

        [0]: /advanced/clients/#request-instances
        """
        if self._state == ClientState.CLOSED:
            raise RuntimeError('Cannot send a request, as the client has been closed.')
        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects
        self._set_timeout(request)
        auth = self._build_request_auth(request, auth)
        response = self._send_handling_auth(request, auth=auth, follow_redirects=follow_redirects, history=[])
        try:
            if not stream:
                response.read()
            return response
        except BaseException as exc:
            response.close()
            raise exc

    def _send_handling_auth(self, request, auth, follow_redirects, history):
        auth_flow = auth.sync_auth_flow(request)
        try:
            request = next(auth_flow)
            while True:
                response = self._send_handling_redirects(request, follow_redirects=follow_redirects, history=history)
                try:
                    try:
                        next_request = auth_flow.send(response)
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

    def _send_handling_redirects(self, request, follow_redirects, history):
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects('Exceeded maximum allowed redirects.', request=request)
            for hook in self._event_hooks['request']:
                hook(request)
            response = self._send_single_request(request)
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

    def _send_single_request(self, request):
        """
        Sends a single request, without handling any redirections.
        """
        transport = self._transport_for_url(request.url)
        start = time.perf_counter()
        if not isinstance(request.stream, SyncByteStream):
            raise RuntimeError('Attempted to send an async request with a sync Client instance.')
        with request_context(request=request):
            response = transport.handle_request(request)
        assert isinstance(response.stream, SyncByteStream)
        response.request = request
        response.stream = BoundSyncStream(response.stream, response=response, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding
        logger.info('HTTP Request: %s %s "%s %d %s"', request.method, request.url, response.http_version, response.status_code, response.reason_phrase)
        return response

    def get(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `GET` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def options(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send an `OPTIONS` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def head(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `HEAD` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def post(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `POST` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def put(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `PUT` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def patch(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `PATCH` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def delete(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `DELETE` request.

        **Parameters**: See `httpx.request`.
        """
        return self.request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    def close(self):
        """
        Close transport and proxies.
        """
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED
            self._transport.close()
            for transport in self._mounts.values():
                if transport is not None:
                    transport.close()

    def __enter__(self):
        if self._state != ClientState.UNOPENED:
            msg = {ClientState.OPENED: 'Cannot open a client instance more than once.', ClientState.CLOSED: 'Cannot reopen a client instance, once it has been closed.'}[self._state]
            raise RuntimeError(msg)
        self._state = ClientState.OPENED
        self._transport.__enter__()
        for transport in self._mounts.values():
            if transport is not None:
                transport.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self._state = ClientState.CLOSED
        self._transport.__exit__(exc_type, exc_value, traceback)
        for transport in self._mounts.values():
            if transport is not None:
                transport.__exit__(exc_type, exc_value, traceback)

class AsyncClient(BaseClient):
    """
    An asynchronous HTTP client, with connection pooling, HTTP/2, redirects,
    cookie persistence, etc.

    It can be shared between tasks.

    Usage:

    ```python
    >>> async with httpx.AsyncClient() as client:
    >>>     response = await client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(self, *, auth=None, params=None, headers=None, cookies=None, verify=True, cert=None, http1=True, http2=False, proxy=None, mounts=None, timeout=DEFAULT_TIMEOUT_CONFIG, follow_redirects=False, limits=DEFAULT_LIMITS, max_redirects=DEFAULT_MAX_REDIRECTS, event_hooks=None, base_url='', transport=None, trust_env=True, default_encoding='utf-8'):
        super().__init__(auth=auth, params=params, headers=headers, cookies=cookies, timeout=timeout, follow_redirects=follow_redirects, max_redirects=max_redirects, event_hooks=event_hooks, base_url=base_url, trust_env=trust_env, default_encoding=default_encoding)
        if http2:
            try:
                import h2
            except ImportError:
                raise ImportError("Using http2=True, but the 'h2' package is not installed. Make sure to install httpx using `pip install httpx[http2]`.") from None
        allow_env_proxies = trust_env and transport is None
        proxy_map = self._get_proxy_map(proxy, allow_env_proxies)
        self._transport = self._init_transport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, transport=transport)
        self._mounts = {URLPattern(key): None if proxy is None else self._init_proxy_transport(proxy, verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits) for key, proxy in proxy_map.items()}
        if mounts is not None:
            self._mounts.update({URLPattern(key): transport for key, transport in mounts.items()})
        self._mounts = dict(sorted(self._mounts.items()))

    def _init_transport(self, verify=True, cert=None, trust_env=True, http1=True, http2=False, limits=DEFAULT_LIMITS, transport=None):
        if transport is not None:
            return transport
        return AsyncHTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits)

    def _init_proxy_transport(self, proxy, verify=True, cert=None, trust_env=True, http1=True, http2=False, limits=DEFAULT_LIMITS):
        return AsyncHTTPTransport(verify=verify, cert=cert, trust_env=trust_env, http1=http1, http2=http2, limits=limits, proxy=proxy)

    def _transport_for_url(self, url):
        """
        Returns the transport instance that should be used for a given URL.
        This will either be the standard connection pool, or a proxy.
        """
        for pattern, transport in self._mounts.items():
            if pattern.matches(url):
                return self._transport if transport is None else transport
        return self._transport

    async def request(self, method, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Build and send a request.

        Equivalent to:

        ```python
        request = client.build_request(...)
        response = await client.send(request, ...)
        ```

        See `AsyncClient.build_request()`, `AsyncClient.send()`
        and [Merging of configuration][0] for how the various parameters
        are merged with client-level configuration.

        [0]: /advanced/clients/#merging-of-configuration
        """
        if cookies is not None:
            message = 'Setting per-request cookies=<...> is being deprecated, because the expected behaviour on cookie persistence is ambiguous. Set cookies directly on the client instance instead.'
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        return await self.send(request, auth=auth, follow_redirects=follow_redirects)

    @asynccontextmanager
    async def stream(self, method, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Alternative to `httpx.request()` that streams the response body
        instead of loading it into memory at once.

        **Parameters**: See `httpx.request`.

        See also: [Streaming Responses][0]

        [0]: /quickstart#streaming-responses
        """
        request = self.build_request(method=method, url=url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, timeout=timeout, extensions=extensions)
        response = await self.send(request=request, auth=auth, follow_redirects=follow_redirects, stream=True)
        try:
            yield response
        finally:
            await response.aclose()

    async def send(self, request, *, stream=False, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT):
        """
        Send a request.

        The request is sent as-is, unmodified.

        Typically you'll want to build one with `AsyncClient.build_request()`
        so that any client-level configuration is merged into the request,
        but passing an explicit `httpx.Request()` is supported as well.

        See also: [Request instances][0]

        [0]: /advanced/clients/#request-instances
        """
        if self._state == ClientState.CLOSED:
            raise RuntimeError('Cannot send a request, as the client has been closed.')
        self._state = ClientState.OPENED
        follow_redirects = self.follow_redirects if isinstance(follow_redirects, UseClientDefault) else follow_redirects
        self._set_timeout(request)
        auth = self._build_request_auth(request, auth)
        response = await self._send_handling_auth(request, auth=auth, follow_redirects=follow_redirects, history=[])
        try:
            if not stream:
                await response.aread()
            return response
        except BaseException as exc:
            await response.aclose()
            raise exc

    async def _send_handling_auth(self, request, auth, follow_redirects, history):
        auth_flow = auth.async_auth_flow(request)
        try:
            request = await auth_flow.__anext__()
            while True:
                response = await self._send_handling_redirects(request, follow_redirects=follow_redirects, history=history)
                try:
                    try:
                        next_request = await auth_flow.asend(response)
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

    async def _send_handling_redirects(self, request, follow_redirects, history):
        while True:
            if len(history) > self.max_redirects:
                raise TooManyRedirects('Exceeded maximum allowed redirects.', request=request)
            for hook in self._event_hooks['request']:
                await hook(request)
            response = await self._send_single_request(request)
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

    async def _send_single_request(self, request):
        """
        Sends a single request, without handling any redirections.
        """
        transport = self._transport_for_url(request.url)
        start = time.perf_counter()
        if not isinstance(request.stream, AsyncByteStream):
            raise RuntimeError('Attempted to send an sync request with an AsyncClient instance.')
        with request_context(request=request):
            response = await transport.handle_async_request(request)
        assert isinstance(response.stream, AsyncByteStream)
        response.request = request
        response.stream = BoundAsyncStream(response.stream, response=response, start=start)
        self.cookies.extract_cookies(response)
        response.default_encoding = self._default_encoding
        logger.info('HTTP Request: %s %s "%s %d %s"', request.method, request.url, response.http_version, response.status_code, response.reason_phrase)
        return response

    async def get(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `GET` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('GET', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def options(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send an `OPTIONS` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('OPTIONS', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def head(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `HEAD` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('HEAD', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def post(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `POST` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('POST', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def put(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `PUT` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('PUT', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def patch(self, url, *, content=None, data=None, files=None, json=None, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `PATCH` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('PATCH', url, content=content, data=data, files=files, json=json, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def delete(self, url, *, params=None, headers=None, cookies=None, auth=USE_CLIENT_DEFAULT, follow_redirects=USE_CLIENT_DEFAULT, timeout=USE_CLIENT_DEFAULT, extensions=None):
        """
        Send a `DELETE` request.

        **Parameters**: See `httpx.request`.
        """
        return await self.request('DELETE', url, params=params, headers=headers, cookies=cookies, auth=auth, follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

    async def aclose(self):
        """
        Close transport and proxies.
        """
        if self._state != ClientState.CLOSED:
            self._state = ClientState.CLOSED
            await self._transport.aclose()
            for proxy in self._mounts.values():
                if proxy is not None:
                    await proxy.aclose()

    async def __aenter__(self):
        if self._state != ClientState.UNOPENED:
            msg = {ClientState.OPENED: 'Cannot open a client instance more than once.', ClientState.CLOSED: 'Cannot reopen a client instance, once it has been closed.'}[self._state]
            raise RuntimeError(msg)
        self._state = ClientState.OPENED
        await self._transport.__aenter__()
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aenter__()
        return self

    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None):
        self._state = ClientState.CLOSED
        await self._transport.__aexit__(exc_type, exc_value, traceback)
        for proxy in self._mounts.values():
            if proxy is not None:
                await proxy.__aexit__(exc_type, exc_value, traceback)