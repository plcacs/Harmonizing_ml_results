# -*- coding: utf-8 -*-
"""
requests.http_session
~~~~~~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
"""
import os
import sys
import time
from collections import Mapping, OrderedDict
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import rfc3986
from .core._http._backends.trio_backend import TrioBackend

from .http_auth import _basic_auth_str
from ._basics import cookielib, urljoin, urlparse, str
from .http_cookies import (
    cookiejar_from_dict,
    extract_cookies_to_jar,
    RequestsCookieJar,
    merge_cookies,
    _copy_cookie_jar,
)
from .http_models import Request, PreparedRequest, DEFAULT_REDIRECT_LIMIT, Response
from ._hooks import default_hooks, dispatch_hook
from ._internal_utils import to_native_string
from .http_utils import to_key_val_list, default_headers, DEFAULT_PORTS, requote_uri, get_environ_proxies, get_netrc_auth, should_bypass_proxies, get_auth_from_url, is_valid_location, rewind_body
from .exceptions import (
    TooManyRedirects,
    InvalidScheme,
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    InvalidHeader,
)
from ._structures import CaseInsensitiveDict
from .http_adapters import HTTPAdapter, AsyncHTTPAdapter

from .http_stati import codes

# formerly defined here, reexposed here for backward compatibility
from .http_models import REDIRECT_STATI

# Preferred clock, based on which one is more accurate on a given system.
if sys.platform == "win32":
    try:  # Python 3.4+
        preferred_clock: Callable[[], float] = time.perf_counter
    except AttributeError:  # Earlier than Python 3.
        preferred_clock = time.clock  # type: ignore
else:
    preferred_clock = time.time  # type: ignore


def merge_setting(
    request_setting: Optional[Any],
    session_setting: Optional[Any],
    dict_class: Callable[..., Any] = OrderedDict,
) -> Any:
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`.
    """
    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
        isinstance(session_setting, Mapping)
        and isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))
    # Remove keys that are set to None. Extract keys first to avoid altering
    # the dictionary during iteration.
    none_keys = [k for (k, v) in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]
    return merged_setting


def merge_hooks(
    request_hooks: Optional[Dict[str, List[Callable[..., Any]]]],
    session_hooks: Optional[Dict[str, List[Callable[..., Any]]]],
    dict_class: Callable[..., Any] = OrderedDict,
) -> Optional[Dict[str, List[Callable[..., Any]]]]:
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get("response") == []:
        return request_hooks

    if request_hooks is None or request_hooks.get("response") == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)


class SessionRedirectMixin:
    def get_redirect_target(self, response: Response) -> Optional[str]:
        """Receives a Response. Returns a redirect URI or ``None``"""
        # Due to the nature of how requests processes redirects this method will
        # be called at least once upon the original response and at least twice
        # on each subsequent redirect response (if any).
        # If a custom mixin is used to handle this logic, it may be advantageous
        # to cache the redirect location onto the response object as a private
        # attribute.
        if response.is_redirect:
            if not is_valid_location(response):
                raise InvalidHeader(
                    "Response contains multiple Location headers. "
                    "Unable to perform redirect."
                )

            location = response.headers["location"]
            # Currently the underlying http module on py3 decode headers
            # in latin1, but empirical evidence suggests that latin1 is very
            # rarely used with non-ASCII characters in HTTP headers.
            # It is more likely to get UTF8 header rather than latin1.
            # This causes incorrect handling of UTF8 encoded location headers.
            # To solve this, we re-encode the location in latin1.
            location = location.encode("latin1")

            return to_native_string(location, "utf8")

        return None

    def resolve_redirects(
        self,
        response: Response,
        request: PreparedRequest,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, str]] = None,
        yield_requests: bool = False,
        **adapter_kwargs: Any,
    ) -> Generator[Response, None, None]:
        """Given a Response, yields Responses until 'Location' header-based
        redirection ceases, or the Session.max_redirects limit has been
        reached.
        """
        history: List[Response] = [response]  # keep track of history; seed it with the original response
        location_url: Optional[str] = self.get_redirect_target(response)
        while location_url:
            prepared_request = request.copy()
            try:
                _ = response.content  # Consume socket so it can be released
            except (
                ChunkedEncodingError,
                ConnectionError,
                ContentDecodingError,
                RuntimeError,
            ):
                response.raw.read(decode_content=False)
            if len(response.history) >= self.max_redirects:
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=response
                )

            # Release the connection back into the pool.
            response.close()
            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if location_url.startswith("//"):
                parsed_rurl = urlparse(response.url)
                location_url = f"{to_native_string(parsed_rurl.scheme)}:{location_url}"
            # The scheme should be lower case...
            parsed = urlparse(location_url)
            location_url = parsed.geturl()
            # Facilitate relative 'location' headers, as allowed by RFC 7231.
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not parsed.netloc:
                location_url = urljoin(response.url, requote_uri(location_url))
            else:
                location_url = requote_uri(location_url)
            prepared_request.url = to_native_string(location_url)
            method_changed: bool = self.rebuild_method(prepared_request, response)
            # https://github.com/kennethreitz/requests/issues/2590
            # If method is changed to GET we need to remove body and associated headers.
            if method_changed and prepared_request.method == "GET":
                # https://github.com/requests/requests/issues/3490
                purged_headers = (
                    "Content-Length",
                    "Content-Type",
                    "Transfer-Encoding",
                )
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None
            headers = prepared_request.headers
            try:
                del headers["Cookie"]
            except KeyError:
                pass
            # Extract any cookies sent on the response to the cookiejar
            # in the new request. Because we've mutated our copied prepared
            # request, use the old one that we haven't yet touched.
            extract_cookies_to_jar(
                prepared_request._cookies, request, response.raw
            )
            merge_cookies(prepared_request._cookies, self.cookies)
            prepared_request.prepare_cookies(prepared_request._cookies)
            # Rebuild auth and proxy information.
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, response)
            # A failed tell() sets `_body_position` to `object()`. This non-None
            # value ensures `rewindable` will be True, allowing us to raise an
            # UnrewindableBodyError, instead of hanging the connection.
            rewindable = prepared_request._body_position is not None and (
                "Content-Length" in headers or "Transfer-Encoding" in headers
            )
            # Attempt to rewind consumed file-like object.
            if rewindable:
                rewind_body(prepared_request)
            # Override the original request.
            request = prepared_request
            if yield_requests:
                yield request

            else:
                response = self.send(
                    request,
                    stream=stream,
                    timeout=timeout,
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                    allow_redirects=False,
                    **adapter_kwargs,
                )
                # copy our history tracker into the response
                response.history = history[:]
                # append the new response to the history tracker for the next iteration
                history.append(response)
                extract_cookies_to_jar(
                    self.cookies, prepared_request, response.raw
                )
                # extract redirect url, if any, for the next loop
                location_url = self.get_redirect_target(response)
                yield response

    def rebuild_auth(self, prepared_request: PreparedRequest, response: Response) -> None:
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently
        removes
        and reapplies authentication where possible to avoid credential loss.
        """
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            # If we get redirected to a new host, we should strip out any
            # authentication headers.
            original_parsed = urlparse(response.request.url)
            redirect_parsed = urlparse(url)
            if original_parsed.hostname != redirect_parsed.hostname:
                del headers["Authorization"]
        # .netrc might have more auth for us on our new host.
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)
        return

    def rebuild_proxies(
        self, prepared_request: PreparedRequest, proxies: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """This method re-evaluates the proxy configuration by
        considering the environment variables. If we are redirected to a
        URL covered by NO_PROXY, we strip the proxy configuration.
        Otherwise, we set missing proxy keys for this URL (in case they
        were stripped by a previous redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
        proxies = proxies if proxies is not None else {}
        headers = prepared_request.headers
        url = prepared_request.url
        scheme = urlparse(url).scheme
        new_proxies: Dict[str, str] = proxies.copy()
        no_proxy = proxies.get("no_proxy")
        bypass_proxy = should_bypass_proxies(url, no_proxy=no_proxy)
        if self.trust_env and not bypass_proxy:
            environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            proxy = environ_proxies.get(scheme, environ_proxies.get("all"))
            if proxy:
                new_proxies.setdefault(scheme, proxy)
        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]
        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None
        if username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(
                username, password
            )
        return new_proxies

    def rebuild_method(
        self, prepared_request: PreparedRequest, response: Response
    ) -> bool:
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.

        :rtype bool:
        :return: boolean expressing if the method changed during rebuild.
        """
        method = original_method = prepared_request.method
        # http://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.see_other and method != "HEAD":
            method = "GET"
        # If a POST is responded to with a 301 or 302, turn it into a GET. This has
        # become a common pattern in browsers and was introduced into later versions
        # of HTTP RFCs. While some browsers transform other methods to GET, little of
        # that has been standardized. For that reason, we're using curl as a model
        # which only supports POST->GET.
        if (
            response.status_code in (codes.found, codes.moved)
            and method == "POST"
        ):
            method = "GET"
        prepared_request.method = method
        return method != original_method


class HTTPSession(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.request('get', 'https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      >>>     s.request('get', 'https://httpbin.org/get')
      <Response [200]>
    """

    headers: CaseInsensitiveDict
    cookies: RequestsCookieJar
    auth: Optional[Any]
    proxies: Dict[str, str]
    hooks: Dict[str, List[Callable[..., Any]]]
    params: Dict[str, Any]
    verify: Union[bool, str]
    cert: Optional[Union[str, Tuple[str, str]]]
    prefetch: Any
    adapters: OrderedDict
    stream: bool
    trust_env: bool
    max_redirects: int

    __slots__ = [
        "headers",
        "cookies",
        "auth",
        "proxies",
        "hooks",
        "params",
        "verify",
        "cert",
        "prefetch",
        "adapters",
        "stream",
        "trust_env",
        "max_redirects",
    ]

    def __init__(self) -> None:
        # : A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers: CaseInsensitiveDict = default_headers()
        # : Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth: Optional[Any] = None
        # : Dictionary mapping protocol or protocol and host to the URL of the proxy
        #: (e.g. {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}) to
        #: be used on each :class:`Request <Request>`.
        self.proxies: Dict[str, str] = {}
        # : Event-handling hooks.
        self.hooks: Dict[str, List[Callable[..., Any]]] = default_hooks()
        # : Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params: Dict[str, Any] = {}
        # : Stream response content default.
        self.stream: bool = False
        # : SSL Verification default.
        self.verify: Union[bool, str] = True
        # : SSL client certificate default, if String, path to ssl client
        #: cert file (.pem). If Tuple, ('cert', 'key') pair.
        self.cert: Optional[Union[str, Tuple[str, str]]] = None
        # : Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        #: This defaults to requests.models.DEFAULT_REDIRECT_LIMIT, which is
        #: 30.
        self.max_redirects: int = DEFAULT_REDIRECT_LIMIT
        # : Trust environment settings for proxy configuration, default
        #: authentication and similar.
        self.trust_env: bool = True
        # : A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies: RequestsCookieJar = cookiejar_from_dict({})
        # Default connection adapters.
        self.adapters: OrderedDict = OrderedDict()
        self.mount("https://", HTTPAdapter())
        self.mount("http://", HTTPAdapter())

    def __enter__(self) -> "HTTPSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def prepare_request(self, request: Request) -> PreparedRequest:
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
            Session's settings.
        :rtype: requests.PreparedRequest
        """
        cookies = request.cookies or {}
        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)
        # Merge with session cookies
        session_cookies = _copy_cookie_jar(self.cookies)
        merged_cookies = merge_cookies(session_cookies, cookies)
        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(request.url)
        p = PreparedRequest()
        p.prepare(
            method=request.method.upper(),
            url=request.url,
            files=request.files,
            data=request.data,
            json=request.json,
            headers=merge_setting(
                request.headers, self.headers, dict_class=CaseInsensitiveDict
            ),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Union[Dict[str, Any], Iterable[Tuple[str, Any]]]] = None,
        data: Optional[Union[Dict[str, Any], List[Tuple[str, Any]], bytes, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Union[Dict[str, Any], cookielib.CookieJar]] = None,
        files: Optional[Dict[str, Any]] = None,
        auth: Optional[Any] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        allow_redirects: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        hooks: Optional[Dict[str, List[Callable[..., Any]]]] = None,
        stream: Optional[bool] = None,
        verify: Optional[Union[bool, str]] = None,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        json: Optional[Any] = None,
    ) -> Response:
        """Constructs a :class:`Request <Request>`, prepares it, and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)
        proxies = proxies or {}
        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )
        # Send the request.
        send_kwargs: Dict[str, Any] = {"timeout": timeout, "allow_redirects": allow_redirects}
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)
        return resp

    def get(self, url: str, **kwargs: Any) -> Response:
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> Response:
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def send(
        self,
        request: PreparedRequest,
        stream: Optional[bool] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Optional[Union[bool, str]] = None,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, str]] = None,
        allow_redirects: bool = True,
        **kwargs: Any,
    ) -> Response:
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        stream = self.stream if stream is None else stream
        verify = self.verify if verify is None else verify
        cert = self.cert if cert is None else cert
        proxies = self.proxies if proxies is None else proxies
        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        # Set up variables needed for resolve_redirects and dispatching of
        # hooks
        allow_redirects = allow_redirects
        stream = stream
        hooks = request.hooks
        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)
        # Start time (approximately) of the request
        start = preferred_clock()
        # Send the request
        r = adapter.send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies, **kwargs)
        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)
        # Response manipulation hooks.
        r = dispatch_hook("response", hooks, r, **kwargs)
        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)
        extract_cookies_to_jar(self.cookies, request, r.raw)
        # Redirect resolving generator.
        gen = self.resolve_redirects(r, request, **kwargs)
        # Resolve redirects, if allowed.
        history = [resp for resp in gen] if allow_redirects else []
        # If there is a history, replace ``r`` with the last response
        if history:
            r = history.pop()
        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(
                    self.resolve_redirects(
                        r, request, yield_requests=True, **kwargs
                    )
                )
            except StopIteration:
                pass
        if not stream:
            _ = r.content
        return r

    def merge_environment_settings(
        self,
        url: str,
        proxies: Dict[str, str],
        stream: Optional[bool],
        verify: Optional[Union[bool, str]],
        cert: Optional[Union[str, Tuple[str, str]]],
    ) -> Dict[str, Any]:
        """
        Check the environment and merge it with some settings.

        :rtype: dict
        """
        # Merge all the kwargs except for proxies.
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)
        # Gather clues from the surrounding environment.
        # We do this after merging the Session values to make sure we don't
        # accidentally exclude them.
        if self.trust_env:
            # Look for requests environment configuration and be compatible
            # with cURL.
            if verify is True or verify is None:
                verify = (
                    os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("CURL_CA_BUNDLE")
                    or verify
                )
        # Now we handle proxies.
        # Proxies need to be built up backwards. This is because None values
        # can delete proxy information, which can then be re-added by a more
        # specific layer. So we begin by getting the environment's proxies,
        # then add the Session, then add the request.
        no_proxy = proxies.get("no_proxy") if proxies is not None else None
        if no_proxy is None:
            no_proxy = self.proxies.get("no_proxy")
        env_proxies: Dict[str, str] = {}
        if self.trust_env:
            env_proxies = get_environ_proxies(url, no_proxy=no_proxy) or {}
        new_proxies = merge_setting(self.proxies, env_proxies)
        proxies = merge_setting(proxies, new_proxies)
        return {
            "verify": verify,
            "proxies": proxies,
            "stream": stream,
            "cert": cert,
        }

    def get_adapter(self, url: str) -> HTTPAdapter:
        """
        Returns the appropriate connection adapter for the given URL.

        :rtype: requests.adapters.BaseAdapter
        """
        for (prefix, adapter) in self.adapters.items():
            if url.lower().startswith(prefix):
                return adapter

        # Nothing matches :-/
        raise InvalidScheme(f"No connection adapters were found for '{url}'")

    def close(self) -> None:
        """Closes all adapters and, as such, the Session."""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix: str, adapter: HTTPAdapter) -> None:
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by prefix length.
        """
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]
        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self) -> Dict[str, Any]:
        state = {attr: getattr(self, attr, None) for attr in self.__slots__}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for attr, value in state.items():
            setattr(self, attr, value)


class AsyncHTTPSession(HTTPSession):
    def __init__(self, backend: Optional[Any] = None) -> None:
        super(AsyncHTTPSession, self).__init__()

        self.mount("https://", AsyncHTTPAdapter())
        self.mount("http://", AsyncHTTPAdapter())

    async def get(self, url: str, **kwargs: Any) -> Response:
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault("allow_redirects", True)
        return await self.request("get", url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> Response:
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault("allow_redirects", False)
        return await self.request("HEAD", url, **kwargs)

    async def request(
        self,
        method: str,
        url: str,
        params: Optional[Union[Dict[str, Any], Iterable[Tuple[str, Any]]]] = None,
        data: Optional[Union[Dict[str, Any], List[Tuple[str, Any]], bytes, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Union[Dict[str, Any], cookielib.CookieJar]] = None,
        files: Optional[Dict[str, Any]] = None,
        auth: Optional[Any] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        allow_redirects: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        hooks: Optional[Dict[str, List[Callable[..., Any]]]] = None,
        stream: Optional[bool] = None,
        verify: Optional[Union[bool, str]] = None,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        json: Optional[Any] = None,
    ) -> Response:
        """Constructs a :class:`Request <Request>`, prepares it, and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, bytes, or file-like object to send
            in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)
        proxies = proxies or {}
        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )
        # Send the request.
        send_kwargs: Dict[str, Any] = {"timeout": timeout, "allow_redirects": allow_redirects}
        send_kwargs.update(settings)
        resp = await self.send(prep, **send_kwargs)
        return resp

    async def send(
        self,
        request: PreparedRequest,
        stream: Optional[bool] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Optional[Union[bool, str]] = None,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, str]] = None,
        allow_redirects: bool = True,
        **kwargs: Any,
    ) -> Response:
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        stream = self.stream if stream is None else stream
        verify = self.verify if verify is None else verify
        cert = self.cert if cert is None else cert
        proxies = self.proxies if proxies is None else proxies
        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        # Set up variables needed for resolve_redirects and dispatching of
        # hooks
        allow_redirects = allow_redirects
        stream = stream
        hooks = request.hooks
        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)
        # Start time (approximately) of the request
        start = preferred_clock()
        # Send the request
        r = await adapter.send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies, **kwargs)
        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)
        # Response manipulation hooks.
        r = dispatch_hook("response", hooks, r, **kwargs)
        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)
        extract_cookies_to_jar(self.cookies, request, r.raw)
        # Redirect resolving generator.
        gen = self.resolve_redirects(r, request, **kwargs)
        # Resolve redirects, if allowed.
        history = [resp for resp in gen] if allow_redirects else []
        # If there is a history, replace ``r`` with the last response
        if history:
            r = history.pop()
        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(
                    self.resolve_redirects(
                        r, request, yield_requests=True, **kwargs
                    )
                )
            except StopIteration:
                pass
        if not stream:
            await r.content
        return r
