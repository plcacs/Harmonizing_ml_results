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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union, cast
import rfc3986
from .core._http._backends.trio_backend import TrioBackend
from .http_auth import _basic_auth_str
from ._basics import cookielib, urljoin, urlparse, str
from .http_cookies import cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar, merge_cookies, _copy_cookie_jar
from .http_models import Request, PreparedRequest, DEFAULT_REDIRECT_LIMIT
from ._hooks import default_hooks, dispatch_hook
from ._internal_utils import to_native_string
from .http_utils import to_key_val_list, default_headers, DEFAULT_PORTS
from .exceptions import TooManyRedirects, InvalidScheme, ChunkedEncodingError, ConnectionError, ContentDecodingError, InvalidHeader
from ._structures import CaseInsensitiveDict
from .http_adapters import HTTPAdapter, AsyncHTTPAdapter
from .http_utils import requote_uri, get_environ_proxies, get_netrc_auth, should_bypass_proxies, get_auth_from_url, is_valid_location, rewind_body
from .http_stati import codes
from .http_models import REDIRECT_STATI

if sys.platform == 'win32':
    try:
        preferred_clock = time.perf_counter
    except AttributeError:
        preferred_clock = time.clock
else:
    preferred_clock = time.time

def merge_setting(request_setting: Any, session_setting: Any, dict_class: Type[Dict[Any, Any]] = OrderedDict) -> Any:
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`.
    """
    if session_setting is None:
        return request_setting
    if request_setting is None:
        return session_setting
    if not (isinstance(session_setting, Mapping) and isinstance(request_setting, Mapping)):
        return request_setting
    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))
    none_keys = [k for k, v in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]
    return merged_setting

def merge_hooks(request_hooks: Optional[Dict[str, List[Any]]], session_hooks: Optional[Dict[str, List[Any]]], dict_class: Type[Dict[Any, Any]] = OrderedDict) -> Optional[Dict[str, List[Any]]]:
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get('response') == []:
        return request_hooks
    if request_hooks is None or request_hooks.get('response') == []:
        return session_hooks
    return merge_setting(request_hooks, session_hooks, dict_class)

class SessionRedirectMixin(object):

    def get_redirect_target(self, response: Any) -> Optional[str]:
        """Receives a Response. Returns a redirect URI or ``None``"""
        if response.is_redirect:
            if not is_valid_location(response):
                raise InvalidHeader('Response contains multiple Location headers. Unable to perform redirect.')
            location = response.headers['location']
            location = location.encode('latin1')
            return to_native_string(location, 'utf8')
        return None

    def resolve_redirects(self, response: Any, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: bool = True, cert: Optional[Union[str, Tuple[str, str]]] = None, proxies: Optional[Dict[str, str]] = None, yield_requests: bool = False, **adapter_kwargs: Any) -> Iterator[Any]:
        """Given a Response, yields Responses until 'Location' header-based
        redirection ceases, or the Session.max_redirects limit has been
        reached.
        """
        history = [response]
        location_url = self.get_redirect_target(response)
        while location_url:
            prepared_request = request.copy()
            try:
                response.content
            except (ChunkedEncodingError, ConnectionError, ContentDecodingError, RuntimeError):
                response.raw.read(decode_content=False)
            if len(response.history) >= self.max_redirects:
                raise TooManyRedirects('Exceeded %s redirects.' % self.max_redirects, response=response)
            response.close()
            if location_url.startswith('//'):
                parsed_rurl = urlparse(response.url)
                location_url = '%s:%s' % (to_native_string(parsed_rurl.scheme), location_url)
            parsed = urlparse(location_url)
            location_url = parsed.geturl()
            if not parsed.netloc:
                location_url = urljoin(response.url, requote_uri(location_url))
            else:
                location_url = requote_uri(location_url)
            prepared_request.url = to_native_string(location_url)
            method_changed = self.rebuild_method(prepared_request, response)
            if method_changed and prepared_request.method == 'GET':
                purged_headers = ('Content-Length', 'Content-Type', 'Transfer-Encoding')
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None
            headers = prepared_request.headers
            try:
                del headers['Cookie']
            except KeyError:
                pass
            extract_cookies_to_jar(prepared_request._cookies, request, response.raw)
            merge_cookies(prepared_request._cookies, self.cookies)
            prepared_request.prepare_cookies(prepared_request._cookies)
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, response)
            rewindable = prepared_request._body_position is not None and ('Content-Length' in headers or 'Transfer-Encoding' in headers)
            if rewindable:
                rewind_body(prepared_request)
            request = prepared_request
            if yield_requests:
                yield request
            else:
                response = self.send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies, allow_redirects=False, **adapter_kwargs)
                response.history = history[:]
                history.append(response)
                extract_cookies_to_jar(self.cookies, prepared_request, response.raw)
                location_url = self.get_redirect_target(response)
                yield response

    def rebuild_auth(self, prepared_request: Any, response: Any) -> None:
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently
        removes
        and reapplies authentication where possible to avoid credential loss.
        """
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = urlparse(response.request.url)
            redirect_parsed = urlparse(url)
            if original_parsed.hostname != redirect_parsed.hostname:
                del headers['Authorization']
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)
        return

    def rebuild_proxies(self, prepared_request: Any, proxies: Optional[Dict[str, str]]) -> Dict[str, str]:
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
        new_proxies = proxies.copy()
        no_proxy = proxies.get('no_proxy')
        bypass_proxy = should_bypass_proxies(url, no_proxy=no_proxy)
        if self.trust_env and (not bypass_proxy):
            environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            proxy = environ_proxies.get(scheme, environ_proxies.get('all'))
            if proxy:
                new_proxies.setdefault(scheme, proxy)
        if 'Proxy-Authorization' in headers:
            del headers['Proxy-Authorization']
        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = (None, None)
        if username and password:
            headers['Proxy-Authorization'] = _basic_auth_str(username, password)
        return new_proxies

    def rebuild_method(self, prepared_request: Any, response: Any) -> bool:
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.

        :rtype bool:
        :return: boolean expressing if the method changed during rebuild.
        """
        method = original_method = prepared_request.method
        if response.status_code == codes.see_other and method != 'HEAD':
            method = 'GET'
        if response.status_code in (codes.found, codes.moved) and method == 'POST':
            method = 'GET'
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
    __slots__ = ['headers', 'cookies', 'auth', 'proxies', 'hooks', 'params', 'verify', 'cert', 'prefetch', 'adapters', 'stream', 'trust_env', 'max_redirects']

    def __init__(self) -> None:
        self.headers = default_headers()
        self.auth = None
        self.proxies = {}
        self.hooks = default_hooks()
        self.params = {}
        self.stream = False
        self.verify = True
        self.cert = None
        self.max_redirects = DEFAULT_REDIRECT_LIMIT
        self.trust_env = True
        self.cookies = cookiejar_from_dict({})
        self.adapters = OrderedDict()
        self.mount('https://', HTTPAdapter())
        self.mount('http://', HTTPAdapter())

    def __enter__(self) -> 'HTTPSession':
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
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)
        session_cookies = _copy_cookie_jar(self.cookies)
        merged_cookies = merge_cookies(session_cookies, cookies)
        auth = request.auth
        if self.trust_env and (not auth) and (not self.auth):
            auth = get_netrc_auth(request.url)
        p = PreparedRequest()
        p.prepare(method=request.method.upper(), url=request.url, files=request.files, data=request.data, json=request.json, headers=merge_setting(request.headers, self.headers, dict_class=CaseInsensitiveDict), params=merge_setting(request.params, self.params), auth=merge_setting(auth, self.auth), cookies=merged_cookies, hooks=merge_hooks(request.hooks, self.hooks))
        return p

    def request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Union[Dict[str, Any], bytes, str]] = None, headers: Optional[Dict[str, str]] = None, cookies: Optional[Union[Dict[str, str], cookielib.CookieJar]] = None, files: Optional[Dict[str, Any]] = None, auth: Optional[Tuple[str, str]] = None, timeout: Optional[Union[float, Tuple[float, float]]] = None, allow_redirects: bool = True, proxies: Optional[Dict[str, str]] = None, hooks: Optional[Dict[str, List[Any]]] = None, stream: Optional[bool] = None, verify: Optional[Union[bool, str]] = None, cert: Optional[Union[str, Tuple[str, str]]] = None, json: Optional[Any] = None) -> Any:
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
        req = Request(method=method.upper(), url=url, headers=headers, files=files, data=data or {}, json=json, params=params or {}, auth=auth, cookies=cookies, hooks=hooks)
        prep = self.prepare_request(req)
        proxies = proxies or {}
        settings = self.merge_environment_settings(prep.url, proxies, stream, verify, cert)
        send_kwargs = {'timeout': timeout, 'allow_redirects': allow_redirects}
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)
        return resp

    def get(self, url: str, **kwargs: Any) -> Any:
        """Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \\*\\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault('allow_redirects', True)
        return self.request('GET', url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> Any:
        """Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \\*\\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        kwargs.setdefault('allow_redirects', False)
        return self.request('HEAD', url, **kwargs)

    def send(self, request: PreparedRequest, **kwargs: Any) -> Any:
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        kwargs.setdefault('stream', self.stream)
        kwargs.setdefault('verify', self.verify)
        kwargs.setdefault('cert', self.cert)
        kwargs.setdefault('proxies', self.proxies)
        if isinstance(request, Request):
            raise ValueError('You can only send PreparedRequests.')
        allow_redirects = kwargs.pop('allow_redirects', True)
        stream = kwargs.get('stream')
        hooks = request.hooks
        adapter = self.get_adapter(url=request.url)
        start = preferred_clock()
        r = adapter.send(request, **kwargs)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)
        r = dispatch_hook('response', hooks, r, **kwargs)
        if r.history:
            for resp in r.history:
                extract_c