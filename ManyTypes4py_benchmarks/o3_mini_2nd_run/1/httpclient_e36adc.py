#!/usr/bin/env python
"""
Blocking and non-blocking HTTP client interfaces.

This module defines a common interface shared by two implementations,
``simple_httpclient`` and ``curl_httpclient``.  Applications may either
instantiate their chosen implementation class directly or use the
`AsyncHTTPClient` class from this module, which selects an implementation
that can be overridden with the `AsyncHTTPClient.configure` method.

The default implementation is ``simple_httpclient``, and this is expected
to be suitable for most users' needs.  However, some applications may wish
to switch to ``curl_httpclient`` for reasons such as the following:

* ``curl_httpclient`` has some features not found in ``simple_httpclient``,
  including support for HTTP proxies and the ability to use a specified
  network interface.

* ``curl_httpclient`` is more likely to be compatible with sites that are
  not-quite-compliant with the HTTP spec, or sites that use little-exercised
  features of HTTP.

* ``curl_httpclient`` is faster.

Note that if you are using ``curl_httpclient``, it is highly
recommended that you use a recent version of ``libcurl`` and
``pycurl``.  Currently the minimum supported version of libcurl is
7.22.0, and the minimum version of pycurl is 7.18.2.  It is highly
recommended that your ``libcurl`` installation is built with
asynchronous DNS resolver (threaded or c-ares), otherwise you may
encounter various problems with request timeouts (for more
information, see
http://curl.haxx.se/libcurl/c/curl_easy_setopt.html#CURLOPTCONNECTTIMEOUTMS
and comments in curl_httpclient.py).

To select ``curl_httpclient``, call `AsyncHTTPClient.configure` at startup::

    AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")
"""
import datetime
import functools
from io import BytesIO
import ssl
import time
import weakref
from typing import Any, Callable, Dict, Optional, Union, Type

from tornado.concurrent import Future, future_set_result_unless_cancelled, future_set_exception_unless_cancelled
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable

# Forward declarations for type annotations:
class HTTPRequest: pass
class HTTPResponse: pass

class HTTPClient:
    """A blocking HTTP client."""

    def __init__(self, async_client_class: Optional[Type["AsyncHTTPClient"]] = None, **kwargs: Any) -> None:
        self._closed: bool = True
        self._io_loop: IOLoop = IOLoop(make_current=False)
        if async_client_class is None:
            async_client_class = AsyncHTTPClient

        async def make_client() -> "AsyncHTTPClient":
            await gen.sleep(0)
            assert async_client_class is not None
            return async_client_class(**kwargs)
        self._async_client: AsyncHTTPClient = self._io_loop.run_sync(make_client)
        self._closed = False

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Closes the HTTPClient, freeing any resources used."""
        if not self._closed:
            self._async_client.close()
            self._io_loop.close()
            self._closed = True

    def fetch(self, request: Union[str, HTTPRequest], **kwargs: Any) -> "HTTPResponse":
        """Executes a request, returning an HTTPResponse."""
        response: HTTPResponse = self._io_loop.run_sync(functools.partial(self._async_client.fetch, request, **kwargs))
        return response

class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client."""

    _instance_cache: Any = None

    @classmethod
    def configurable_base(cls) -> Type["AsyncHTTPClient"]:
        return AsyncHTTPClient

    @classmethod
    def configurable_default(cls) -> Type[Any]:
        from tornado.simple_httpclient import SimpleAsyncHTTPClient  # type: ignore
        return SimpleAsyncHTTPClient

    @classmethod
    def _async_clients(cls) -> weakref.WeakKeyDictionary:
        attr_name: str = '_async_client_dict_' + cls.__name__
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, weakref.WeakKeyDictionary())
        return getattr(cls, attr_name)

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> "AsyncHTTPClient":
        io_loop: IOLoop = IOLoop.current()
        if force_instance:
            instance_cache = None
        else:
            instance_cache = cls._async_clients()
        if instance_cache is not None and io_loop in instance_cache:
            return instance_cache[io_loop]
        instance: "AsyncHTTPClient" = super().__new__(cls, **kwargs)  # type: ignore
        instance._instance_cache = instance_cache
        if instance_cache is not None:
            instance_cache[io_loop] = instance
        return instance

    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self.io_loop: IOLoop = IOLoop.current()
        self.defaults: Dict[str, Any] = dict(HTTPRequest._DEFAULTS)  # type: ignore
        if defaults is not None:
            self.defaults.update(defaults)
        self._closed: bool = False

    def close(self) -> None:
        """Destroys this HTTP client, freeing any file descriptors used."""
        if self._closed:
            return
        self._closed = True
        if self._instance_cache is not None:
            cached_val = self._instance_cache.pop(self.io_loop, None)
            if cached_val is not None and cached_val is not self:
                raise RuntimeError('inconsistent AsyncHTTPClient cache')

    def fetch(self, request: Union[str, "HTTPRequest"], raise_error: bool = True, **kwargs: Any) -> Future:
        """Executes a request, asynchronously returning an HTTPResponse."""
        if self._closed:
            raise RuntimeError('fetch() called on closed AsyncHTTPClient')
        if not isinstance(request, HTTPRequest):
            request = HTTPRequest(url=request, **kwargs)
        elif kwargs:
            raise ValueError("kwargs can't be used if request is an HTTPRequest object")
        request.headers = httputil.HTTPHeaders(request.headers)
        request_proxy: _RequestProxy = _RequestProxy(request, self.defaults)
        future: Future = Future()

        def handle_response(response: HTTPResponse) -> None:
            if response.error:
                if raise_error or not response._error_is_response_code:
                    future_set_exception_unless_cancelled(future, response.error)
                    return
            future_set_result_unless_cancelled(future, response)
        self.fetch_impl(request_proxy, handle_response)
        return future

    def fetch_impl(self, request: "HTTPRequest", callback: Callable[[HTTPResponse], None]) -> None:
        raise NotImplementedError()

    @classmethod
    def configure(cls, impl: Union[str, Type[Any], None], **kwargs: Any) -> None:
        """Configures the AsyncHTTPClient subclass to use."""
        super().configure(impl, **kwargs)

class HTTPRequest:
    """HTTP client request object."""
    _headers: Optional[httputil.HTTPHeaders] = None
    _DEFAULTS: Dict[str, Any] = dict(connect_timeout=20.0, request_timeout=20.0, follow_redirects=True, max_redirects=5, decompress_response=True, proxy_password='', allow_nonstandard_methods=False, validate_cert=True)

    def __init__(self, url: str, method: str = 'GET', headers: Optional[Union[httputil.HTTPHeaders, Dict[str, str]]] = None,
                 body: Optional[Union[str, bytes]] = None, auth_username: Optional[str] = None, auth_password: Optional[str] = None,
                 auth_mode: Optional[str] = None, connect_timeout: Optional[float] = None, request_timeout: Optional[float] = None,
                 if_modified_since: Optional[Union[datetime.datetime, float]] = None, follow_redirects: Optional[bool] = None,
                 max_redirects: Optional[int] = None, user_agent: Optional[str] = None, use_gzip: Optional[bool] = None,
                 network_interface: Optional[str] = None, streaming_callback: Optional[Callable[[bytes], None]] = None,
                 header_callback: Optional[Callable[[str], None]] = None, prepare_curl_callback: Optional[Callable[[Any], None]] = None,
                 proxy_host: Optional[str] = None, proxy_port: Optional[int] = None, proxy_username: Optional[str] = None,
                 proxy_password: Optional[str] = None, proxy_auth_mode: Optional[str] = None,
                 allow_nonstandard_methods: Optional[bool] = None, validate_cert: Optional[bool] = None, ca_certs: Optional[str] = None,
                 allow_ipv6: Optional[bool] = None, client_key: Optional[str] = None, client_cert: Optional[str] = None,
                 body_producer: Optional[Callable[..., Any]] = None, expect_100_continue: bool = False, decompress_response: Optional[bool] = None,
                 ssl_options: Optional[ssl.SSLContext] = None) -> None:
        self.headers: Union[httputil.HTTPHeaders, Dict[str, str]] = headers  # set via property setter below
        if if_modified_since:
            self.headers['If-Modified-Since'] = httputil.format_timestamp(if_modified_since)
        self.proxy_host: Optional[str] = proxy_host
        self.proxy_port: Optional[int] = proxy_port
        self.proxy_username: Optional[str] = proxy_username
        self.proxy_password: Optional[str] = proxy_password
        self.proxy_auth_mode: Optional[str] = proxy_auth_mode
        self.url: str = url
        self.method: str = method
        self.body = body
        self.body_producer: Optional[Callable[..., Any]] = body_producer
        self.auth_username: Optional[str] = auth_username
        self.auth_password: Optional[str] = auth_password
        self.auth_mode: Optional[str] = auth_mode
        self.connect_timeout: Optional[float] = connect_timeout
        self.request_timeout: Optional[float] = request_timeout
        self.follow_redirects: Optional[bool] = follow_redirects
        self.max_redirects: Optional[int] = max_redirects
        self.user_agent: Optional[str] = user_agent
        if decompress_response is not None:
            self.decompress_response: Optional[bool] = decompress_response
        else:
            self.decompress_response = use_gzip
        self.network_interface: Optional[str] = network_interface
        self.streaming_callback: Optional[Callable[[bytes], None]] = streaming_callback
        self.header_callback: Optional[Callable[[str], None]] = header_callback
        self.prepare_curl_callback: Optional[Callable[[Any], None]] = prepare_curl_callback
        self.allow_nonstandard_methods: Optional[bool] = allow_nonstandard_methods
        self.validate_cert: Optional[bool] = validate_cert
        self.ca_certs: Optional[str] = ca_certs
        self.allow_ipv6: Optional[bool] = allow_ipv6
        self.client_key: Optional[str] = client_key
        self.client_cert: Optional[str] = client_cert
        self.ssl_options: Optional[ssl.SSLContext] = ssl_options
        self.expect_100_continue: bool = expect_100_continue
        self.start_time: float = time.time()

    @property
    def headers(self) -> httputil.HTTPHeaders:
        return self._headers  # type: ignore

    @headers.setter
    def headers(self, value: Optional[Union[httputil.HTTPHeaders, Dict[str, str]]]) -> None:
        if value is None:
            self._headers = httputil.HTTPHeaders()
        elif isinstance(value, httputil.HTTPHeaders):
            self._headers = value
        else:
            self._headers = httputil.HTTPHeaders(value)

    @property
    def body(self) -> bytes:
        return self._body  # type: ignore

    @body.setter
    def body(self, value: Optional[Union[str, bytes]]) -> None:
        if value is None:
            self._body = b""
        else:
            self._body = utf8(value)

class HTTPResponse:
    """HTTP Response object."""

    error: Optional[Exception] = None
    request: Union[HTTPRequest, "_RequestProxy", None] = None

    def __init__(self, request: Union[HTTPRequest, "_RequestProxy"], code: int, headers: Optional[httputil.HTTPHeaders] = None,
                 buffer: Optional[BytesIO] = None, effective_url: Optional[str] = None, error: Optional[Exception] = None,
                 request_time: Optional[float] = None, time_info: Optional[Dict[str, Any]] = None, reason: Optional[str] = None,
                 start_time: Optional[float] = None) -> None:
        if isinstance(request, _RequestProxy):
            self.request = request.request
        else:
            self.request = request
        self.code: int = code
        self.reason: str = reason or httputil.responses.get(code, 'Unknown')
        if headers is not None:
            self.headers: httputil.HTTPHeaders = headers
        else:
            self.headers = httputil.HTTPHeaders()
        self.buffer: Optional[BytesIO] = buffer
        self._body: Optional[bytes] = None
        if effective_url is None:
            self.effective_url: str = request.url  # type: ignore
        else:
            self.effective_url = effective_url
        self._error_is_response_code: bool = False
        if error is None:
            if self.code < 200 or self.code >= 300:
                self._error_is_response_code = True
                self.error = HTTPError(self.code, message=self.reason, response=self)
            else:
                self.error = None
        else:
            self.error = error
        self.start_time: Optional[float] = start_time
        self.request_time: Optional[float] = request_time
        self.time_info: Dict[str, Any] = time_info or {}

    @property
    def body(self) -> bytes:
        if self.buffer is None:
            return b''
        elif self._body is None:
            self._body = self.buffer.getvalue()
        return self._body

    def rethrow(self) -> None:
        """If there was an error on the request, raise an HTTPError."""
        if self.error:
            raise self.error

    def __repr__(self) -> str:
        args = ','.join(('%s=%r' % i for i in sorted(self.__dict__.items())))
        return f'{self.__class__.__name__}({args})'

class HTTPClientError(Exception):
    """Exception thrown for an unsuccessful HTTP request.

    Attributes:
    * code - HTTP error integer error code, e.g. 404.
    * response - HTTPResponse object, if any.
    """

    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None:
        self.code: int = code
        self.message: str = message or httputil.responses.get(code, 'Unknown')
        self.response: Optional[HTTPResponse] = response
        super().__init__(code, message, response)

    def __str__(self) -> str:
        return 'HTTP %d: %s' % (self.code, self.message)
    __repr__ = __str__
HTTPError = HTTPClientError

class _RequestProxy:
    """Combines an object with a dictionary of defaults."""

    def __init__(self, request: HTTPRequest, defaults: Dict[str, Any]) -> None:
        self.request: HTTPRequest = request
        self.defaults: Dict[str, Any] = defaults

    def __getattr__(self, name: str) -> Any:
        request_attr: Any = getattr(self.request, name)
        if request_attr is not None:
            return request_attr
        elif self.defaults is not None:
            return self.defaults.get(name, None)
        else:
            return None

def main() -> None:
    from tornado.options import define, options, parse_command_line
    define('print_headers', type=bool, default=False)
    define('print_body', type=bool, default=True)
    define('follow_redirects', type=bool, default=True)
    define('validate_cert', type=bool, default=True)
    define('proxy_host', type=str)
    define('proxy_port', type=int)
    args = parse_command_line()
    client = HTTPClient()
    for arg in args:
        try:
            response: HTTPResponse = client.fetch(arg, follow_redirects=options.follow_redirects, validate_cert=options.validate_cert, proxy_host=options.proxy_host, proxy_port=options.proxy_port)
        except HTTPError as e:
            if e.response is not None:
                response = e.response
            else:
                raise
        if options.print_headers:
            print(response.headers)
        if options.print_body:
            print(native_str(response.body))
    client.close()

if __name__ == '__main__':
    main()