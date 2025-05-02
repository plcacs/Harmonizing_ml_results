```python
"""Blocking and non-blocking HTTP client interfaces.

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
from typing import Type, Any, Union, Dict, Callable, Optional, cast, ClassVar

from tornado.concurrent import (
    Future,
    future_set_result_unless_cancelled,
    future_set_exception_unless_cancelled,
)
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable


class HTTPClient:
    """A blocking HTTP client.

    This interface is provided to make it easier to share code between
    synchronous and asynchronous applications. Applications that are
    running an `.IOLoop` must use `AsyncHTTPClient` instead.

    Typical usage looks like this::

        http_client = httpclient.HTTPClient()
        try:
            response = http_client.fetch("http://www.google.com/")
            print(response.body)
        except httpclient.HTTPError as e:
            # HTTPError is raised for non-200 responses; the response
            # can be found in e.response.
            print("Error: " + str(e))
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Error: " + str(e))
        http_client.close()

    .. versionchanged:: 5.0

       Due to limitations in `asyncio`, it is no longer possible to
       use the synchronous ``HTTPClient`` while an `.IOLoop` is running.
       Use `AsyncHTTPClient` instead.

    """

    _closed: bool
    _io_loop: IOLoop
    _async_client: "AsyncHTTPClient"

    def __init__(
        self,
        async_client_class: Optional[Type["AsyncHTTPClient"]] = None,
        **kwargs: Any,
    ) -> None:
        self._closed = True
        self._io_loop = IOLoop(make_current=False)
        if async_client_class is None:
            async_client_class = AsyncHTTPClient

        async def make_client() -> "AsyncHTTPClient":
            await gen.sleep(0)
            assert async_client_class is not None
            return async_client_class(**kwargs)

        self._async_client = self._io_loop.run_sync(make_client)
        self._closed = False

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Closes the HTTPClient, freeing any resources used."""
        if not self._closed:
            self._async_client.close()
            self._io_loop.close()
            self._closed = True

    def fetch(
        self, request: Union["HTTPRequest", str], **kwargs: Any
    ) -> "HTTPResponse":
        """Executes a request, returning an `HTTPResponse`.

        The request may be either a string URL or an `HTTPRequest` object.
        If it is a string, we construct an `HTTPRequest` using any additional
        kwargs: ``HTTPRequest(request, **kwargs)``

        If an error occurs during the fetch, we raise an `HTTPError` unless
        the ``raise_error`` keyword argument is set to False.
        """
        response = self._io_loop.run_sync(
            functools.partial(self._async_client.fetch, request, **kwargs)
        )
        return response


class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client.

    Example usage::

        async def f():
            http_client = AsyncHTTPClient()
            try:
                response = await http_client.fetch("http://www.google.com")
            except Exception as e:
                print("Error: %s" % e)
            else:
                print(response.body)

    The constructor for this class is magic in several respects: It
    actually creates an instance of an implementation-specific
    subclass, and instances are reused as a kind of pseudo-singleton
    (one per `.IOLoop`). The keyword argument ``force_instance=True``
    can be used to suppress this singleton behavior. Unless
    ``force_instance=True`` is used, no arguments should be passed to
    the `AsyncHTTPClient` constructor. The implementation subclass as
    well as arguments to its constructor can be set with the static
    method `configure()`

    All `AsyncHTTPClient` implementations support a ``defaults``
    keyword argument, which can be used to set default values for
    `HTTPRequest` attributes.  For example::

        AsyncHTTPClient.configure(
            None, defaults=dict(user_agent="MyUserAgent"))
        # or with force_instance:
        client = AsyncHTTPClient(force_instance=True,
            defaults=dict(user_agent="MyUserAgent"))

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    """

    _instance_cache: ClassVar[Optional[Dict[IOLoop, "AsyncHTTPClient"]]] = None
    io_loop: IOLoop
    defaults: Dict[str, Any]
    _closed: bool

    @classmethod
    def configurable_base(cls) -> Type[Configurable]:
        return AsyncHTTPClient

    @classmethod
    def configurable_default(cls) -> Type[Configurable]:
        from tornado.simple_httpclient import SimpleAsyncHTTPClient

        return SimpleAsyncHTTPClient

    @classmethod
    def _async_clients(cls) -> Dict[IOLoop, "AsyncHTTPClient"]:
        attr_name = "_async_client_dict_" + cls.__name__
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, weakref.WeakKeyDictionary())
        return getattr(cls, attr_name)

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> "AsyncHTTPClient":
        io_loop = IOLoop.current()
        if force_instance:
            instance_cache = None
        else:
            instance_cache = cls._async_clients()
        if instance_cache is not None and io_loop in instance_cache:
            return instance_cache[io_loop]
        instance = super().__new__(cls, **kwargs)  # type: ignore
        instance._instance_cache = instance_cache
        if instance_cache is not None:
            instance_cache[instance.io_loop] = instance
        return instance

    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self.io_loop = IOLoop.current()
        self.defaults = dict(HTTPRequest._DEFAULTS)
        if defaults is not None:
            self.defaults.update(defaults)
        self._closed = False

    def close(self) -> None:
        """Destroys this HTTP client, freeing any file descriptors used.

        This method is **not needed in normal use** due to the way
        that `AsyncHTTPClient` objects are transparently reused.
        ``close()`` is generally only necessary when either the
        `.IOLoop` is also being closed, or the ``force_instance=True``
        argument was used when creating the `AsyncHTTPClient`.

        No other methods may be called on the `AsyncHTTPClient` after
        ``close()``.

        """
        if self._closed:
            return
        self._closed = True
        if self._instance_cache is not None:
            cached_val = self._instance_cache.pop(self.io_loop, None)
            if cached_val is not None and cached_val is not self:
                raise RuntimeError("inconsistent AsyncHTTPClient cache")

    def fetch(
        self,
        request: Union[str, "HTTPRequest"],
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "Future[HTTPResponse]":
        """Executes a request, asynchronously returning an `HTTPResponse`.

        The request may be either a string URL or an `HTTPRequest` object.
        If it is a string, we construct an `HTTPRequest` using any additional
        kwargs: ``HTTPRequest(request, **kwargs)``

        This method returns a `.Future` whose result is an
        `HTTPResponse`. By default, the ``Future`` will raise an
        `HTTPError` if the request returned a non-200 response code
        (other errors may also be raised if the server could not be
        contacted). Instead, if ``raise_error`` is set to False, the
        response will always be returned regardless of the response
        code.

        If a ``callback`` is given, it will be invoked with the `HTTPResponse`.
        In the callback interface, `HTTPError` is not automatically raised.
        Instead, you must check the response's ``error`` attribute or
        call its `~HTTPResponse.rethrow` method.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

           The ``raise_error=False`` argument only affects the
           `HTTPError` raised when a non-200 response code is used,
           instead of suppressing all errors.
        """
        if self._closed:
            raise RuntimeError("fetch() called on closed AsyncHTTPClient")
        if not isinstance(request, HTTPRequest):
            request = HTTPRequest(url=request, **kwargs)
        else:
            if kwargs:
                raise ValueError(
                    "kwargs can't be used if request is an HTTPRequest object"
                )
        request.headers = httputil.HTTPHeaders(request.headers)
        request_proxy = _RequestProxy(request, self.defaults)
        future: Future[HTTPResponse] = Future()

        def handle_response(response: "HTTPResponse") -> None:
            if response.error:
                if raise_error or not response._error_is_response_code:
                    future_set_exception_unless_cancelled(future, response.error)
                    return
            future_set_result_unless_cancelled(future, response)

        self.fetch_impl(cast(HTTPRequest, request_proxy), handle_response)
        return future

    def fetch_impl(
        self, request: "HTTPRequest", callback: Callable[["HTTPResponse"], None]
    ) -> None:
        raise NotImplementedError()

    @classmethod
    def configure(
        cls, impl: Union[None, str, Type[Configurable]], **kwargs: Any
    ) -> None:
        super().configure(impl, **kwargs)


class HTTPRequest:
    """HTTP client request object."""

    _DEFAULTS: ClassVar[Dict[str, Any]] = dict(
        connect_timeout=20.0,
        request_timeout=20.0,
        follow_redirects=True,
        max_redirects=5,
        decompress_response=True,
        proxy_password="",
        allow_nonstandard_methods=False,
        validate_cert=True,
    )

    _headers: Union[Dict[str, str], httputil.HTTPHeaders]
    _body: bytes

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Union[Dict[str, str], httputil.HTTPHeaders]] = None,
        body: Optional[Union[bytes, str]] = None,
        auth_username: Optional[str] = None,
        auth_password: Optional[str] = None,
        auth_mode: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
        if_modified_since: Optional[Union[float, datetime.datetime]] = None,
        follow_redirects: Optional[bool] = None,
        max_redirects: Optional[int] = None,
        user_agent: Optional[str] = None,
        use_gzip: Optional[bool] = None,
        network_interface: Optional[str] = None,
        streaming_callback: Optional[Callable[[bytes], None]] = None,
        header_callback: Optional[Callable[[str], None]] = None,
        prepare_curl_callback: Optional[Callable[[Any], None]] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
        proxy_auth_mode: Optional[str] = None,
        allow_nonstandard_methods: Optional[bool] = None,
        validate_cert: Optional[bool] = None,
        ca_certs: Optional[str] = None,
        allow_ipv6: Optional[bool] = None,
        client_key: Optional[str] = None,
        client_cert: Optional[str] = None,
        body_producer: Optional[Callable[[Callable[[bytes], None]], "Future[None]"]] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None,
    ) -> None:
        self.headers = headers  # type: ignore
        if if_modified_since:
            self.headers["If-Modified-Since"] = httputil.format_timestamp(
                if_modified_since
            )
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.proxy_auth_mode = proxy_auth_mode
        self.url = url
        self.method = method
        self.body = body  # type: ignore
        self.body_producer = body_producer
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.auth_mode = auth_mode
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        self.decompress_response = decompress_response if decompress_response is not None else use_gzip
        self.network_interface = network_interface
        self.streaming_callback = streaming_callback
        self.header_callback = header_callback
        self.prepare_curl_callback = prepare_curl_callback
        self.allow_nonstandard_methods = allow_nonstandard_methods
        self.validate_cert = validate_cert
        self.ca_certs = ca_certs
        self.allow_ipv6 = allow_ipv6
        self.client_key = client_key
        self.client_cert = client_cert
        self.ssl_options = ssl_options
        self.expect_100_continue = expect_100_continue
        self.start_time = time.time()

    @property
    def headers(self) -> httputil.HTTPHeaders:
        return self._headers  # type: ignore

    @headers.setter
    def headers(self, value: Union[Dict[str, str], httputil.HTTPHeaders]) -> None:
        if value is None:
            self._headers = httputil.HTTPHeaders()
        else:
            self._headers = value  # type: ignore

    @property
    def body(self) -> bytes:
        return self._body

    @body.setter
    def body(self, value: Union[bytes, str]) -> None:
        self._body = utf8(value)


class HTTPResponse:
    """HTTP Response object."""

    request: HTTPRequest
    code: int
    reason: str
    headers: httputil.HTTPHeaders
    buffer: Optional[BytesIO]
    effective_url: str
    error: Optional[BaseException]
    request_time: Optional[float]
    time_info: Dict[str, float]
    start_time: Optional[float]
    _body: Optional[bytes]
    _error_is_response_code: bool = False

    def __init__(
        self,
        request: HTTPRequest,
        code: int,
        headers: Optional[httputil.HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional[BaseException] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None:
        if isinstance(request, _RequestProxy):
            self.request = request.request
        else:
            self.request = request
        self.code = code
        self.reason = reason or httputil.responses.get(code, "Unknown")
        self.headers = headers if headers is not None else httputil.HTTPHeaders()
        self.buffer = buffer
        self._body = None
        self.effective_url = effective_url if effective_url is not None else request.url
        if error is None:
            if self.code < 200 or self.code >= 300:
                self._error_is_response_code = True
                self.error = HTTPError(self.code, message=self.reason, response=self)
           