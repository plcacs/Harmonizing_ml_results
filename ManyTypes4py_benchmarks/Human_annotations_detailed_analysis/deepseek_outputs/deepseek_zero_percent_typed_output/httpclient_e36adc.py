"""Blocking and non-blocking HTTP client interfaces."""
import datetime
import functools
from io import BytesIO
import ssl
import time
import weakref
from tornado.concurrent import Future, future_set_result_unless_cancelled, future_set_exception_unless_cancelled
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable
from typing import Type, Any, Union, Dict, Callable, Optional, cast, List, Tuple

class HTTPClient:
    def __init__(self, async_client_class: Optional[Type['AsyncHTTPClient']] = None, **kwargs: Any) -> None:
        self._closed = True
        self._io_loop = IOLoop(make_current=False)
        if async_client_class is None:
            async_client_class = AsyncHTTPClient

        async def make_client() -> 'AsyncHTTPClient':
            await gen.sleep(0)
            assert async_client_class is not None
            return async_client_class(**kwargs)
        self._async_client = self._io_loop.run_sync(make_client)
        self._closed = False

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._async_client.close()
            self._io_loop.close()
            self._closed = True

    def fetch(self, request: Union[str, 'HTTPRequest'], **kwargs: Any) -> 'HTTPResponse':
        response = self._io_loop.run_sync(functools.partial(self._async_client.fetch, request, **kwargs))
        return response

class AsyncHTTPClient(Configurable):
    _instance_cache: Optional[weakref.WeakKeyDictionary[IOLoop, 'AsyncHTTPClient'] = None

    @classmethod
    def configurable_base(cls) -> Type['AsyncHTTPClient']:
        return AsyncHTTPClient

    @classmethod
    def configurable_default(cls) -> Type['AsyncHTTPClient']:
        from tornado.simple_httpclient import SimpleAsyncHTTPClient
        return SimpleAsyncHTTPClient

    @classmethod
    def _async_clients(cls) -> weakref.WeakKeyDictionary[IOLoop, 'AsyncHTTPClient']:
        attr_name = '_async_client_dict_' + cls.__name__
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, weakref.WeakKeyDictionary())
        return getattr(cls, attr_name)

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> 'AsyncHTTPClient':
        io_loop = IOLoop.current()
        if force_instance:
            instance_cache = None
        else:
            instance_cache = cls._async_clients()
        if instance_cache is not None and io_loop in instance_cache:
            return instance_cache[io_loop]
        instance = super().__new__(cls, **kwargs)
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
        if self._closed:
            return
        self._closed = True
        if self._instance_cache is not None:
            cached_val = self._instance_cache.pop(self.io_loop, None)
            if cached_val is not None and cached_val is not self:
                raise RuntimeError('inconsistent AsyncHTTPClient cache')

    def fetch(self, request: Union[str, 'HTTPRequest'], raise_error: bool = True, **kwargs: Any) -> Future['HTTPResponse']:
        if self._closed:
            raise RuntimeError('fetch() called on closed AsyncHTTPClient')
        if not isinstance(request, HTTPRequest):
            request = HTTPRequest(url=request, **kwargs)
        elif kwargs:
            raise ValueError("kwargs can't be used if request is an HTTPRequest object")
        request.headers = httputil.HTTPHeaders(request.headers)
        request_proxy = _RequestProxy(request, self.defaults)
        future = Future()

        def handle_response(response: 'HTTPResponse') -> None:
            if response.error:
                if raise_error or not response._error_is_response_code:
                    future_set_exception_unless_cancelled(future, response.error)
                    return
            future_set_result_unless_cancelled(future, response)
        self.fetch_impl(cast(HTTPRequest, request_proxy), handle_response)
        return future

    def fetch_impl(self, request: 'HTTPRequest', callback: Callable[['HTTPResponse'], None]) -> None:
        raise NotImplementedError()

    @classmethod
    def configure(cls, impl: Optional[Union[str, Type['AsyncHTTPClient']], **kwargs: Any) -> None:
        super().configure(impl, **kwargs)

class HTTPRequest:
    _headers: httputil.HTTPHeaders
    _DEFAULTS: Dict[str, Any] = dict(connect_timeout=20.0, request_timeout=20.0, follow_redirects=True, max_redirects=5, decompress_response=True, proxy_password='', allow_nonstandard_methods=False, validate_cert=True)

    def __init__(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Union[httputil.HTTPHeaders, Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        auth_username: Optional[str] = None,
        auth_password: Optional[str] = None,
        auth_mode: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
        if_modified_since: Optional[Union[datetime.datetime, float]] = None,
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
        body_producer: Optional[Callable[[Callable[[bytes], Future[None]]], Future[None]]] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None
    ) -> None:
        self.headers = headers
        if if_modified_since:
            self.headers['If-Modified-Since'] = httputil.format_timestamp(if_modified_since)
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.proxy_auth_mode = proxy_auth_mode
        self.url = url
        self.method = method
        self.body = body
        self.body_producer = body_producer
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.auth_mode = auth_mode
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        if decompress_response is not None:
            self.decompress_response = decompress_response
        else:
            self.decompress_response = use_gzip
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
        return self._headers

    @headers.setter
    def headers(self, value: Optional[Union[httputil.HTTPHeaders, Dict[str, str]]]) -> None:
        if value is None:
            self._headers = httputil.HTTPHeaders()
        else:
            self._headers = value

    @property
    def body(self) -> Optional[bytes]:
        return self._body

    @body.setter
    def body(self, value: Optional[Union[str, bytes]]) -> None:
        self._body = utf8(value) if value is not None else None

class HTTPResponse:
    error: Optional[Exception]
    _error_is_response_code: bool
    request: Optional['HTTPRequest']

    def __init__(
        self,
        request: Union['HTTPRequest', '_RequestProxy'],
        code: int,
        headers: Optional[httputil.HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional[Exception] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None
    ) -> None:
        if isinstance(request, _RequestProxy):
            self.request = request.request
        else:
            self.request = request
        self.code = code
        self.reason = reason or httputil.responses.get(code, 'Unknown')
        if headers is not None:
            self.headers = headers
        else:
            self.headers = httputil.HTTPHeaders()
        self.buffer = buffer
        self._body = None
        if effective_url is None:
            self.effective_url = request.url
        else:
            self.effective_url = effective_url
        self._error_is_response_code = False
        if error is None:
            if self.code < 200 or self.code >= 300:
                self._error_is_response_code = True
                self.error = HTTPError(self.code, message=self.reason, response=self)
            else:
                self.error = None
        else:
            self.error = error
        self.start_time = start_time
        self.request_time = request_time
        self.time_info = time_info or {}

    @property
    def body(self) -> bytes:
        if self.buffer is None:
            return b''
        elif self._body is None:
            self._body = self.buffer.getvalue()
        return self._body

    def rethrow(self) -> None:
        if self.error:
            raise self.error

    def __repr__(self) -> str:
        args = ','.join(('%s=%r' % i for i in sorted(self.__dict__.items())))
        return f'{self.__class__.__name__}({args})'

class HTTPClientError(Exception):
    def __init__(
        self,
        code: int,
        message: Optional[str] = None,
        response: Optional[HTTPResponse] = None
    ) -> None:
        self.code = code
        self.message = message or httputil.responses.get(code, 'Unknown')
        self.response = response
        super().__init__(code, message, response)

    def __str__(self) -> str:
        return 'HTTP %d: %s' % (self.code, self.message)
    __repr__ = __str__
HTTPError = HTTPClientError

class _RequestProxy:
    def __init__(self, request: HTTPRequest, defaults: Optional[Dict[str, Any]]) -> None:
        self.request = request
        self.defaults = defaults

    def __getattr__(self, name: str) -> Any:
        request_attr = getattr(self.request, name)
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
            response = client.fetch(arg, follow_redirects=options.follow_redirects, validate_cert=options.validate_cert, proxy_host=options.proxy_host, proxy_port=options.proxy_port)
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
