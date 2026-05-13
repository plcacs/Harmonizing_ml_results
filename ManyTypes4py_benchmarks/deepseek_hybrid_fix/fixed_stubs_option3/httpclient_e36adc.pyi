import datetime
from io import BytesIO
import ssl
import weakref
from typing import Type, Any, Union, Dict, Callable, Optional
from tornado.concurrent import Future
from tornado.escape import native_str
from tornado.httputil import HTTPHeaders
from tornado.ioloop import IOLoop
from tornado.util import Configurable


class HTTPClient:
    def __init__(self, async_client_class: Optional[Type['AsyncHTTPClient']] = None, **kwargs: Any) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], **kwargs: Any) -> 'HTTPResponse': ...


class AsyncHTTPClient(Configurable):
    _instance_cache = None

    @classmethod
    def configurable_base(cls) -> Type['AsyncHTTPClient']: ...
    @classmethod
    def configurable_default(cls) -> Type['AsyncHTTPClient']: ...
    @classmethod
    def _async_clients(cls) -> Dict[IOLoop, 'AsyncHTTPClient']: ...
    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> 'AsyncHTTPClient': ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], raise_error: bool = True, **kwargs: Any) -> Future['HTTPResponse']: ...
    def fetch_impl(self, request: 'HTTPRequest', callback: Callable[['HTTPResponse'], None]) -> None: ...
    @classmethod
    def configure(cls, impl: Optional[Union[str, Type['AsyncHTTPClient']]], **kwargs: Any) -> None: ...
    io_loop: IOLoop
    defaults: Dict[str, Any]
    _closed: bool


class HTTPRequest:
    _headers: Optional[Union[HTTPHeaders, Dict[str, str]]]
    _body: Optional[bytes]
    _DEFAULTS: Dict[str, Any] = ...

    def __init__(self, url: str, method: str = 'GET', headers: Optional[Union[HTTPHeaders, Dict[str, str]]] = None, body: Optional[Union[str, bytes]] = None, auth_username: Optional[str] = None, auth_password: Optional[str] = None, auth_mode: Optional[str] = None, connect_timeout: Optional[float] = None, request_timeout: Optional[float] = None, if_modified_since: Optional[Union[datetime.datetime, float]] = None, follow_redirects: Optional[bool] = None, max_redirects: Optional[int] = None, user_agent: Optional[str] = None, use_gzip: Optional[bool] = None, network_interface: Optional[str] = None, streaming_callback: Optional[Callable[[bytes], None]] = None, header_callback: Optional[Callable[[str], None]] = None, prepare_curl_callback: Optional[Callable[[Any], None]] = None, proxy_host: Optional[str] = None, proxy_port: Optional[int] = None, proxy_username: Optional[str] = None, proxy_password: Optional[str] = None, proxy_auth_mode: Optional[str] = None, allow_nonstandard_methods: Optional[bool] = None, validate_cert: Optional[bool] = None, ca_certs: Optional[str] = None, allow_ipv6: Optional[bool] = None, client_key: Optional[str] = None, client_cert: Optional[str] = None, body_producer: Optional[Callable[[Callable[[bytes], Future[None]]], Future[None]]] = None, expect_100_continue: bool = False, decompress_response: Optional[bool] = None, ssl_options: Optional[ssl.SSLContext] = None) -> None: ...
    @property
    def headers(self) -> Optional[Union[HTTPHeaders, Dict[str, str]]]: ...
    @headers.setter
    def headers(self, value: Optional[Union[HTTPHeaders, Dict[str, str]]]) -> None: ...
    @property
    def body(self) -> Optional[bytes]: ...
    @body.setter
    def body(self, value: Optional[Union[str, bytes]]) -> None: ...
    proxy_host: Optional[str]
    proxy_port: Optional[int]
    proxy_username: Optional[str]
    proxy_password: Optional[str]
    proxy_auth_mode: Optional[str]
    url: str
    method: str
    body_producer: Optional[Callable[[Callable[[bytes], Future[None]]], Future[None]]]
    auth_username: Optional[str]
    auth_password: Optional[str]
    auth_mode: Optional[str]
    connect_timeout: Optional[float]
    request_timeout: Optional[float]
    follow_redirects: Optional[bool]
    max_redirects: Optional[int]
    user_agent: Optional[str]
    decompress_response: Optional[bool]
    network_interface: Optional[str]
    streaming_callback: Optional[Callable[[bytes], None]]
    header_callback: Optional[Callable[[str], None]]
    prepare_curl_callback: Optional[Callable[[Any], None]]
    allow_nonstandard_methods: Optional[bool]
    validate_cert: Optional[bool]
    ca_certs: Optional[str]
    allow_ipv6: Optional[bool]
    client_key: Optional[str]
    client_cert: Optional[str]
    ssl_options: Optional[ssl.SSLContext]
    expect_100_continue: bool
    start_time: float


class HTTPResponse:
    error: Optional['HTTPClientError']
    _error_is_response_code: bool
    request: Optional['HTTPRequest']

    def __init__(self, request: Union['HTTPRequest', '_RequestProxy'], code: int, headers: Optional[HTTPHeaders] = None, buffer: Optional[BytesIO] = None, effective_url: Optional[str] = None, error: Optional['HTTPClientError'] = None, request_time: Optional[float] = None, time_info: Optional[Dict[str, float]] = None, reason: Optional[str] = None, start_time: Optional[float] = None) -> None: ...
    @property
    def body(self) -> bytes: ...
    def rethrow(self) -> None: ...
    def __repr__(self) -> str: ...
    code: int
    reason: str
    headers: HTTPHeaders
    buffer: Optional[BytesIO]
    _body: Optional[bytes]
    effective_url: str
    start_time: Optional[float]
    request_time: Optional[float]
    time_info: Dict[str, float]


class HTTPClientError(Exception):
    code: int
    message: str
    response: Optional[HTTPResponse]

    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None: ...
    def __str__(self) -> str: ...


HTTPError = HTTPClientError


class _RequestProxy:
    request: HTTPRequest
    defaults: Optional[Dict[str, Any]]

    def __init__(self, request: HTTPRequest, defaults: Optional[Dict[str, Any]]) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


def main() -> None: ...