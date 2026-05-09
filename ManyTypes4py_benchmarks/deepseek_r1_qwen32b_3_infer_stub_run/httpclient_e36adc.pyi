"""Blocking and non-blocking HTTP client interfaces."""

import datetime
import ssl
import time
from io import BytesIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from weakref import WeakKeyDictionary

from tornado.concurrent import Future
from tornado.escape import utf8, native_str
from tornado.ioloop import IOLoop
from tornado.util import Configurable

__all__ = [
    "HTTPClient",
    "AsyncHTTPClient",
    "HTTPRequest",
    "HTTPResponse",
    "HTTPError",
    "HTTPClientError",
]

class HTTPClient:
    """A blocking HTTP client."""
    def __init__(self, async_client_class: Optional[Type[AsyncHTTPClient]] = None, **kwargs: Any) -> None: ...
    def __del__(self) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Any, **kwargs: Any) -> HTTPResponse: ...

class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client."""
    _instance_cache: Optional[WeakKeyDictionary]
    _async_clients: ClassVar[Callable[[], WeakKeyDictionary]]
    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> Any: ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, HTTPRequest], raise_error: bool = True, **kwargs: Any) -> Future[HTTPResponse]: ...
    def fetch_impl(self, request: HTTPRequest, callback: Callable[[HTTPResponse], Any]) -> None: ...
    @classmethod
    def configure(cls, impl: Optional[Union[Type[AsyncHTTPClient], str]], **kwargs: Any) -> None: ...

class HTTPRequest:
    """HTTP client request object."""
    def __init__(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Union[httputil.HTTPHeaders, Dict[str, str]]] = None,
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
        streaming_callback: Optional[Callable[[bytes], Any]] = None,
        header_callback: Optional[Callable[[str], Any]] = None,
        prepare_curl_callback: Optional[Callable[[Any], Any]] = None,
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
        body_producer: Optional[Callable[[Callable[[bytes], Future[bool]]], Future[Any]]] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[ssl.SSLContext] = None,
    ) -> None: ...
    headers: httputil.HTTPHeaders
    body: bytes

class HTTPResponse:
    """HTTP Response object."""
    def __init__(
        self,
        request: HTTPRequest,
        code: int,
        headers: Optional[httputil.HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional[HTTPError] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None: ...
    request: HTTPRequest
    code: int
    reason: str
    headers: httputil.HTTPHeaders
    effective_url: str
    buffer: BytesIO
    body: bytes
    error: Optional[HTTPError]
    request_time: float
    start_time: Optional[float]
    time_info: Dict[str, Any]

class HTTPClientError(Exception):
    """Exception thrown for an unsuccessful HTTP request."""
    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None: ...
    code: int
    message: str
    response: Optional[HTTPResponse]

HTTPError = HTTPClientError

class _RequestProxy:
    """Combines an object with a dictionary of defaults."""
    def __init__(self, request: HTTPRequest, defaults: Dict[str, Any]) -> None: ...
    request: HTTPRequest
    defaults: Dict[str, Any]
    def __getattr__(self, name: str) -> Any: ...