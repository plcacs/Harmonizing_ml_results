"""Stub file for 'httpclient_e36adc' module."""

from tornado.concurrent import Future
from tornado.httputil import HTTPHeaders
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

class HTTPClient:
    """Blocking HTTP client interface."""
    def __init__(self, async_client_class: Optional[Type[Any]] = None, **kwargs: Any) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], **kwargs: Any) -> 'HTTPResponse': ...

class AsyncHTTPClient:
    """Non-blocking HTTP client interface."""
    _instance_cache: ClassVar[Optional[weakref.WeakKeyDictionary]] = ...
    _async_clients: ClassVar[Callable[[], weakref.WeakKeyDictionary]] = ...

    @classmethod
    def configure(cls, impl: Union[Type[Any], str, None], **kwargs: Any) -> None: ...
    @classmethod
    def _async_clients(cls) -> weakref.WeakKeyDictionary: ...

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> 'AsyncHTTPClient': ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], raise_error: bool = True, **kwargs: Any) -> Future['HTTPResponse']: ...
    def fetch_impl(self, request: 'HTTPRequest', callback: Callable[['HTTPResponse'], None]) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type['AsyncHTTPClient']: ...
    @classmethod
    def configurable_default(cls) -> Type['SimpleAsyncHTTPClient']: ...

class HTTPRequest:
    """HTTP client request object."""
    def __init__(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Union[HTTPHeaders, Dict[str, str]]] = None,
        body: Optional[Union[bytes, str]] = None,
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
        ssl_options: Optional[ssl.SSLContext] = None,
    ) -> None: ...
    headers: HTTPHeaders
    body: bytes

class HTTPResponse:
    """HTTP response object."""
    def __init__(
        self,
        request: Union['HTTPRequest', '_RequestProxy'],
        code: int,
        headers: Optional[HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional['HTTPError'] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None: ...
    request: 'HTTPRequest'
    code: int
    reason: str
    headers: HTTPHeaders
    effective_url: str
    body: bytes
    error: Optional['HTTPError']
    request_time: Optional[float]
    start_time: Optional[float]
    time_info: Dict[str, float]

    def rethrow(self) -> None: ...
    def __repr__(self) -> str: ...

class HTTPClientError(Exception):
    """Exception for unsuccessful HTTP requests."""
    def __init__(self, code: int, message: Optional[str] = None, response: Optional['HTTPResponse'] = None) -> None: ...
    code: int
    message: str
    response: Optional['HTTPResponse']
    def __str__(self) -> str: ...
    __repr__ = __str__

HTTPError = HTTPClientError

class _RequestProxy:
    """Internal request proxy combining defaults."""
    def __init__(self, request: 'HTTPRequest', defaults: Dict[str, Any]) -> None: ...
    request: 'HTTPRequest'
    def __getattr__(self, name: str) -> Any: ...

def main() -> None: ...