```python
import datetime
import ssl
from typing import Any, Callable, Dict, Optional, Union
from typing import Type, TypeVar

from tornado.concurrent import Future
from tornado.httputil import HTTPHeaders
from tornado.ioloop import IOLoop

_T = TypeVar("_T")

class HTTPClient:
    _closed: bool
    _async_client: "AsyncHTTPClient"
    _io_loop: IOLoop

    def __init__(self, async_client_class: Optional[Type["AsyncHTTPClient"]] = None, **kwargs: Any) -> None: ...
    def __del__(self) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, "HTTPRequest"], **kwargs: Any) -> "HTTPResponse": ...

class AsyncHTTPClient:
    io_loop: IOLoop
    defaults: Dict[str, Any]
    _closed: bool
    _instance_cache: Any

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> "AsyncHTTPClient": ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, "HTTPRequest"], raise_error: bool = True, **kwargs: Any) -> Future["HTTPResponse"]: ...
    def fetch_impl(self, request: "HTTPRequest", callback: Callable[["HTTPResponse"], None]) -> None: ...
    @classmethod
    def configure(cls, impl: Optional[Union[str, Type["AsyncHTTPClient"]]], **kwargs: Any) -> None: ...
    @classmethod
    def configurable_base(cls) -> Type["AsyncHTTPClient"]: ...
    @classmethod
    def configurable_default(cls) -> Type["AsyncHTTPClient"]: ...
    @classmethod
    def _async_clients(cls) -> Any: ...

class HTTPRequest:
    _DEFAULTS: Dict[str, Any]
    _headers: HTTPHeaders
    _body: bytes
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

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Union[HTTPHeaders, Dict[str, str]]] = None,
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
        ssl_options: Optional[ssl.SSLContext] = None,
    ) -> None: ...
    @property
    def headers(self) -> HTTPHeaders: ...
    @headers.setter
    def headers(self, value: Optional[Union[HTTPHeaders, Dict[str, str]]]) -> None: ...
    @property
    def body(self) -> bytes: ...
    @body.setter
    def body(self, value: Optional[Union[str, bytes]]) -> None: ...

class HTTPResponse:
    error: Optional[Exception]
    _error_is_response_code: bool
    request: HTTPRequest
    code: int
    reason: str
    headers: HTTPHeaders
    buffer: Any
    _body: Optional[bytes]
    effective_url: str
    start_time: Optional[float]
    request_time: Optional[float]
    time_info: Dict[str, float]

    def __init__(
        self,
        request: Union["HTTPRequest", "_RequestProxy"],
        code: int,
        headers: Optional[HTTPHeaders] = None,
        buffer: Any = None,
        effective_url: Optional[str] = None,
        error: Optional[Exception] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None: ...
    @property
    def body(self) -> bytes: ...
    def rethrow(self) -> None: ...
    def __repr__(self) -> str: ...

class HTTPClientError(Exception):
    code: int
    message: str
    response: Optional[HTTPResponse]

    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

HTTPError: Type[HTTPClientError] = ...

class _RequestProxy:
    request: HTTPRequest
    defaults: Dict[str, Any]

    def __init__(self, request: HTTPRequest, defaults: Dict[str, Any]) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

def main() -> None: ...
```