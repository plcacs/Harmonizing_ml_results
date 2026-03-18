from typing import Any, Union, Dict, Callable, Optional, Type
from io import BytesIO
from datetime import datetime
import ssl
from tornado.concurrent import Future
from tornado import httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable

class HTTPClient:
    def __init__(self, async_client_class: Optional[Type[Any]] = ..., **kwargs: Any) -> None: ...
    def __del__(self) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, "HTTPRequest"], **kwargs: Any) -> "HTTPResponse": ...

class AsyncHTTPClient(Configurable):
    _instance_cache: Any
    io_loop: IOLoop
    defaults: Dict[str, Any]
    _closed: bool

    @classmethod
    def configurable_base(cls) -> Type["AsyncHTTPClient"]: ...
    @classmethod
    def configurable_default(cls) -> Type["AsyncHTTPClient"]: ...
    @classmethod
    def _async_clients(cls) -> Any: ...
    def __new__(cls, force_instance: bool = ..., **kwargs: Any) -> "AsyncHTTPClient": ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = ...) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, "HTTPRequest"], raise_error: bool = ..., **kwargs: Any) -> Future: ...
    def fetch_impl(self, request: "HTTPRequest", callback: Callable[["HTTPResponse"], None]) -> None: ...
    @classmethod
    def configure(cls, impl: Any, **kwargs: Any) -> None: ...

class HTTPRequest:
    _headers: Any
    _DEFAULTS: Dict[str, Any]

    headers: httputil.HTTPHeaders
    proxy_host: Optional[str]
    proxy_port: Optional[int]
    proxy_username: Optional[str]
    proxy_password: Optional[str]
    proxy_auth_mode: Optional[str]
    url: str
    method: str
    body_producer: Optional[Callable[..., Any]]
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
    streaming_callback: Optional[Callable[..., Any]]
    header_callback: Optional[Callable[..., Any]]
    prepare_curl_callback: Optional[Callable[..., Any]]
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
        method: str = ...,
        headers: Optional[Union[httputil.HTTPHeaders, Dict[str, Any]]] = ...,
        body: Optional[Union[str, bytes]] = ...,
        auth_username: Optional[str] = ...,
        auth_password: Optional[str] = ...,
        auth_mode: Optional[str] = ...,
        connect_timeout: Optional[float] = ...,
        request_timeout: Optional[float] = ...,
        if_modified_since: Optional[Union[datetime, float]] = ...,
        follow_redirects: Optional[bool] = ...,
        max_redirects: Optional[int] = ...,
        user_agent: Optional[str] = ...,
        use_gzip: Optional[bool] = ...,
        network_interface: Optional[str] = ...,
        streaming_callback: Optional[Callable[..., Any]] = ...,
        header_callback: Optional[Callable[..., Any]] = ...,
        prepare_curl_callback: Optional[Callable[..., Any]] = ...,
        proxy_host: Optional[str] = ...,
        proxy_port: Optional[int] = ...,
        proxy_username: Optional[str] = ...,
        proxy_password: Optional[str] = ...,
        proxy_auth_mode: Optional[str] = ...,
        allow_nonstandard_methods: Optional[bool] = ...,
        validate_cert: Optional[bool] = ...,
        ca_certs: Optional[str] = ...,
        allow_ipv6: Optional[bool] = ...,
        client_key: Optional[str] = ...,
        client_cert: Optional[str] = ...,
        body_producer: Optional[Callable[..., Any]] = ...,
        expect_100_continue: bool = ...,
        decompress_response: Optional[bool] = ...,
        ssl_options: Optional[ssl.SSLContext] = ...,
    ) -> None: ...
    @property
    def headers(self) -> httputil.HTTPHeaders: ...
    @headers.setter
    def headers(self, value: Optional[Union[httputil.HTTPHeaders, Dict[str, Any]]]) -> None: ...
    @property
    def body(self) -> bytes: ...
    @body.setter
    def body(self, value: Union[str, bytes]) -> None: ...

class HTTPResponse:
    error: Optional[Exception]
    _error_is_response_code: bool
    request: Optional[HTTPRequest]
    code: int
    reason: str
    headers: httputil.HTTPHeaders
    buffer: Optional[BytesIO]
    effective_url: str
    start_time: Optional[float]
    request_time: Optional[float]
    time_info: Dict[str, Any]

    def __init__(
        self,
        request: Union["HTTPRequest", "_RequestProxy"],
        code: int,
        headers: Optional[httputil.HTTPHeaders] = ...,
        buffer: Optional[BytesIO] = ...,
        effective_url: Optional[str] = ...,
        error: Optional[Exception] = ...,
        request_time: Optional[float] = ...,
        time_info: Optional[Dict[str, Any]] = ...,
        reason: Optional[str] = ...,
        start_time: Optional[float] = ...,
    ) -> None: ...
    @property
    def body(self) -> bytes: ...
    def rethrow(self) -> None: ...
    def __repr__(self) -> str: ...

class HTTPClientError(Exception):
    code: int
    message: str
    response: Optional[HTTPResponse]
    def __init__(self, code: int, message: Optional[str] = ..., response: Optional[HTTPResponse] = ...) -> None: ...
    def __str__(self) -> str: ...
    __repr__: Callable[["HTTPClientError"], str]

HTTPError = HTTPClientError

class _RequestProxy:
    request: Any
    defaults: Optional[Dict[str, Any]]
    def __init__(self, request: Any, defaults: Optional[Dict[str, Any]]) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

def main() -> None: ...