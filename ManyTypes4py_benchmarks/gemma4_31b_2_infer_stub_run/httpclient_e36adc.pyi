import datetime
import ssl
from typing import Any, Callable, Dict, Optional, Union, Type, overload
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
from tornado.httputil import HTTPHeaders

class HTTPClient:
    def __init__(self, async_client_class: Optional[Type['AsyncHTTPClient']] = None, **kwargs: Any) -> None: ...
    def __del__(self) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], **kwargs: Any) -> 'HTTPResponse': ...

class AsyncHTTPClient:
    _instance_cache: Optional[Any]
    io_loop: IOLoop
    defaults: Dict[str, Any]
    _closed: bool

    @classmethod
    def configurable_base(cls) -> Type['AsyncHTTPClient']: ...
    @classmethod
    def configurable_default(cls) -> Type['AsyncHTTPClient']: ...
    @classmethod
    def _async_clients(cls) -> Any: ...
    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> 'AsyncHTTPClient': ...
    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None: ...
    def close(self) -> None: ...
    def fetch(self, request: Union[str, 'HTTPRequest'], raise_error: bool = True, **kwargs: Any) -> Future: ...
    def fetch_impl(self, request: 'HTTPRequest', callback: Callable[[ 'HTTPResponse'], Any]) -> None: ...
    @classmethod
    def configure(cls, impl: Optional[Union[Type['AsyncHTTPClient'], str]], **kwargs: Any) -> None: ...

class HTTPRequest:
    _headers: Optional[HTTPHeaders]
    _DEFAULTS: Dict[str, Any]

    def __init__(
        self,
        url: str,
        method: str = 'GET',
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
        body_producer: Optional[Callable[[Callable[[bytes], Future]], Future]] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[ssl.SSLContext] = None,
    ) -> None: ...

    @property
    def headers(self) -> HTTPHeaders: ...
    @headers.setter
    def headers(self, value: Optional[Union[HTTPHeaders, Dict[str, str]]]) -> None: ...
    @property
    def body(self) -> Optional[bytes]: ...
    @body.setter
    def body(self, value: Union[str, bytes]) -> None: ...

class HTTPResponse:
    error: Optional[Exception]
    _error_is_response_code: bool
    request: 'HTTPRequest'
    code: int
    reason: str
    headers: HTTPHeaders
    buffer: Optional[Any]
    effective_url: str
    start_time: Optional[float]
    request_time: Optional[float]
    time_info: Dict[str, Any]

    def __init__(
        self,
        request: Union['HTTPRequest', Any],
        code: int,
        headers: Optional[HTTPHeaders] = None,
        buffer: Optional[Any] = None,
        effective_url: Optional[str] = None,
        error: Optional[Exception] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, Any]] = None,
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

HTTPError: Type[HTTPClientError] = HTTPClientError

def main() -> None: ...