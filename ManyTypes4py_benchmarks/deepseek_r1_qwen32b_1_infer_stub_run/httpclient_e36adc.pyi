"""Blocking and non-blocking HTTP client interfaces."""

import datetime
import ssl
from io import BytesIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Future,
    List,
    Optional,
    Type,
    Union,
    WeakKeyDictionary,
    overload,
)
from tornado.concurrent import Future
from tornado.escape import utf8
from tornado.httputil import HTTPHeaders
from tornado.ioloop import IOLoop
from tornado.util import Configurable


class HTTPClient:
    """A blocking HTTP client."""

    def __init__(self, async_client_class: Optional[Type[AsyncHTTPClient]] = None, **kwargs: Any) -> None:
        ...

    def __del__(self) -> None:
        ...

    def close(self) -> None:
        ...

    def fetch(self, request: Union[str, HTTPRequest], **kwargs: Any) -> HTTPResponse:
        ...


class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client."""

    _instance_cache: Optional[WeakKeyDictionary]
    _closed: bool

    @classmethod
    def configurable_base(cls) -> Type[AsyncHTTPClient]:
        ...

    @classmethod
    def configurable_default(cls) -> Type[AsyncHTTPClient]:
        ...

    @classmethod
    def _async_clients(cls) -> WeakKeyDictionary:
        ...

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> AsyncHTTPClient:
        ...

    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        ...

    def close(self) -> None:
        ...

    def fetch(self, request: Union[str, HTTPRequest], raise_error: bool = True, **kwargs: Any) -> Future[HTTPResponse]:
        ...

    def fetch_impl(self, request: HTTPRequest, callback: Callable[[HTTPResponse], Any]) -> None:
        ...

    @classmethod
    def configure(cls, impl: Optional[Union[Type[AsyncHTTPClient], str]], **kwargs: Any) -> None:
        ...


class HTTPRequest:
    """HTTP client request object."""

    _headers: HTTPHeaders
    _body: Optional[bytes]
    _DEFAULTS: ClassVar[Dict[str, Any]]

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
        body_producer: Optional[Callable[[Callable[[bytes], Future[None]]], Future[None]]] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[ssl.SSLContext] = None,
    ) -> None:
        ...

    @property
    def headers(self) -> HTTPHeaders:
        ...

    @headers.setter
    def headers(self, value: Optional[Union[HTTPHeaders, Dict[str, str]]]) -> None:
        ...

    @property
    def body(self) -> Optional[bytes]:
        ...

    @body.setter
    def body(self, value: Optional[Union[str, bytes]]) -> None:
        ...


class HTTPResponse:
    """HTTP Response object."""

    error: Optional[HTTPClientError]
    _error_is_response_code: bool
    request: Optional[HTTPRequest]
    code: int
    reason: str
    headers: HTTPHeaders
    effective_url: str
    buffer: Optional[BytesIO]
    _body: Optional[bytes]
    request_time: Optional[float]
    start_time: Optional[float]
    time_info: Dict[str, Any]

    def __init__(
        self,
        request: Union[HTTPRequest, _RequestProxy],
        code: int,
        headers: Optional[HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional[HTTPClientError] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None:
        ...

    @property
    def body(self) -> bytes:
        ...

    def rethrow(self) -> None:
        ...

    def __repr__(self) -> str:
        ...


class HTTPClientError(Exception):
    """Exception thrown for an unsuccessful HTTP request."""

    code: int
    message: str
    response: Optional[HTTPResponse]

    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None:
        ...

    def __str__(self) -> str:
        ...

    __repr__ = __str__


class _RequestProxy:
    """Combines an object with a dictionary of defaults."""

    def __init__(self, request: HTTPRequest, defaults: Dict[str, Any]) -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        ...


def main() -> None:
    ...