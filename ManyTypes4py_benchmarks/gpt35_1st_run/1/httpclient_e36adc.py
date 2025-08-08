from typing import Type, Any, Union, Dict, Callable, Optional

class HTTPClient:
    def __init__(self, async_client_class: Type[AsyncHTTPClient] = None, **kwargs: Any) -> None:
        ...

    def close(self) -> None:
        ...

    def fetch(self, request: Union[str, HTTPRequest], **kwargs: Any) -> HTTPResponse:
        ...

class AsyncHTTPClient(Configurable):
    _instance_cache = None

    @classmethod
    def configurable_base(cls) -> Type[AsyncHTTPClient]:
        ...

    @classmethod
    def configurable_default(cls) -> Type[SimpleAsyncHTTPClient]:
        ...

    @classmethod
    def _async_clients(cls) -> Dict[str, Any]:
        ...

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> AsyncHTTPClient:
        ...

    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        ...

    def close(self) -> None:
        ...

    def fetch(self, request: Union[str, HTTPRequest], raise_error: bool = True, **kwargs: Any) -> Future[HTTPResponse]:
        ...

    def fetch_impl(self, request: HTTPRequest, callback: Callable[[HTTPResponse], None]) -> None:
        ...

    @classmethod
    def configure(cls, impl: Union[Type[AsyncHTTPClient], str] = None, **kwargs: Any) -> None:
        ...

class HTTPRequest:
    def __init__(self, url: str, method: str = 'GET', headers: Optional[Union[HTTPHeaders, Dict[str, str]]] = None, body: Optional[Union[str, bytes]] = None, auth_username: Optional[str] = None, auth_password: Optional[str] = None, auth_mode: Optional[str] = None, connect_timeout: Optional[float] = None, request_timeout: Optional[float] = None, if_modified_since: Optional[Union[datetime.datetime, float]] = None, follow_redirects: Optional[bool] = None, max_redirects: Optional[int] = None, user_agent: Optional[str] = None, use_gzip: Optional[bool] = None, network_interface: Optional[str] = None, streaming_callback: Optional[Callable] = None, header_callback: Optional[Callable] = None, prepare_curl_callback: Optional[Callable] = None, proxy_host: Optional[str] = None, proxy_port: Optional[int] = None, proxy_username: Optional[str] = None, proxy_password: Optional[str] = None, proxy_auth_mode: Optional[str] = None, allow_nonstandard_methods: Optional[bool] = None, validate_cert: Optional[bool] = None, ca_certs: Optional[str] = None, allow_ipv6: Optional[bool] = None, client_key: Optional[str] = None, client_cert: Optional[str] = None, body_producer: Optional[Callable] = None, expect_100_continue: Optional[bool] = False, decompress_response: Optional[bool] = None, ssl_options: Optional[ssl.SSLContext] = None) -> None:
        ...

class HTTPResponse:
    def __init__(self, request: Union[HTTPRequest, _RequestProxy], code: int, headers: Optional[HTTPHeaders] = None, buffer: Optional[BytesIO] = None, effective_url: Optional[str] = None, error: Optional[Exception] = None, request_time: Optional[float] = None, time_info: Optional[Dict[str, Any]] = None, reason: Optional[str] = None, start_time: Optional[float] = None) -> None:
        ...

class HTTPClientError(Exception):
    def __init__(self, code: int, message: Optional[str] = None, response: Optional[HTTPResponse] = None) -> None:
        ...

class _RequestProxy:
    def __init__(self, request: HTTPRequest, defaults: Dict[str, Any]) -> None:
        ...

def main() -> None:
    ...
