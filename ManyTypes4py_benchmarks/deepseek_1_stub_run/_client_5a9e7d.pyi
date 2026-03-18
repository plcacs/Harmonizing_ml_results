```python
from __future__ import annotations

import datetime
import enum
import logging
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import Limits, Proxy, Timeout
from ._exceptions import InvalidURL, RemoteProtocolError, TooManyRedirects
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._types import (
    AsyncByteStream,
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    SyncByteStream,
    TimeoutTypes,
)
from ._urls import URL, QueryParams
from ._utils import URLPattern

T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")

__all__: Tuple[str, ...] = ("USE_CLIENT_DEFAULT", "AsyncClient", "Client")

DEFAULT_TIMEOUT_CONFIG: Timeout = ...
DEFAULT_MAX_REDIRECTS: int = ...
DEFAULT_LIMITS: Limits = ...


def _is_https_redirect(url: URL, location: URL) -> bool: ...
def _port_or_default(url: URL) -> int: ...
def _same_origin(url: URL, other: URL) -> bool: ...


class UseClientDefault:
    pass


USE_CLIENT_DEFAULT: UseClientDefault = ...
logger: logging.Logger = ...
USER_AGENT: str = ...
ACCEPT_ENCODING: str = ...


class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3


class BoundSyncStream(SyncByteStream):
    def __init__(self, stream: SyncByteStream, response: Response, start: float) -> None: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...


class BoundAsyncStream(AsyncByteStream):
    def __init__(self, stream: AsyncByteStream, response: Response, start: float) -> None: ...
    async def __aiter__(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...


EventHook = typing.Callable[..., Any]


class BaseClient:
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: Union[str, URL] = "",
        trust_env: bool = True,
        default_encoding: Union[str, typing.Callable[[bytes], str]] = "utf-8",
    ) -> None: ...

    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...
    @property
    def event_hooks(self) -> Dict[str, List[EventHook]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: Dict[str, List[EventHook]]) -> None: ...
    @property
    def auth(self) -> Optional[Auth]: ...
    @auth.setter
    def auth(self, auth: Optional[AuthTypes]) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: Union[str, URL]) -> None: ...
    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: Optional[HeaderTypes]) -> None: ...
    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: Optional[CookieTypes]) -> None: ...
    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: Optional[QueryParamTypes]) -> None: ...

    def build_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Request: ...

    def _merge_url(self, url: Union[str, URL]) -> URL: ...
    def _merge_cookies(self, cookies: Optional[CookieTypes] = None) -> Optional[Cookies]: ...
    def _merge_headers(self, headers: Optional[HeaderTypes] = None) -> Headers: ...
    def _merge_queryparams(self, params: Optional[QueryParamTypes] = None) -> Optional[QueryParams]: ...
    def _build_auth(self, auth: Optional[AuthTypes]) -> Optional[Auth]: ...
    def _build_request_auth(
        self, request: Request, auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT
    ) -> Auth: ...
    def _build_redirect_request(self, request: Request, response: Response) -> Request: ...
    def _redirect_method(self, request: Request, response: Response) -> str: ...
    def _redirect_url(self, request: Request, response: Response) -> URL: ...
    def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers: ...
    def _redirect_stream(self, request: Request, method: str) -> Optional[SyncByteStream]: ...
    def _set_timeout(self, request: Request) -> None: ...


class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: Union[bool, "ssl.SSLContext"] = True,
        cert: Optional[CertTypes] = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        proxy: Optional[ProxyTypes] = None,
        mounts: Optional[Mapping[Union[str, URLPattern], Optional[BaseTransport]]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: Union[str, URL] = "",
        transport: Optional[BaseTransport] = None,
        default_encoding: Union[str, typing.Callable[[bytes], str]] = "utf-8",
    ) -> None: ...

    def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    @contextmanager
    def stream(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Iterator[Response]: ...

    def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response: ...

    def get(
        self,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def options(
        self,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def head(
        self,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def post(
        self,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def put(
        self,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def patch(
        self,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def delete(
        self,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    def close(self) -> None: ...
    def __enter__(self) -> Client: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...


class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: Union[bool, "ssl.SSLContext"] = True,
        cert: Optional[CertTypes] = None,
        http1: bool = True,
        http2: bool = False,
        proxy: Optional[ProxyTypes] = None,
        mounts: Optional[Mapping[Union[str, URLPattern], Optional[AsyncBaseTransport]]] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Optional[Dict[str, List[EventHook]]] = None,
        base_url: Union[str, URL] = "",
        transport: Optional[AsyncBaseTransport] = None,
        trust_env: bool = True,
        default_encoding: Union[str, typing.Callable[[bytes], str]] = "utf-8",
    ) -> None: ...

    async def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[RequestContent] = None,
        data: Optional[RequestData] = None,
        files: Optional[RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> AsyncIterator[Response]: ...

    async def send(
        self,
        request: Request,
        *,
        stream: bool = False,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
    ) -> Response: ...

    async def get(
        self,
        url: Union[str, URL],
        *,
        params: Optional[QueryParamTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: Optional[RequestExtensions] = None,
    ) -> Response: ...

    async def options(
        self,
        url: Union[str, URL],
        *,
