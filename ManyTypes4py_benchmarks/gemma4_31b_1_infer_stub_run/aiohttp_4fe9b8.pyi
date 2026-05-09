"""Aiohttp test utils."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from http import HTTPStatus
from typing import Any, Awaitable, Generator, Optional, Union, overload
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL
from homeassistant.core import HomeAssistant
import re

RETYPE: type[re.Pattern[str]] = type(re.compile(''))

def mock_stream(data: bytes) -> StreamReader: ...

class AiohttpClientMocker:
    """Mock Aiohttp client requests."""

    def __init__(self) -> None: ...

    def request(
        self,
        method: str,
        url: Union[str, URL, re.Pattern[str]],
        *,
        auth: Any = None,
        status: int = HTTPStatus.OK,
        text: Optional[str] = None,
        data: Any = None,
        content: Optional[bytes] = None,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        exc: Optional[Exception] = None,
        cookies: Optional[Mapping[str, Any]] = None,
        side_effect: Optional[Callable[[str, URL, Any], Awaitable[AiohttpClientMockResponse]]] = None,
        closing: Optional[bool] = None,
    ) -> None: ...

    def get(self, *args: Any, **kwargs: Any) -> None: ...

    def put(self, *args: Any, **kwargs: Any) -> None: ...

    def post(self, *args: Any, **kwargs: Any) -> None: ...

    def delete(self, *args: Any, **kwargs: Any) -> None: ...

    def options(self, *args: Any, **kwargs: Any) -> None: ...

    def patch(self, *args: Any, **kwargs: Any) -> None: ...

    @property
    def call_count(self) -> int: ...

    def clear_requests(self) -> None: ...

    def create_session(self, loop: asyncio.AbstractEventLoop) -> ClientSession: ...

    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Any = None,
        auth: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Any = None,
        json: Any = None,
        cookies: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> AiohttpClientMockResponse: ...

class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""

    def __init__(
        self,
        method: str,
        url: Union[str, URL, re.Pattern[str]],
        status: int = HTTPStatus.OK,
        response: Optional[bytes] = None,
        json: Any = None,
        text: Optional[str] = None,
        cookies: Optional[Mapping[str, Any]] = None,
        exc: Optional[Exception] = None,
        headers: Optional[Mapping[str, str]] = None,
        side_effect: Optional[Callable[[str, URL, Any], Awaitable[AiohttpClientMockResponse]]] = None,
        closing: Optional[bool] = None,
    ) -> None: ...

    def match_request(self, method: str, url: URL, params: Optional[Mapping[str, Any]] = None) -> bool: ...

    @property
    def headers(self) -> CIMultiDict[str, str]: ...

    @property
    def cookies(self) -> Mapping[str, Any]: ...

    @property
    def url(self) -> Union[URL, re.Pattern[str]]: ...

    @property
    def content_type(self) -> Optional[str]: ...

    @property
    def content(self) -> StreamReader: ...

    async def read(self) -> bytes: ...

    async def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str: ...

    async def json(self, encoding: str = 'utf-8', content_type: Optional[str] = None, loads: Callable[[str], Any] = ...) -> Any: ...

    def release(self) -> None: ...

    def raise_for_status(self) -> None: ...

    def close(self) -> None: ...

    async def wait_for_close(self) -> None: ...

    @property
    def response(self) -> bytes: ...

    async def __aenter__(self) -> 'AiohttpClientMockResponse': ...

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None: ...

def mock_aiohttp_client() -> Generator[AiohttpClientMocker, None, None]: ...

class MockLongPollSideEffect:
    """Imitate a long_poll request."""

    def __init__(self) -> None: ...

    async def __call__(self, method: str, url: URL, data: Any) -> AiohttpClientMockResponse: ...

    def queue_response(self, **kwargs: Any) -> None: ...

    def stop(self) -> None: ...