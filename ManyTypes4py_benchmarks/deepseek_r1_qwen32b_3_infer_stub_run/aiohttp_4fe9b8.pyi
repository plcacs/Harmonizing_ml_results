"""Aiohttp test utils."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
)
from unittest import mock
from urllib.parse import parse_qs
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE
from homeassistant.core import HomeAssistant

RETYPE = type(re.compile(''))

def mock_stream(data: bytes) -> StreamReader: ...

class AiohttpClientMocker:
    """Mock Aiohttp client requests."""

    def __init__(self) -> None: ...
    
    def request(
        self,
        method: str,
        url: Union[RETYPE, str],
        *,
        auth: Any = ...,
        status: HTTPStatus = ...,
        text: Optional[str] = ...,
        data: Optional[bytes] = ...,
        content: Optional[bytes] = ...,
        json: Optional[Any] = ...,
        params: Optional[Dict[str, str]] = ...,
        headers: Optional[Dict[str, str]] = ...,
        exc: Optional[Exception] = ...,
        cookies: Optional[Dict[str, str]] = ...,
        side_effect: Optional[Callable] = ...,
        closing: Optional[bool] = ...
    ) -> None: ...
    
    def get(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    def put(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    def post(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    def delete(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    def options(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    def patch(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    
    @property
    def call_count(self) -> int: ...
    
    def clear_requests(self) -> None: ...
    
    def create_session(
        self,
        loop: asyncio.AbstractEventLoop
    ) -> ClientSession: ...
    
    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Optional[Any] = ...,
        auth: Any = ...,
        params: Optional[Dict[str, str]] = ...,
        headers: Optional[Dict[str, str]] = ...,
        allow_redirects: Optional[bool] = ...,
        timeout: Any = ...,
        json: Optional[Any] = ...,
        cookies: Optional[Dict[str, str]] = ...
    ) -> AiohttpClientMockResponse: ...

class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""

    def __init__(
        self,
        method: str,
        url: Union[RETYPE, URL],
        status: HTTPStatus = ...,
        response: Optional[bytes] = ...,
        json: Optional[Any] = ...,
        text: Optional[str] = ...,
        cookies: Optional[Dict[str, str]] = ...,
        exc: Optional[Exception] = ...,
        headers: Optional[Dict[str, str]] = ...,
        side_effect: Optional[Callable] = ...,
        closing: Optional[bool] = ...
    ) -> None: ...
    
    def match_request(
        self,
        method: str,
        url: Union[str, URL],
        params: Optional[Dict[str, str]] = ...
    ) -> bool: ...
    
    @property
    def headers(self) -> CIMultiDict: ...
    
    @property
    def cookies(self) -> Dict[str, mock.MagicMock]: ...
    
    @property
    def url(self) -> URL: ...
    
    @property
    def content_type(self) -> str: ...
    
    @property
    def content(self) -> StreamReader: ...
    
    async def read(self) -> bytes: ...
    
    async def text(
        self,
        encoding: str = ...,
        errors: str = ...
    ) -> str: ...
    
    async def json(
        self,
        encoding: str = ...,
        content_type: Optional[str] = ...,
        loads: Callable = ...
    ) -> Any: ...
    
    def release(self) -> None: ...
    
    def raise_for_status(self) -> None: ...
    
    def close(self) -> None: ...
    
    async def wait_for_close(self) -> None: ...
    
    @property
    def response(self) -> bytes: ...
    
    async def __aenter__(self) -> AiohttpClientMockResponse: ...
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None: ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]: ...

class MockLongPollSideEffect:
    """Imitate a long_poll request."""

    def __init__(self) -> None: ...
    
    async def __call__(
        self,
        method: str,
        url: str,
        data: Any
    ) -> AiohttpClientMockResponse: ...
    
    def queue_response(
        self,
        **kwargs: Any
    ) -> None: ...
    
    def stop(self) -> None: ...