"""Aiohttp test utils."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import Any, Optional, Union, Dict, List, Pattern, Callable, Awaitable
from unittest import mock
from urllib.parse import parse_qs
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.json import json_dumps
from homeassistant.util.json import json_loads

RETYPE: type[Pattern[str]] = type(re.compile(''))

def mock_stream(data: bytes) -> StreamReader:
    """Mock a stream with data."""
    ...

class AiohttpClientMocker:
    """Mock Aiohttp client requests."""

    def __init__(self) -> None:
        """Initialize the request mocker."""
        self._mocks: List[AiohttpClientMockResponse] = ...
        self._cookies: Dict[str, Any] = ...
        self.mock_calls: List[tuple[str, URL, Optional[Union[bytes, str, Dict[str, Any]]], Optional[Dict[str, str]]]] = ...

    def request(
        self,
        method: str,
        url: Union[str, Pattern[str], URL],
        *,
        auth: Optional[Any] = None,
        status: HTTPStatus = HTTPStatus.OK,
        text: Optional[str] = None,
        data: Optional[Union[bytes, str, Dict[str, Any]]] = None,
        content: Optional[bytes] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        exc: Optional[Exception] = None,
        cookies: Optional[Dict[str, str]] = None,
        side_effect: Optional[Callable[[str, URL, Optional[Union[bytes, str, Dict[str, Any]]]], Awaitable['AiohttpClientMockResponse']]] = None,
        closing: Optional[bool] = None
    ) -> None:
        """Mock a request."""
        ...

    def get(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock get request."""
        ...

    def put(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock put request."""
        ...

    def post(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock post request."""
        ...

    def delete(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock delete request."""
        ...

    def options(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock options request."""
        ...

    def patch(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock patch request."""
        ...

    @property
    def call_count(self) -> int:
        """Return the number of requests made."""
        ...

    def clear_requests(self) -> None:
        """Reset mock calls."""
        ...

    def create_session(self, loop: asyncio.AbstractEventLoop) -> ClientSession:
        """Create a ClientSession that is bound to this mocker."""
        ...

    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Optional[Union[bytes, str, Dict[str, Any]]] = None,
        auth: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Optional[Any] = None,
        json: Optional[Any] = None,
        cookies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> 'AiohttpClientMockResponse':
        """Match a request against pre-registered requests."""
        ...

class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""

    def __init__(
        self,
        method: str,
        url: Union[str, Pattern[str], URL],
        status: HTTPStatus = HTTPStatus.OK,
        response: Optional[bytes] = None,
        json: Optional[Any] = None,
        text: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        exc: Optional[Exception] = None,
        headers: Optional[Dict[str, str]] = None,
        side_effect: Optional[Callable[[str, URL, Optional[Union[bytes, str, Dict[str, Any]]]], Awaitable['AiohttpClientMockResponse']]] = None,
        closing: Optional[bool] = None
    ) -> None:
        """Initialize a fake response."""
        self.charset: str = ...
        self.method: str = ...
        self._url: Union[Pattern[str], URL] = ...
        self.status: HTTPStatus = ...
        self._response: bytes = ...
        self.exc: Optional[Exception] = ...
        self.side_effect: Optional[Callable[[str, URL, Optional[Union[bytes, str, Dict[str, Any]]]], Awaitable['AiohttpClientMockResponse']]] = ...
        self.closing: Optional[bool] = ...
        self._headers: CIMultiDict[str] = ...
        self._cookies: Dict[str, mock.MagicMock] = ...

    def match_request(self, method: str, url: URL, params: Optional[Dict[str, str]] = None) -> bool:
        """Test if response answers request."""
        ...

    @property
    def headers(self) -> CIMultiDict[str]:
        """Return content_type."""
        ...

    @property
    def cookies(self) -> Dict[str, mock.MagicMock]:
        """Return dict of cookies."""
        ...

    @property
    def url(self) -> URL:
        """Return yarl of URL."""
        ...

    @property
    def content_type(self) -> Optional[str]:
        """Return yarl of URL."""
        ...

    @property
    def content(self) -> StreamReader:
        """Return content."""
        ...

    async def read(self) -> bytes:
        """Return mock response."""
        ...

    async def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str:
        """Return mock response as a string."""
        ...

    async def json(
        self,
        encoding: str = 'utf-8',
        content_type: Optional[str] = None,
        loads: Callable[[str], Any] = json_loads
    ) -> Any:
        """Return mock response as a json."""
        ...

    def release(self) -> None:
        """Mock release."""
        ...

    def raise_for_status(self) -> None:
        """Raise error if status is 400 or higher."""
        ...

    def close(self) -> None:
        """Mock close."""
        ...

    async def wait_for_close(self) -> None:
        """Wait until all requests are done.

        Do nothing as we are mocking.
        """
        ...

    @property
    def response(self) -> bytes:
        """Property method to expose the response to other read methods."""
        ...

    async def __aenter__(self) -> 'AiohttpClientMockResponse':
        """Enter the context manager."""
        ...

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Exit the context manager."""
        ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]:
    """Context manager to mock aiohttp client."""
    ...

class MockLongPollSideEffect:
    """Imitate a long_poll request.

    It should be created and used as a side effect for a GET/PUT/etc. request.
    Once created, actual responses are queued with queue_response
    If queue is empty, will await until done.
    """

    def __init__(self) -> None:
        """Initialize the queue."""
        self.semaphore: asyncio.Semaphore = ...
        self.response_list: List[Dict[str, Any]] = ...
        self.stopping: bool = ...

    async def __call__(
        self,
        method: str,
        url: URL,
        data: Optional[Union[bytes, str, Dict[str, Any]]]
    ) -> AiohttpClientMockResponse:
        """Fetch the next response from the queue or wait until the queue has items."""
        ...

    def queue_response(self, **kwargs: Any) -> None:
        """Add a response to the long_poll queue."""
        ...

    def stop(self) -> None:
        """Stop the current request and future ones.

        This avoids an exception if there is someone waiting when exiting test.
        """
        ...