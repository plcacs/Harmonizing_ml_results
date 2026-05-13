from collections.abc import Iterator, Awaitable
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import Any, Optional, Union, Dict, List, Pattern, Callable
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
import asyncio

RETYPE = type(re.compile(''))

def mock_stream(data: bytes) -> StreamReader: ...

class AiohttpClientMocker:
    _mocks: List['AiohttpClientMockResponse']
    _cookies: Dict[str, mock.MagicMock]
    mock_calls: List[Any]

    def __init__(self) -> None: ...
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
        data: Optional[Union[bytes, str, Dict[str, Any]]] = None,
        auth: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Optional[Any] = None,
        json: Optional[Any] = None,
        cookies: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> 'AiohttpClientMockResponse': ...

class AiohttpClientMockResponse:
    charset: str
    method: str
    _url: URL
    status: int
    _response: bytes
    exc: Optional[Exception]
    side_effect: Optional[Callable[[str, URL, Optional[Union[bytes, str, Dict[str, Any]]]], Awaitable['AiohttpClientMockResponse']]]
    closing: Optional[bool]
    _headers: CIMultiDict[str]
    _cookies: Dict[str, mock.MagicMock]

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
        closing: Optional[bool] = None,
    ) -> None: ...
    def match_request(self, method: str, url: URL, params: Optional[Dict[str, str]] = None) -> bool: ...
    @property
    def headers(self) -> CIMultiDict[str]: ...
    @property
    def cookies(self) -> Dict[str, mock.MagicMock]: ...
    @property
    def url(self) -> URL: ...
    @property
    def content_type(self) -> Optional[str]: ...
    @property
    def content(self) -> StreamReader: ...
    async def read(self) -> bytes: ...
    async def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str: ...
    async def json(self, encoding: str = 'utf-8', content_type: Optional[str] = None, loads: Callable[[str], Any] = json_loads) -> Any: ...
    def release(self) -> None: ...
    def raise_for_status(self) -> None: ...
    def close(self) -> None: ...
    async def wait_for_close(self) -> None: ...
    @property
    def response(self) -> bytes: ...
    async def __aenter__(self) -> 'AiohttpClientMockResponse': ...
    async def __aexit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None: ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]: ...

class MockLongPollSideEffect:
    semaphore: asyncio.Semaphore
    response_list: List[Dict[str, Any]]
    stopping: bool

    def __init__(self) -> None: ...
    async def __call__(self, method: str, url: URL, data: Optional[Union[bytes, str, Dict[str, Any]]]) -> AiohttpClientMockResponse: ...
    def queue_response(self, **kwargs: Any) -> None: ...
    def stop(self) -> None: ...