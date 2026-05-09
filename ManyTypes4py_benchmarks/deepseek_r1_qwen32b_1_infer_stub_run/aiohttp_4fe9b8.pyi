"""Aiohttp test utils."""

from __future__ import annotations
import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import (
    Any,
    Dict,
    Generator,
    Iterator as TIterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
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
from homeassistant.helpers.json import json_dumps
from homeassistant.util.json import json_loads

RETYPE = Pattern[str]

def mock_stream(data: bytes) -> StreamReader: ...

class AiohttpClientMocker:
    _mocks: List[AiohttpClientMockResponse]
    _cookies: Dict[str, mock.MagicMock]
    mock_calls: List[Tuple[str, URL, Union[bytes, str, dict, None], Dict[str, str]]]

    def __init__(self) -> None: ...

    def request(
        self,
        method: str,
        url: Union[str, RETYPE, URL],
        *,
        auth: Any = ...,
        status: HTTPStatus = ...,
        text: Optional[str] = ...,
        data: Optional[bytes] = ...,
        content: Optional[bytes] = ...,
        json: Optional[dict] = ...,
        params: Optional[dict] = ...,
        headers: Optional[dict] = ...,
        exc: Optional[Exception] = ...,
        cookies: Optional[dict] = ...,
        side_effect: Optional[Callable[[str, URL, Union[bytes, str, dict, None]], AiohttpClientMockResponse]] = ...,
        closing: Optional[bool] = ...,
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
        url: URL,
        *,
        data: Optional[Union[bytes, str, dict]] = ...,
        auth: Any = ...,
        params: Optional[dict] = ...,
        headers: Optional[dict] = ...,
        allow_redirects: Optional[bool] = ...,
        timeout: Any = ...,
        json: Optional[dict] = ...,
        cookies: Optional[dict] = ...,
        **kwargs: Any,
    ) -> AiohttpClientMockResponse: ...

class AiohttpClientMockResponse:
    method: str
    _url: Union[RETYPE, URL]
    status: HTTPStatus
    _response: bytes
    exc: Optional[Exception]
    side_effect: Optional[Callable[[str, URL, Union[bytes, str, dict, None]], AiohttpClientMockResponse]]
    closing: Optional[bool]
    _headers: CIMultiDict
    _cookies: Dict[str, mock.MagicMock]

    def __init__(
        self,
        method: str,
        url: Union[RETYPE, URL],
        status: HTTPStatus = ...,
        response: Optional[bytes] = ...,
        json: Optional[dict] = ...,
        text: Optional[str] = ...,
        cookies: Optional[dict] = ...,
        exc: Optional[Exception] = ...,
        headers: Optional[dict] = ...,
        side_effect: Optional[Callable[[str, URL, Union[bytes, str, dict, None]], AiohttpClientMockResponse]] = ...,
        closing: Optional[bool] = ...,
    ) -> None: ...

    def match_request(self, method: str, url: URL, params: Optional[dict] = ...) -> bool: ...

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
    @property
    def response(self) -> bytes: ...

    async def read(self) -> bytes: ...
    async def text(self, encoding: str = ..., errors: str = ...) -> str: ...
    async def json(
        self,
        encoding: str = ...,
        content_type: Optional[str] = ...,
        loads: Callable[[str], dict] = ...,
    ) -> dict: ...

    def release(self) -> None: ...
    def raise_for_status(self) -> None: ...
    def close(self) -> None: ...
    async def wait_for_close(self) -> None: ...

    async def __aenter__(self) -> AiohttpClientMockResponse: ...
    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None: ...

@contextmanager
def mock_aiohttp_client() -> Generator[AiohttpClientMocker, None, None]: ...

class MockLongPollSideEffect:
    def __init__(self) -> None: ...

    async def __call__(self, method: str, url: URL, data: Union[bytes, str, dict, None]) -> AiohttpClientMockResponse: ...

    def queue_response(self, **kwargs: Any) -> None: ...
    def stop(self) -> None: ...