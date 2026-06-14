"""Aiohttp test utils."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import Any
from unittest import mock
from urllib.parse import parse_qs

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL

from homeassistant.core import HomeAssistant

RETYPE: type

def mock_stream(data: bytes) -> StreamReader: ...

class AiohttpClientMocker:
    _mocks: list[AiohttpClientMockResponse]
    _cookies: dict[str, Any]
    mock_calls: list[tuple[str, URL, Any, Any]]

    def __init__(self) -> None: ...
    def request(
        self,
        method: str,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = ...,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: dict[str, str] | None = ...,
        headers: dict[str, str] | None = ...,
        exc: BaseException | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Any | None = ...,
        closing: bool | None = ...,
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
        url: str | URL,
        *,
        data: Any = ...,
        auth: Any = ...,
        params: dict[str, str] | None = ...,
        headers: dict[str, str] | None = ...,
        allow_redirects: bool | None = ...,
        timeout: Any | None = ...,
        json: Any | None = ...,
        cookies: dict[str, str] | None = ...,
        **kwargs: Any,
    ) -> AiohttpClientMockResponse: ...

class AiohttpClientMockResponse:
    charset: str
    method: str
    _url: URL | re.Pattern[str]
    status: int
    _response: bytes
    exc: BaseException | None
    side_effect: Any | None
    closing: bool | None
    _headers: CIMultiDict[str]
    _cookies: dict[str, Any]

    def __init__(
        self,
        method: str,
        url: URL | re.Pattern[str],
        status: int = ...,
        response: bytes | None = ...,
        json: Any | None = ...,
        text: str | None = ...,
        cookies: dict[str, str] | None = ...,
        exc: BaseException | None = ...,
        headers: dict[str, str] | None = ...,
        side_effect: Any | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def match_request(
        self,
        method: str,
        url: URL,
        params: dict[str, str] | None = ...,
    ) -> bool: ...
    @property
    def headers(self) -> CIMultiDict[str]: ...
    @property
    def cookies(self) -> dict[str, Any]: ...
    @property
    def url(self) -> URL | re.Pattern[str]: ...
    @property
    def content_type(self) -> str | None: ...
    @property
    def content(self) -> StreamReader: ...
    async def read(self) -> bytes: ...
    async def text(self, encoding: str = ..., errors: str = ...) -> str: ...
    async def json(
        self,
        encoding: str = ...,
        content_type: str | None = ...,
        loads: Any = ...,
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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]: ...

class MockLongPollSideEffect:
    semaphore: asyncio.Semaphore
    response_list: list[dict[str, Any]]
    stopping: bool

    def __init__(self) -> None: ...
    async def __call__(
        self, method: str, url: str | URL, data: Any
    ) -> AiohttpClientMockResponse: ...
    def queue_response(self, **kwargs: Any) -> None: ...
    def stop(self) -> None: ...