from __future__ import annotations

import asyncio
import re
from contextlib import contextmanager
from http import HTTPStatus
from types import TracebackType
from typing import Any
from collections.abc import Awaitable, Callable, Iterator, Mapping
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL

RETYPE: type[re.Pattern[str]]


def mock_stream(data: bytes) -> StreamReader: ...


class AiohttpClientMocker:
    def __init__(self) -> None: ...
    def request(
        self,
        method: str,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def get(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def put(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def post(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def delete(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def options(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def patch(
        self,
        url: str | URL | re.Pattern[str],
        *,
        auth: Any | None = ...,
        status: int = HTTPStatus.OK,
        text: str | None = ...,
        data: bytes | None = ...,
        content: bytes | None = ...,
        json: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        exc: Exception | None = ...,
        cookies: dict[str, str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    @property
    def call_count(self) -> int: ...
    def clear_requests(self) -> None: ...
    def create_session(self, loop: asyncio.AbstractEventLoop) -> ClientSession: ...
    async def match_request(
        self,
        method: str,
        url: str | URL,
        *,
        data: Any | None = ...,
        auth: Any | None = ...,
        params: Any = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        allow_redirects: bool | None = ...,
        timeout: Any | None = ...,
        json: Any | None = ...,
        cookies: Any | None = ...,
        **kwargs: Any,
    ) -> 'AiohttpClientMockResponse': ...
    mock_calls: list[tuple[str, URL, Any, Mapping[str, str] | CIMultiDict[str] | None]]


class AiohttpClientMockResponse:
    charset: str
    method: str
    status: int
    exc: Exception | None
    side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None
    closing: bool | None
    def __init__(
        self,
        method: str,
        url: URL | re.Pattern[str],
        status: int = HTTPStatus.OK,
        response: bytes | None = ...,
        json: Any | None = ...,
        text: str | None = ...,
        cookies: dict[str, str] | None = ...,
        exc: Exception | None = ...,
        headers: Mapping[str, str] | CIMultiDict[str] | None = ...,
        side_effect: Callable[[str, URL, Any], Awaitable['AiohttpClientMockResponse']] | None = ...,
        closing: bool | None = ...,
    ) -> None: ...
    def match_request(self, method: str, url: URL, params: Any | None = ...) -> bool: ...
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
    async def json(self, encoding: str = ..., content_type: str | None = ..., loads: Callable[[str], Any] = ...) -> Any: ...
    def release(self) -> None: ...
    def raise_for_status(self) -> None: ...
    def close(self) -> None: ...
    async def wait_for_close(self) -> None: ...
    @property
    def response(self) -> bytes: ...
    async def __aenter__(self) -> 'AiohttpClientMockResponse': ...
    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...


@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]: ...


class MockLongPollSideEffect:
    semaphore: asyncio.Semaphore
    response_list: list[dict[str, Any]]
    stopping: bool
    def __init__(self) -> None: ...
    async def __call__(self, method: str, url: URL, data: Any) -> AiohttpClientMockResponse: ...
    def queue_response(self, **kwargs: Any) -> None: ...
    def stop(self) -> None: ...