```python
"""Aiohttp test utils."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import Any, Dict, Optional, Pattern, Union
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

RETYPE: type = ...

def mock_stream(data: Any) -> StreamReader: ...

class AiohttpClientMocker:
    """Mock Aiohttp client requests."""
    
    def __init__(self) -> None: ...
    
    def request(
        self,
        method: str,
        url: Union[str, Pattern[str]],
        *,
        auth: Any = ...,
        status: HTTPStatus = ...,
        text: Optional[str] = ...,
        data: Any = ...,
        content: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        exc: Any = ...,
        cookies: Any = ...,
        side_effect: Any = ...,
        closing: Any = ...
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
    
    def create_session(self, loop: Any) -> ClientSession: ...
    
    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Any = ...,
        auth: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        allow_redirects: Any = ...,
        timeout: Any = ...,
        json: Any = ...,
        cookies: Any = ...,
        **kwargs: Any
    ) -> Any: ...

class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""
    
    def __init__(
        self,
        method: str,
        url: Union[str, Pattern[str], URL],
        status: HTTPStatus = ...,
        response: Any = ...,
        json: Any = ...,
        text: Optional[str] = ...,
        cookies: Any = ...,
        exc: Any = ...,
        headers: Any = ...,
        side_effect: Any = ...,
        closing: Any = ...
    ) -> None: ...
    
    def match_request(self, method: str, url: URL, params: Any = ...) -> bool: ...
    
    @property
    def headers(self) -> CIMultiDict: ...
    
    @property
    def cookies(self) -> Dict[str, Any]: ...
    
    @property
    def url(self) -> URL: ...
    
    @property
    def content_type(self) -> Optional[str]: ...
    
    @property
    def content(self) -> StreamReader: ...
    
    async def read(self) -> bytes: ...
    
    async def text(self, encoding: str = ..., errors: str = ...) -> str: ...
    
    async def json(
        self,
        encoding: str = ...,
        content_type: Any = ...,
        loads: Any = ...
    ) -> Any: ...
    
    def release(self) -> None: ...
    
    def raise_for_status(self) -> None: ...
    
    def close(self) -> None: ...
    
    async def wait_for_close(self) -> None: ...
    
    @property
    def response(self) -> bytes: ...
    
    async def __aenter__(self) -> "AiohttpClientMockResponse": ...
    
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None: ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]: ...

class MockLongPollSideEffect:
    """Imitate a long_poll request."""
    
    def __init__(self) -> None: ...
    
    async def __call__(self, method: str, url: URL, data: Any) -> AiohttpClientMockResponse: ...
    
    def queue_response(self, **kwargs: Any) -> None: ...
    
    def stop(self) -> None: ...
```