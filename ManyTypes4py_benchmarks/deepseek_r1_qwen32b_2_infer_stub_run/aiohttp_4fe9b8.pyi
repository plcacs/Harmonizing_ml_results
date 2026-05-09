"""Aiohttp test utils."""

import asyncio
from collections.abc import Iterator
from http import HTTPStatus
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from yarl import URL
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE
from homeassistant.core import HomeAssistant

RETYPE = type(re.compile(''))

def mock_stream(data: Any) -> StreamReader:
    ...

class AiohttpClientMocker:
    def __init__(self) -> None:
        ...
    
    def request(
        self,
        method: str,
        url: Union[RETYPE, str, URL],
        *,
        auth: Any = ...,
        status: HTTPStatus = ...,
        text: Optional[str] = ...,
        data: Any = ...,
        content: Optional[bytes] = ...,
        json: Any = ...,
        params: Optional[Dict[str, Any]] = ...,
        headers: Optional[Dict[str, str]] = ...,
        exc: Optional[Type[Exception]] = ...,
        cookies: Optional[Dict[str, Any]] = ...,
        side_effect: Optional[Callable[..., Any]] = ...,
        closing: Optional[bool] = ...
    ) -> None:
        ...
    
    def get(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    def put(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    def post(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    def delete(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    def options(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    def patch(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        ...
    
    @property
    def call_count(self) -> int:
        ...
    
    def clear_requests(self) -> None:
        ...
    
    def create_session(self, loop: asyncio.AbstractEventLoop) -> ClientSession:
        ...
    
    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Any = ...,
        auth: Any = ...,
        params: Optional[Dict[str, Any]] = ...,
        headers: Optional[Dict[str, str]] = ...,
        allow_redirects: Any = ...,
        timeout: Any = ...,
        json: Any = ...,
        cookies: Any = ...,
        **kwargs: Any
    ) -> AiohttpClientMockResponse:
        ...

class AiohttpClientMockResponse:
    def __init__(
        self,
        method: str,
        url: Union[RETYPE, str, URL],
        status: HTTPStatus = ...,
        response: Optional[bytes] = ...,
        json: Any = ...,
        text: Optional[str] = ...,
        cookies: Optional[Dict[str, Any]] = ...,
        exc: Optional[Type[Exception]] = ...,
        headers: Optional[Dict[str, str]] = ...,
        side_effect: Optional[Callable[..., Any]] = ...,
        closing: Optional[bool] = ...
    ) -> None:
        ...
    
    def match_request(
        self,
        method: str,
        url: URL,
        params: Optional[Dict[str, Any]] = ...
    ) -> bool:
        ...
    
    @property
    def headers(self) -> CIMultiDict:
        ...
    
    @property
    def cookies(self) -> Dict[str, Any]:
        ...
    
    @property
    def url(self) -> URL:
        ...
    
    @property
    def content_type(self) -> str:
        ...
    
    @property
    def content(self) -> StreamReader:
        ...
    
    async def read(self) -> bytes:
        ...
    
    async def text(
        self,
        encoding: str = ...,
        errors: str = ...
    ) -> str:
        ...
    
    async def json(
        self,
        encoding: str = ...,
        content_type: Optional[str] = ...,
        loads: Callable[[str], Any] = ...
    ) -> Any:
        ...
    
    def release(self) -> None:
        ...
    
    def raise_for_status(self) -> None:
        ...
    
    def close(self) -> None:
        ...
    
    async def wait_for_close(self) -> None:
        ...
    
    @property
    def response(self) -> bytes:
        ...
    
    async def __aenter__(self) -> AiohttpClientMockResponse:
        ...
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Optional[TracebackType]
    ) -> None:
        ...

@contextmanager
def mock_aiohttp_client() -> Iterator[AiohttpClientMocker]:
    ...

class MockLongPollSideEffect:
    def __init__(self) -> None:
        ...
    
    async def __call__(
        self,
        method: str,
        url: Union[str, URL],
        data: Any
    ) -> AiohttpClientMockResponse:
        ...
    
    def queue_response(self, **kwargs: Any) -> None:
        ...
    
    def stop(self) -> None:
        ...