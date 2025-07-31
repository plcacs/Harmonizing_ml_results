#!/usr/bin/env python3
"""Aiohttp test utils."""
import asyncio
import re
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from types import TracebackType
from typing import Any, Callable, Dict, Iterator as TypingIterator, Optional, Pattern, Union, Awaitable

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectionError, ClientError, ClientResponseError
from aiohttp.streams import StreamReader
from multidict import CIMultiDict
from urllib.parse import parse_qs
from yarl import URL

from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.json import json_dumps
from homeassistant.util.json import json_loads

RETYPE: type = type(re.compile(''))


def mock_stream(data: bytes) -> StreamReader:
    """Mock a stream with data."""
    protocol = asyncio.Protocol()  # using a dummy protocol via mock in original implementation; using asyncio.Protocol as placeholder
    # In the original, protocol is a mock object; however, for type safety we accept it.
    # We can maintain the use of mock.Mock here.
    from unittest import mock
    protocol = mock.Mock(_reading_paused=False)
    stream = StreamReader(protocol, limit=2 ** 16)
    stream.feed_data(data)
    stream.feed_eof()
    return stream


class AiohttpClientMocker:
    """Mock Aiohttp client requests."""

    def __init__(self) -> None:
        """Initialize the request mocker."""
        self._mocks: list[AiohttpClientMockResponse] = []
        self._cookies: Dict[str, Any] = {}
        self.mock_calls: list[tuple[str, URL, Any, Any]] = []

    def request(
        self,
        method: str,
        url: Union[str, Pattern[str]],
        *,
        auth: Optional[Any] = None,
        status: int = HTTPStatus.OK,
        text: Optional[str] = None,
        data: Optional[Any] = None,
        content: Optional[bytes] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        exc: Optional[Exception] = None,
        cookies: Optional[Dict[str, str]] = None,
        side_effect: Optional[Callable[..., Awaitable[Any]]] = None,
        closing: Optional[bool] = None,
    ) -> None:
        """Mock a request."""
        if not isinstance(url, RETYPE):
            url = URL(url)  # type: ignore
        if params:
            url = url.with_query(params)
        self._mocks.append(
            AiohttpClientMockResponse(
                method=method,
                url=url,
                status=status,
                response=content,
                json=json,
                text=text,
                cookies=cookies,
                exc=exc,
                headers=headers,
                side_effect=side_effect,
                closing=closing,
            )
        )

    def get(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock get request."""
        self.request('get', *args, **kwargs)

    def put(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock put request."""
        self.request('put', *args, **kwargs)

    def post(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock post request."""
        self.request('post', *args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock delete request."""
        self.request('delete', *args, **kwargs)

    def options(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock options request."""
        self.request('options', *args, **kwargs)

    def patch(self, *args: Any, **kwargs: Any) -> None:
        """Register a mock patch request."""
        self.request('patch', *args, **kwargs)

    @property
    def call_count(self) -> int:
        """Return the number of requests made."""
        return len(self.mock_calls)

    def clear_requests(self) -> None:
        """Reset mock calls."""
        self._mocks.clear()
        self._cookies.clear()
        self.mock_calls.clear()

    def create_session(self, loop: asyncio.AbstractEventLoop) -> ClientSession:
        """Create a ClientSession that is bound to this mocker."""
        session = ClientSession(loop=loop, json_serialize=json_dumps)
        object.__setattr__(session, '_request', self.match_request)
        return session

    async def match_request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        data: Optional[Any] = None,
        auth: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Any] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Optional[Any] = None,
        json: Optional[Any] = None,
        cookies: Optional[Any] = None,
        **kwargs: Any,
    ) -> "AiohttpClientMockResponse":
        """Match a request against pre-registered requests."""
        data = data or json
        url = URL(url)  # ensure url is URL type
        if params:
            url = url.with_query(params)
        for response in self._mocks:
            if response.match_request(method, url, params):
                self.mock_calls.append((method, url, data, headers))
                if response.side_effect:
                    response_new = await response.side_effect(method, url, data)
                    response = response_new  # type: ignore
                if response.exc:
                    raise response.exc
                return response
        raise AssertionError(f'No mock registered for {method.upper()} {url} {params}')


class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""

    def __init__(
        self,
        method: str,
        url: URL,
        *,
        status: int = HTTPStatus.OK,
        response: Optional[bytes] = None,
        json: Optional[Any] = None,
        text: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        exc: Optional[Exception] = None,
        headers: Optional[Dict[str, str]] = None,
        side_effect: Optional[Callable[..., Awaitable[Any]]] = None,
        closing: Optional[bool] = None,
    ) -> None:
        """Initialize a fake response."""
        if json is not None:
            text = json_dumps(json)
        if text is not None:
            response = text.encode('utf-8')
        if response is None:
            response = b''
        self.charset: str = 'utf-8'
        self.method: str = method
        self._url: URL = url
        self.status: int = status
        self._response: bytes = response
        self.exc: Optional[Exception] = exc
        self.side_effect: Optional[Callable[..., Awaitable[Any]]] = side_effect
        self.closing: Optional[bool] = closing
        self._headers: CIMultiDict[str] = CIMultiDict(headers or {})
        self._cookies: Dict[str, Any] = {}
        if cookies:
            from unittest import mock  # local import for mocking cookies
            for name, data in cookies.items():
                cookie = mock.MagicMock()
                cookie.value = data
                self._cookies[name] = cookie

    def match_request(self, method: str, url: URL, params: Optional[Dict[str, Any]] = None) -> bool:
        """Test if response answers request."""
        if method.lower() != self.method.lower():
            return False
        if isinstance(self._url, RETYPE):
            return self._url.search(str(url)) is not None
        if self._url.scheme != url.scheme or self._url.host != url.host or self._url.path != url.path:
            return False
        request_qs = parse_qs(url.query_string)
        matcher_qs = parse_qs(self._url.query_string)
        for key, vals in matcher_qs.items():
            for val in vals:
                try:
                    request_qs.get(key, []).remove(val)
                except ValueError:
                    return False
        return True

    @property
    def headers(self) -> CIMultiDict[str]:
        """Return content_type."""
        return self._headers

    @property
    def cookies(self) -> Dict[str, Any]:
        """Return dict of cookies."""
        return self._cookies

    @property
    def url(self) -> URL:
        """Return yarl URL."""
        return self._url

    @property
    def content_type(self) -> Optional[str]:
        """Return the content type."""
        return self._headers.get('content-type')

    @property
    def content(self) -> StreamReader:
        """Return content as a StreamReader."""
        return mock_stream(self.response)

    async def read(self) -> bytes:
        """Return mock response."""
        return self.response

    async def text(self, encoding: str = 'utf-8', errors: str = 'strict') -> str:
        """Return mock response as a string."""
        return self.response.decode(encoding, errors=errors)

    async def json(self, encoding: str = 'utf-8', content_type: Optional[str] = None, loads: Callable[[str], Any] = json_loads) -> Any:
        """Return mock response as a json."""
        return loads(self.response.decode(encoding))

    def release(self) -> None:
        """Mock release."""
        pass

    def raise_for_status(self) -> None:
        """Raise error if status is 400 or higher."""
        if self.status >= 400:
            from unittest import mock
            request_info = mock.Mock(real_url='http://example.com')
            raise ClientResponseError(request_info=request_info, history=None, status=self.status, headers=self.headers)

    def close(self) -> None:
        """Mock close."""
        pass

    async def wait_for_close(self) -> None:
        """Wait until all requests are done.

        Do nothing as we are mocking.
        """
        pass

    @property
    def response(self) -> bytes:
        """Property method to expose the response to other read methods."""
        if self.closing:
            raise ClientConnectionError('Connection closed')
        return self._response

    async def __aenter__(self) -> "AiohttpClientMockResponse":
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        """Exit the context manager."""
        pass


@contextmanager
def mock_aiohttp_client() -> TypingIterator[AiohttpClientMocker]:
    """Context manager to mock aiohttp client."""
    mocker = AiohttpClientMocker()

    def create_session(hass: HomeAssistant, *args: Any, **kwargs: Any) -> ClientSession:
        session = mocker.create_session(hass.loop)

        async def close_session(event: Any) -> None:
            """Close session."""
            await session.close()
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, close_session)
        return session

    from unittest import mock
    with mock.patch('homeassistant.helpers.aiohttp_client._async_create_clientsession', side_effect=create_session):
        yield mocker


class MockLongPollSideEffect:
    """Imitate a long_poll request.

    It should be created and used as a side effect for a GET/PUT/etc. request.
    Once created, actual responses are queued with queue_response.
    If the queue is empty, it will await until done.
    """

    def __init__(self) -> None:
        """Initialize the queue."""
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(0)
        self.response_list: list[Dict[str, Any]] = []
        self.stopping: bool = False

    async def __call__(self, method: str, url: URL, data: Optional[Any]) -> AiohttpClientMockResponse:
        """Fetch the next response from the queue or wait until the queue has items."""
        if self.stopping:
            raise ClientError
        await self.semaphore.acquire()
        kwargs: Dict[str, Any] = self.response_list.pop(0)
        return AiohttpClientMockResponse(method=method, url=url, **kwargs)

    def queue_response(self, **kwargs: Any) -> None:
        """Add a response to the long_poll queue."""
        self.response_list.append(kwargs)
        self.semaphore.release()

    def stop(self) -> None:
        """Stop the current request and future ones.

        This avoids an exception if there is someone waiting when exiting test.
        """
        self.stopping = True
        self.queue_response(exc=ClientError())
