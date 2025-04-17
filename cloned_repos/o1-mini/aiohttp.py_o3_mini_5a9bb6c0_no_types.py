"""Aiohttp test utils."""
import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
import re
from types import TracebackType
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
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
RETYPE = type(re.compile(''))


def mock_stream(data):
    """Mock a stream with data."""
    protocol = mock.Mock(_reading_paused=False)
    stream = StreamReader(protocol, limit=2 ** 16)
    stream.feed_data(data)
    stream.feed_eof()
    return stream


class AiohttpClientMocker:
    """Mock Aiohttp client requests."""
    _mocks: List['AiohttpClientMockResponse']
    _cookies: Dict[str, mock.Mock]
    mock_calls: List[Tuple[str, URL, Optional[Union[bytes, Dict[str, Any]]],
        Optional[Dict[str, Any]]]]

    def __init__(self):
        """Initialize the request mocker."""
        self._mocks = []
        self._cookies = {}
        self.mock_calls = []

    def request(self, method, url, *, auth: Optional[Any]=None, status:
        HTTPStatus=HTTPStatus.OK, text: Optional[str]=None, data: Optional[
        Union[bytes, Dict[str, Any]]]=None, content: Optional[bytes]=None,
        json: Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any
        ]]=None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Mock a request."""
        if not isinstance(url, RETYPE):
            url = URL(url)
        if params:
            url = url.with_query(params)
        self._mocks.append(AiohttpClientMockResponse(method=method, url=url,
            status=status, response=content, json=json, text=text, cookies=
            cookies, exc=exc, headers=headers, side_effect=side_effect,
            closing=closing))

    def get(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock get request."""
        self.request('get', url, auth=auth, status=status, text=text, data=
            data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    def put(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock put request."""
        self.request('put', url, auth=auth, status=status, text=text, data=
            data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    def post(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock post request."""
        self.request('post', url, auth=auth, status=status, text=text, data
            =data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    def delete(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock delete request."""
        self.request('delete', url, auth=auth, status=status, text=text,
            data=data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    def options(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock options request."""
        self.request('options', url, auth=auth, status=status, text=text,
            data=data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    def patch(self, url, *, auth: Optional[Any]=None, status: HTTPStatus=
        HTTPStatus.OK, text: Optional[str]=None, data: Optional[Union[bytes,
        Dict[str, Any]]]=None, content: Optional[bytes]=None, json:
        Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=
        None, headers: Optional[Dict[str, str]]=None, exc: Optional[
        Exception]=None, cookies: Optional[Dict[str, Any]]=None,
        side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]=None,
        closing: Optional[bool]=None):
        """Register a mock patch request."""
        self.request('patch', url, auth=auth, status=status, text=text,
            data=data, content=content, json=json, params=params, headers=
            headers, exc=exc, cookies=cookies, side_effect=side_effect,
            closing=closing)

    @property
    def call_count(self):
        """Return the number of requests made."""
        return len(self.mock_calls)

    def clear_requests(self):
        """Reset mock calls."""
        self._mocks.clear()
        self._cookies.clear()
        self.mock_calls.clear()

    def create_session(self, loop):
        """Create a ClientSession that is bound to this mocker."""
        session = ClientSession(loop=loop, json_serialize=json_dumps)
        object.__setattr__(session, '_request', self.match_request)
        return session

    async def match_request(self, method: str, url: Union[str, URL], *,
        data: Optional[Union[bytes, Dict[str, Any]]]=None, auth: Optional[
        Any]=None, params: Optional[Dict[str, Any]]=None, headers: Optional
        [Dict[str, str]]=None, allow_redirects: Optional[bool]=None,
        timeout: Optional[Any]=None, json: Optional[Dict[str, Any]]=None,
        cookies: Optional[Dict[str, Any]]=None, **kwargs: Any
        ) ->'AiohttpClientMockResponse':
        """Match a request against pre-registered requests."""
        data = data or json
        if isinstance(url, str):
            url = URL(url)
        if params:
            url = url.with_query(params)
        for response in self._mocks:
            if response.match_request(method, url, params):
                self.mock_calls.append((method, url, data, headers))
                if response.side_effect:
                    response_effect = await response.side_effect(method,
                        url, data)
                    response = response_effect
                if response.exc:
                    raise response.exc
                return response
        raise AssertionError(
            f'No mock registered for {method.upper()} {url} {params}')


class AiohttpClientMockResponse:
    """Mock Aiohttp client response."""
    method: str
    _url: URL
    status: HTTPStatus
    _response: bytes
    exc: Optional[Exception]
    side_effect: Optional[Callable[..., Coroutine[Any, Any, Any]]]
    closing: Optional[bool]
    _headers: CIMultiDict
    _cookies: Dict[str, mock.Mock]
    charset: str

    def __init__(self, method, url, status=HTTPStatus.OK, response=None,
        json=None, text=None, cookies=None, exc=None, headers=None,
        side_effect=None, closing=None):
        """Initialize a fake response."""
        if json is not None:
            text = json_dumps(json)
        if text is not None:
            response = text.encode('utf-8')
        if response is None:
            response = b''
        self.charset = 'utf-8'
        self.method = method
        self._url = url
        self.status = status
        self._response = response
        self.exc = exc
        self.side_effect = side_effect
        self.closing = closing
        self._headers = CIMultiDict(headers or {})
        self._cookies = {}
        if cookies:
            for name, data in cookies.items():
                cookie = mock.MagicMock()
                cookie.value = data
                self._cookies[name] = cookie

    def match_request(self, method, url, params=None):
        """Test if response answers request."""
        if method.lower() != self.method.lower():
            return False
        if isinstance(self._url, RETYPE):
            return self._url.search(str(url)) is not None
        if (self._url.scheme != url.scheme or self._url.host != url.host or
            self._url.path != url.path):
            return False
        request_qs = parse_qs(url.query_string)
        matcher_qs = parse_qs(self._url.query_string)
        for key, vals in matcher_qs.items():
            for val in vals:
                try:
                    request_qs.get(key, []).remove(val)
                except (KeyError, ValueError):
                    return False
        return True

    @property
    def headers(self):
        """Return headers."""
        return self._headers

    @property
    def cookies(self):
        """Return dict of cookies."""
        return self._cookies

    @property
    def url(self):
        """Return yarl of URL."""
        return self._url

    @property
    def content_type(self):
        """Return content_type."""
        return self._headers.get('content-type')

    @property
    def content(self):
        """Return content."""
        return mock_stream(self.response)

    async def read(self) ->bytes:
        """Return mock response."""
        return self.response

    async def text(self, encoding: str='utf-8', errors: str='strict') ->str:
        """Return mock response as a string."""
        return self.response.decode(encoding, errors=errors)

    async def json(self, encoding: str='utf-8', content_type: Optional[str]
        =None, loads: Callable[[str], Any]=json_loads) ->Any:
        """Return mock response as a json."""
        return loads(self.response.decode(encoding))

    def release(self):
        """Mock release."""
        pass

    def raise_for_status(self):
        """Raise error if status is 400 or higher."""
        if self.status >= 400:
            request_info = mock.Mock(real_url=self._url)
            raise ClientResponseError(request_info=request_info, history=
                None, status=self.status, headers=self.headers)

    def close(self):
        """Mock close."""
        pass

    async def wait_for_close(self) ->None:
        """Wait until all requests are done.

        Do nothing as we are mocking.
        """
        pass

    @property
    def response(self):
        """Property method to expose the response to other read methods."""
        if self.closing:
            raise ClientConnectionError('Connection closed')
        return self._response

    async def __aenter__(self) ->'AiohttpClientMockResponse':
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
        ) ->None:
        """Exit the context manager."""
        pass


@contextmanager
def mock_aiohttp_client():
    """Context manager to mock aiohttp client."""
    mocker = AiohttpClientMocker()

    def create_session(hass, *args: Any, **kwargs: Any):
        session = mocker.create_session(hass.loop)

        async def close_session(event: Any) ->None:
            """Close session."""
            await session.close()
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, close_session)
        return session
    with mock.patch(
        'homeassistant.helpers.aiohttp_client._async_create_clientsession',
        side_effect=create_session):
        yield mocker


class MockLongPollSideEffect:
    """Imitate a long_poll request.

    It should be created and used as a side effect for a GET/PUT/etc. request.
    Once created, actual responses are queued with queue_response
    If queue is empty, will await until done.
    """
    semaphore: asyncio.Semaphore
    response_list: List[Dict[str, Any]]
    stopping: bool

    def __init__(self):
        """Initialize the queue."""
        self.semaphore = asyncio.Semaphore(0)
        self.response_list = []
        self.stopping = False

    async def __call__(self, method: str, url: URL, data: Optional[Union[
        bytes, Dict[str, Any]]]) ->AiohttpClientMockResponse:
        """Fetch the next response from the queue or wait until the queue has items."""
        if self.stopping:
            raise ClientError
        await self.semaphore.acquire()
        kwargs = self.response_list.pop(0)
        return AiohttpClientMockResponse(method=method, url=url, **kwargs)

    def queue_response(self, **kwargs: Any):
        """Add a response to the long_poll queue."""
        self.response_list.append(kwargs)
        self.semaphore.release()

    def stop(self):
        """Stop the current request and future ones.

        This avoids an exception if there is someone waiting when exiting test.
        """
        self.stopping = True
        self.queue_response(exc=ClientError())
