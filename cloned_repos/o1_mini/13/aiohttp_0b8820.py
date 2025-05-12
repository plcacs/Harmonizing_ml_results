"""Utilities to help with aiohttp."""
from __future__ import annotations
from http import HTTPStatus
import io
from typing import Any, Optional, Dict, Union
from urllib.parse import parse_qsl
from aiohttp import payload, web
from aiohttp.typedefs import JSONDecoder
from multidict import CIMultiDict, MultiDict
from .json import json_loads


class MockStreamReader:
    """Small mock to imitate stream reader."""

    def __init__(self, content: bytes) -> None:
        """Initialize mock stream reader."""
        self._content = io.BytesIO(content)

    async def read(self, byte_count: int = -1) -> bytes:
        """Read bytes."""
        if byte_count == -1:
            return self._content.read()
        return self._content.read(byte_count)


class MockPayloadWriter:
    """Small mock to imitate payload writer."""

    def enable_chunking(self) -> None:
        """Enable chunking."""
        pass

    async def write_headers(self, *args: Any, **kwargs: Any) -> None:
        """Write headers."""
        pass


_MOCK_PAYLOAD_WRITER = MockPayloadWriter()


class MockRequest:
    """Mock an aiohttp request."""
    mock_source: Optional[Any] = None

    def __init__(
        self,
        content: bytes,
        mock_source: Any,
        method: str = 'GET',
        status: HTTPStatus = HTTPStatus.OK,
        headers: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
        url: str = ''
    ) -> None:
        """Initialize a request."""
        self.method: str = method
        self.url: str = url
        self.status: HTTPStatus = status
        self.headers: CIMultiDict[str] = CIMultiDict(headers or {})
        self.query_string: str = query_string or ''
        self.keep_alive: bool = False
        self.version: tuple[int, int] = (1, 1)
        self._content: bytes = content
        self.mock_source: Any = mock_source
        self._payload_writer: MockPayloadWriter = _MOCK_PAYLOAD_WRITER

    async def _prepare_hook(self, response: web.Response) -> None:
        """Prepare hook."""
        pass

    @property
    def query(self) -> MultiDict[str, str]:
        """Return a dictionary with the query variables."""
        return MultiDict(parse_qsl(self.query_string, keep_blank_values=True))

    @property
    def _text(self) -> str:
        """Return the body as text."""
        return self._content.decode('utf-8')

    @property
    def content(self) -> MockStreamReader:
        """Return the body as text."""
        return MockStreamReader(self._content)

    @property
    def body_exists(self) -> bool:
        """Return True if request has HTTP BODY, False otherwise."""
        return bool(self._text)

    async def json(self, loads: JSONDecoder = json_loads) -> Any:
        """Return the body as JSON."""
        return loads(self._text)

    async def post(self) -> MultiDict[str, str]:
        """Return POST parameters."""
        return MultiDict(parse_qsl(self._text, keep_blank_values=True))

    async def text(self) -> str:
        """Return the body as text."""
        return self._text


def serialize_response(response: web.Response) -> Dict[str, Union[int, Optional[str], Dict[str, str]]]:
    """Serialize an aiohttp response to a dictionary."""
    if (body := response.body) is None:
        body_decoded: Optional[str] = None
    elif isinstance(body, payload.StringPayload):
        body_decoded = body._value.decode(body.encoding or 'utf-8')
    elif isinstance(body, bytes):
        body_decoded = body.decode(response.charset or 'utf-8')
    else:
        raise TypeError('Unknown payload encoding')
    return {
        'status': response.status,
        'body': body_decoded,
        'headers': dict(response.headers)
    }
