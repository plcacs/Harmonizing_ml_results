from __future__ import annotations
from http import HTTPStatus
import io
from typing import Any, Callable, Optional, Dict
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


_MOCK_PAYLOAD_WRITER: MockPayloadWriter = MockPayloadWriter()


class MockRequest:
    """Mock an aiohttp request."""
    mock_source: Any = None

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
        self.mock_source = mock_source
        self._payload_writer: MockPayloadWriter = _MOCK_PAYLOAD_WRITER

    async def _prepare_hook(self, response: web.StreamResponse) -> None:
        """Prepare hook."""
        pass

    @property
    def query(self) -> MultiDict[str]:
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

    async def json(self, loads: Callable[[str], Any] = json_loads) -> Any:
        """Return the body as JSON."""
        return loads(self._text)

    async def post(self) -> MultiDict[str]:
        """Return POST parameters."""
        return MultiDict(parse_qsl(self._text, keep_blank_values=True))

    async def text(self) -> str:
        """Return the body as text."""
        return self._text


def serialize_response(response: web.Response) -> Dict[str, Any]:
    """Serialize an aiohttp response to a dictionary."""
    body = response.body
    if body is None:
        body_decoded: Optional[str] = None
    elif isinstance(body, payload.StringPayload):
        body_decoded = body._value.decode(body.encoding or 'utf-8')
    elif isinstance(body, bytes):
        body_decoded = body.decode(response.charset or 'utf-8')
    else:
        raise TypeError('Unknown payload encoding')
    return {'status': response.status, 'body': body_decoded, 'headers': dict(response.headers)}