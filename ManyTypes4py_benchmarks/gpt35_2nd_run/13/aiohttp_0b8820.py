from __future__ import annotations
from http import HTTPStatus
import io
from typing import Any
from urllib.parse import parse_qsl
from aiohttp import payload, web
from aiohttp.typedefs import JSONDecoder
from multidict import CIMultiDict, MultiDict
from .json import json_loads

class MockStreamReader:
    def __init__(self, content: bytes) -> None:
        self._content = io.BytesIO(content)

    async def read(self, byte_count: int = -1) -> bytes:
        if byte_count == -1:
            return self._content.read()
        return self._content.read(byte_count)

class MockPayloadWriter:
    def enable_chunking(self) -> None:
        pass

    async def write_headers(self, *args: Any, **kwargs: Any) -> None:
        pass

_MOCK_PAYLOAD_WRITER = MockPayloadWriter()

class MockRequest:
    def __init__(self, content: bytes, mock_source: Any, method: str = 'GET', status: HTTPStatus = HTTPStatus.OK, headers: dict = None, query_string: str = None, url: str = '') -> None:
        self.method = method
        self.url = url
        self.status = status
        self.headers = CIMultiDict(headers or {})
        self.query_string = query_string or ''
        self.keep_alive = False
        self.version = (1, 1)
        self._content = content
        self.mock_source = mock_source
        self._payload_writer = _MOCK_PAYLOAD_WRITER

    async def _prepare_hook(self, response: Any) -> None:
        pass

    @property
    def query(self) -> MultiDict:
        return MultiDict(parse_qsl(self.query_string, keep_blank_values=True))

    @property
    def _text(self) -> str:
        return self._content.decode('utf-8')

    @property
    def content(self) -> MockStreamReader:
        return MockStreamReader(self._content)

    @property
    def body_exists(self) -> bool:
        return bool(self._text)

    async def json(self, loads: JSONDecoder = json_loads) -> Any:
        return loads(self._text)

    async def post(self) -> MultiDict:
        return MultiDict(parse_qsl(self._text, keep_blank_values=True))

    async def text(self) -> str:
        return self._text

def serialize_response(response: Any) -> dict:
    if (body := response.body) is None:
        body_decoded = None
    elif isinstance(body, payload.StringPayload):
        body_decoded = body._value.decode(body.encoding or 'utf-8')
    elif isinstance(body, bytes):
        body_decoded = body.decode(response.charset or 'utf-8')
    else:
        raise TypeError('Unknown payload encoding')
    return {'status': response.status, 'body': body_decoded, 'headers': dict(response.headers)}
