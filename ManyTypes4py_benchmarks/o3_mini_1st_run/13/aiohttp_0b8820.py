from __future__ import annotations
from http import HTTPStatus
import io
from typing import Any, Optional, Union, Dict
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


_MOCK_PAYLOAD_WRITER: MockPayloadWriter = MockPayloadWriter()


class MockRequest:
    mock_source: Any

    def __init__(self, content: bytes, mock_source: Any, method: str = 'GET',
                 status: HTTPStatus = HTTPStatus.OK, headers: Optional[Dict[str, str]] = None,
                 query_string: Optional[str] = None, url: str = '') -> None:
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
        pass

    @property
    def query(self) -> MultiDict[str]:
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

    async def post(self) -> MultiDict[str]:
        return MultiDict(parse_qsl(self._text, keep_blank_values=True))

    async def text(self) -> str:
        return self._text


def serialize_response(response: web.StreamResponse) -> Dict[str, Union[int, Optional[str], Dict[str, str]]]:
    body: Optional[Union[payload.StringPayload, bytes]] = getattr(response, 'body', None)
    if body is None:
        body_decoded: Optional[str] = None
    elif isinstance(body, payload.StringPayload):
        body_decoded = body._value.decode(body.encoding or 'utf-8')
    elif isinstance(body, bytes):
        body_decoded = body.decode(getattr(response, 'charset', 'utf-8'))
    else:
        raise TypeError('Unknown payload encoding')
    return {'status': response.status, 'body': body_decoded, 'headers': dict(response.headers)}