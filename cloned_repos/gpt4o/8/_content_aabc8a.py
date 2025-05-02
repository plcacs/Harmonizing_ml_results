from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping, Tuple, Union
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import AsyncByteStream, RequestContent, RequestData, RequestFiles, ResponseContent, SyncByteStream
from ._utils import peek_filelike_length, primitive_value_to_str

__all__ = ['ByteStream']

class ByteStream(AsyncByteStream, SyncByteStream):

    def __init__(self, stream: Union[bytes, str]) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[Union[bytes, str]]:
        yield self._stream

    async def __aiter__(self) -> AsyncIterator[Union[bytes, str]]:
        yield self._stream

class IteratorByteStream(SyncByteStream):
    CHUNK_SIZE = 65536

    def __init__(self, stream: Union[Iterable[bytes], Iterable[str]]) -> None:
        self._stream = stream
        self._is_stream_consumed = False
        self._is_generator = inspect.isgenerator(stream)

    def __iter__(self) -> Iterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()
        self._is_stream_consumed = True
        if hasattr(self._stream, 'read'):
            chunk = self._stream.read(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = self._stream.read(self.CHUNK_SIZE)
        else:
            for part in self._stream:
                yield part

class AsyncIteratorByteStream(AsyncByteStream):
    CHUNK_SIZE = 65536

    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream = stream
        self._is_stream_consumed = False
        self._is_generator = inspect.isasyncgen(stream)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()
        self._is_stream_consumed = True
        if hasattr(self._stream, 'aread'):
            chunk = await self._stream.aread(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = await self._stream.aread(self.CHUNK_SIZE)
        else:
            async for part in self._stream:
                yield part

class UnattachedStream(AsyncByteStream, SyncByteStream):
    """
    If a request or response is serialized using pickle, then it is no longer
    attached to a stream for I/O purposes. Any stream operations should result
    in `httpx.StreamClosed`.
    """

    def __iter__(self) -> Iterator[bytes]:
        raise StreamClosed()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise StreamClosed()
        yield b''

def encode_content(content: Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]]) -> Tuple[dict, Union[ByteStream, IteratorByteStream, AsyncIteratorByteStream]]:
    if isinstance(content, (bytes, str)):
        body = content.encode('utf-8') if isinstance(content, str) else content
        content_length = len(body)
        headers = {'Content-Length': str(content_length)} if body else {}
        return (headers, ByteStream(body))
    elif isinstance(content, Iterable) and (not isinstance(content, dict)):
        content_length_or_none = peek_filelike_length(content)
        if content_length_or_none is None:
            headers = {'Transfer-Encoding': 'chunked'}
        else:
            headers = {'Content-Length': str(content_length_or_none)}
        return (headers, IteratorByteStream(content))
    elif isinstance(content, AsyncIterable):
        headers = {'Transfer-Encoding': 'chunked'}
        return (headers, AsyncIteratorByteStream(content))
    raise TypeError(f"Unexpected type for 'content', {type(content)!r}")

def encode_urlencoded_data(data: Mapping[str, Any]) -> Tuple[dict, ByteStream]:
    plain_data = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            plain_data.extend([(key, primitive_value_to_str(item)) for item in value])
        else:
            plain_data.append((key, primitive_value_to_str(value)))
    body = urlencode(plain_data, doseq=True).encode('utf-8')
    content_length = str(len(body))
    content_type = 'application/x-www-form-urlencoded'
    headers = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))

def encode_multipart_data(data: Mapping[str, Any], files: RequestFiles, boundary: str) -> Tuple[dict, MultipartStream]:
    multipart = MultipartStream(data=data, files=files, boundary=boundary)
    headers = multipart.get_headers()
    return (headers, multipart)

def encode_text(text: str) -> Tuple[dict, ByteStream]:
    body = text.encode('utf-8')
    content_length = str(len(body))
    content_type = 'text/plain; charset=utf-8'
    headers = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))

def encode_html(html: str) -> Tuple[dict, ByteStream]:
    body = html.encode('utf-8')
    content_length = str(len(body))
    content_type = 'text/html; charset=utf-8'
    headers = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))

def encode_json(json: Any) -> Tuple[dict, ByteStream]:
    body = json_dumps(json, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    content_length = str(len(body))
    content_type = 'application/json'
    headers = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))

def encode_request(content: Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]] = None, data: RequestData = None, files: RequestFiles = None, json: Any = None, boundary: str = None) -> Tuple[dict, Union[ByteStream, IteratorByteStream, AsyncIteratorByteStream, MultipartStream]]:
    """
    Handles encoding the given `content`, `data`, `files`, and `json`,
    returning a two-tuple of (<headers>, <stream>).
    """
    if data is not None and (not isinstance(data, Mapping)):
        message = "Use 'content=<...>' to upload raw bytes/text content."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return encode_content(data)
    if content is not None:
        return encode_content(content)
    elif files:
        return encode_multipart_data(data or {}, files, boundary)
    elif data:
        return encode_urlencoded_data(data)
    elif json is not None:
        return encode_json(json)
    return ({}, ByteStream(b''))

def encode_response(content: Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]] = None, text: str = None, html: str = None, json: Any = None) -> Tuple[dict, ByteStream]:
    """
    Handles encoding the given `content`, returning a two-tuple of
    (<headers>, <stream>).
    """
    if content is not None:
        return encode_content(content)
    elif text is not None:
        return encode_text(text)
    elif html is not None:
        return encode_html(html)
    elif json is not None:
        return encode_json(json)
    return ({}, ByteStream(b''))
