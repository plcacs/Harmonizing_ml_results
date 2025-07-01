from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping, Tuple, Dict, Union, Optional
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import AsyncByteStream, RequestContent, RequestData, RequestFiles, ResponseContent, SyncByteStream
from ._utils import peek_filelike_length, primitive_value_to_str
__all__ = ['ByteStream']


class ByteStream(AsyncByteStream, SyncByteStream):

    def __init__(self, stream: bytes) -> None:
        self._stream: bytes = stream

    def __iter__(self) -> Iterator[bytes]:
        yield self._stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._stream


class IteratorByteStream(SyncByteStream):
    CHUNK_SIZE: int = 65536

    def __init__(self, stream: Iterable[bytes]) -> None:
        self._stream: Iterable[bytes] = stream
        self._is_stream_consumed: bool = False
        self._is_generator: bool = inspect.isgenerator(stream)

    def __iter__(self) -> Iterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()
        self._is_stream_consumed = True
        if hasattr(self._stream, 'read') and callable(getattr(self._stream, 'read')):
            chunk: bytes = self._stream.read(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = self._stream.read(self.CHUNK_SIZE)
        else:
            for part in self._stream:
                yield part


class AsyncIteratorByteStream(AsyncByteStream):
    CHUNK_SIZE: int = 65536

    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream: AsyncIterable[bytes] = stream
        self._is_stream_consumed: bool = False
        self._is_generator: bool = inspect.isasyncgen(stream)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()
        self._is_stream_consumed = True
        if hasattr(self._stream, 'aread') and callable(getattr(self._stream, 'aread')):
            chunk: bytes = await self._stream.aread(self.CHUNK_SIZE)
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
        yield b''  # This line will never be reached


def encode_content(content: Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]]) -> Tuple[Dict[str, str], Union[ByteStream, IteratorByteStream, AsyncIteratorByteStream]]:
    if isinstance(content, (bytes, str)):
        body: bytes = content.encode('utf-8') if isinstance(content, str) else content
        content_length: int = len(body)
        headers: Dict[str, str] = {'Content-Length': str(content_length)} if body else {}
        return (headers, ByteStream(body))
    elif isinstance(content, Iterable) and not isinstance(content, dict):
        content_length_or_none: Optional[int] = peek_filelike_length(content)
        if content_length_or_none is None:
            headers: Dict[str, str] = {'Transfer-Encoding': 'chunked'}
        else:
            headers = {'Content-Length': str(content_length_or_none)}
        return (headers, IteratorByteStream(content))
    elif isinstance(content, AsyncIterable):
        headers = {'Transfer-Encoding': 'chunked'}
        return (headers, AsyncIteratorByteStream(content))
    raise TypeError(f"Unexpected type for 'content', {type(content)!r}")


def encode_urlencoded_data(data: Mapping[str, Any]) -> Tuple[Dict[str, str], ByteStream]:
    plain_data: list[tuple[str, str]] = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            plain_data.extend([(key, primitive_value_to_str(item)) for item in value])
        else:
            plain_data.append((key, primitive_value_to_str(value)))
    body: bytes = urlencode(plain_data, doseq=True).encode('utf-8')
    content_length: str = str(len(body))
    content_type: str = 'application/x-www-form-urlencoded'
    headers: Dict[str, str] = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))


def encode_multipart_data(data: Mapping[str, Any], files: RequestFiles, boundary: str) -> Tuple[Dict[str, str], MultipartStream]:
    multipart: MultipartStream = MultipartStream(data=data, files=files, boundary=boundary)
    headers: Dict[str, str] = multipart.get_headers()
    return (headers, multipart)


def encode_text(text: str) -> Tuple[Dict[str, str], ByteStream]:
    body: bytes = text.encode('utf-8')
    content_length: str = str(len(body))
    content_type: str = 'text/plain; charset=utf-8'
    headers: Dict[str, str] = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))


def encode_html(html: str) -> Tuple[Dict[str, str], ByteStream]:
    body: bytes = html.encode('utf-8')
    content_length: str = str(len(body))
    content_type: str = 'text/html; charset=utf-8'
    headers: Dict[str, str] = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))


def encode_json(json_data: Any) -> Tuple[Dict[str, str], ByteStream]:
    body: bytes = json_dumps(json_data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    content_length: str = str(len(body))
    content_type: str = 'application/json'
    headers: Dict[str, str] = {'Content-Length': content_length, 'Content-Type': content_type}
    return (headers, ByteStream(body))


def encode_request(
    content: Optional[Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]]] = None,
    data: Optional[Union[RequestData, Any]] = None,
    files: Optional[RequestFiles] = None,
    json: Optional[Any] = None,
    boundary: Optional[str] = None
) -> Tuple[Dict[str, str], Union[ByteStream, IteratorByteStream, AsyncIteratorByteStream, MultipartStream]]:
    """
    Handles encoding the given `content`, `data`, `files`, and `json`,
    returning a two-tuple of (<headers>, <stream>).
    """
    if data is not None and not isinstance(data, Mapping):
        message = "Use 'content=<...>' to upload raw bytes/text content."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return encode_content(data)
    if content is not None:
        return encode_content(content)
    elif files:
        return encode_multipart_data(data or {}, files, boundary or '')
    elif data:
        return encode_urlencoded_data(data)
    elif json is not None:
        return encode_json(json)
    return ({}, ByteStream(b''))


def encode_response(
    content: Optional[Union[bytes, str, Iterable[bytes], AsyncIterable[bytes]]] = None,
    text: Optional[str] = None,
    html: Optional[str] = None,
    json: Optional[Any] = None
) -> Tuple[Dict[str, str], Union[ByteStream, IteratorByteStream, AsyncIteratorByteStream]]:
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
