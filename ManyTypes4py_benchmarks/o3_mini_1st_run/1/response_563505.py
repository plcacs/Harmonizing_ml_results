from __future__ import absolute_import
from contextlib import contextmanager
import zlib
import io
import logging
from socket import timeout as SocketTimeout
from socket import error as SocketError
import h11
from .._collections import HTTPHeaderDict
from ..exceptions import ProtocolError, DecodeError, ReadTimeoutError
from ..packages.six import string_types as basestring, binary_type
from ..util.ssl_ import BaseSSLError
from typing import Any, Optional, Union, Iterator, AsyncGenerator, Coroutine

log = logging.getLogger(__name__)


class DeflateDecoder(object):
    def __init__(self) -> None:
        self._first_try: bool = True
        self._data: bytes = binary_type()
        self._obj = zlib.decompressobj()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def decompress(self, data: bytes) -> bytes:
        if not data:
            return data
        if not self._first_try:
            return self._obj.decompress(data)
        self._data += data
        try:
            decompressed: bytes = self._obj.decompress(data)
            if decompressed:
                self._first_try = False
                self._data = None  # type: ignore
            return decompressed
        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                return self.decompress(self._data)  # type: ignore
            finally:
                self._data = None  # type: ignore


class GzipDecoder(object):
    def __init__(self) -> None:
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def decompress(self, data: bytes) -> bytes:
        if not data:
            return data
        return self._obj.decompress(data)


def _get_decoder(mode: str) -> Union[GzipDecoder, DeflateDecoder]:
    if mode == 'gzip':
        return GzipDecoder()
    return DeflateDecoder()


class HTTPResponse(io.IOBase):
    """
    HTTP Response container.

    Backwards-compatible to httplib's HTTPResponse but the response ``body`` is
    loaded and decoded on-demand when the ``data`` property is accessed.  This
    class is also compatible with the Python standard library's :mod:`io`
    module, and can hence be treated as a readable object in the context of that
    framework.

    Extra parameters for behaviour not present in httplib.HTTPResponse:

    :param preload_content:
        If True, the response's body will be preloaded during construction.

    :param decode_content:
        If True, attempts to decode specific content-encoding's based on headers
        (like 'gzip' and 'deflate') will be skipped and raw data will be used
        instead.

    :param retries:
        The retries contains the last :class:`~urllib3.util.retry.Retry` that
        was used during the request.
    """
    CONTENT_DECODERS = ['gzip', 'deflate']
    REDIRECT_STATUSES = [301, 302, 303, 307, 308]

    def __init__(self, 
                 body: Union[str, bytes, io.IOBase] = '', 
                 headers: Optional[Any] = None, 
                 status: int = 0, 
                 version: int = 0, 
                 reason: Optional[str] = None, 
                 strict: int = 0, 
                 preload_content: bool = True, 
                 decode_content: bool = True, 
                 original_response: Optional[Any] = None, 
                 pool: Optional[Any] = None, 
                 connection: Optional[Any] = None, 
                 retries: Optional[Any] = None, 
                 request_method: Optional[Any] = None) -> None:
        if isinstance(headers, HTTPHeaderDict):
            self.headers: HTTPHeaderDict = headers
        else:
            self.headers = HTTPHeaderDict(headers)
        self.status: int = status
        self.version: int = version
        self.reason: Optional[str] = reason
        self.strict: int = strict
        self.decode_content: bool = decode_content
        self.retries: Optional[Any] = retries
        self._decoder: Optional[Union[GzipDecoder, DeflateDecoder]] = None
        self._body: Optional[bytes] = None
        self._fp: Optional[Any] = None
        self._original_response: Optional[Any] = original_response
        self._fp_bytes_read: int = 0
        self._buffer: bytes = b''
        if body and isinstance(body, (basestring, binary_type)):
            self._body = body  # type: ignore
        else:
            self._fp = body
        self._pool: Optional[Any] = pool
        self._connection: Optional[Any] = connection
        if preload_content and (not self._body):
            # Since read is async, we must assume it is awaited elsewhere;
            # here we will simulate the preload by calling the coroutine.
            import asyncio
            loop = asyncio.get_event_loop()
            self._body = loop.run_until_complete(self.read(decode_content=decode_content))

    def get_redirect_location(self) -> Union[str, None, bool]:
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get('location')
        return False

    async def release_conn(self) -> None:
        if not self._pool or not self._connection:
            return
        await self._pool._put_conn(self._connection)
        self._connection = None

    @property
    def data(self) -> Optional[bytes]:
        if self._body is not None:
            return self._body
        if self._fp:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.read(cache_content=True))
        return None

    @property
    def connection(self) -> Optional[Any]:
        return self._connection

    def tell(self) -> int:
        return self._fp_bytes_read

    def _init_decoder(self) -> None:
        content_encoding: str = self.headers.get('content-encoding', '').lower()
        if self._decoder is None and content_encoding in self.CONTENT_DECODERS:
            self._decoder = _get_decoder(content_encoding)

    def _decode(self, data: bytes, decode_content: bool, flush_decoder: bool) -> bytes:
        try:
            if decode_content and self._decoder:
                data = self._decoder.decompress(data)
        except (IOError, zlib.error) as e:
            content_encoding: str = self.headers.get('content-encoding', '').lower()
            raise DecodeError('Received response with content-encoding: %s, but failed to decode it.' % content_encoding, e)
        if flush_decoder and decode_content:
            data += self._flush_decoder()
        return data

    def _flush_decoder(self) -> bytes:
        if self._decoder:
            buf: bytes = self._decoder.decompress(b'')
            return buf + self._decoder.flush()
        return b''

    @contextmanager
    def _error_catcher(self) -> Iterator[None]:
        clean_exit: bool = False
        try:
            try:
                yield
            except SocketTimeout:
                raise ReadTimeoutError(self._pool, None, 'Read timed out.')
            except BaseSSLError as e:
                if 'read operation timed out' not in str(e):
                    raise
                raise ReadTimeoutError(self._pool, None, 'Read timed out.')
            except (h11.ProtocolError, SocketError) as e:
                raise ProtocolError('Connection broken: %r' % e, e)
            except GeneratorExit:
                pass
            clean_exit = True
        finally:
            if not clean_exit:
                self.close()
            if False and self._original_response and getattr(self._original_response, 'complete', False):
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.release_conn())

    async def read(self, 
                   amt: Optional[int] = None, 
                   decode_content: Optional[bool] = None, 
                   cache_content: bool = False) -> bytes:
        if self._fp is None and (not self._buffer):
            return b''
        data: bytes = self._buffer
        with self._error_catcher():
            if amt is None:
                chunks: list[bytes] = []
                async for chunk in self.stream(decode_content):
                    chunks.append(chunk)
                data += b''.join(chunks)
                self._buffer = b''
                self._body = data
            else:
                data_len: int = len(data)
                chunks: list[bytes] = [data]
                streamer = self.stream(decode_content)
                async for chunk in streamer:
                    if data_len >= amt:
                        break
                    chunks.append(chunk)
                    data_len += len(chunk)
                data = b''.join(chunks)
                self._buffer = data[amt:]
                data = data[:amt]
        return data

    async def stream(self, decode_content: Optional[bool] = None) -> AsyncGenerator[bytes, None]:
        if self._fp is None:
            return
        self._init_decoder()
        if decode_content is None:
            decode_content = self.decode_content
        with self._error_catcher():
            async for raw_chunk in self._fp:
                self._fp_bytes_read += len(raw_chunk)
                decoded_chunk: bytes = self._decode(raw_chunk, decode_content, flush_decoder=False)
                if decoded_chunk:
                    yield decoded_chunk
            final_chunk: bytes = self._decode(b'', decode_content, flush_decoder=True)
            if final_chunk:
                yield final_chunk
            self._fp = None

    @classmethod
    def from_base(cls, r: Any, **response_kw: Any) -> "HTTPResponse":
        for kw in ('redirect', 'assert_same_host', 'enforce_content_length'):
            if kw in response_kw:
                response_kw.pop(kw)
        resp = cls(body=r.body, 
                   headers=r.headers, 
                   status=r.status_code, 
                   version=r.version, 
                   reason=None, 
                   original_response=r, 
                   connection=r.body, 
                   **response_kw)
        return resp

    def getheaders(self) -> HTTPHeaderDict:
        return self.headers

    def getheader(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.headers.get(name, default)

    def info(self) -> HTTPHeaderDict:
        return self.headers

    def close(self) -> None:
        if not self.closed:
            if self._fp is not None:
                self._fp.close()
            self._buffer = b''
            self._fp = None
        if self._connection:
            self._connection.close()

    @property
    def closed(self) -> bool:
        if self._fp is None and (not self._buffer):
            return True
        elif hasattr(self._fp, 'complete'):
            return self._fp.complete  # type: ignore
        else:
            return False

    def fileno(self) -> int:
        if self._fp is None:
            raise IOError('HTTPResponse has no file to get a fileno from')
        elif hasattr(self._fp, 'fileno'):
            return self._fp.fileno()  # type: ignore
        else:
            raise IOError('The file-like object this HTTPResponse is wrapped around has no file descriptor')

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        temp: bytes = self.read(len(b)).result()  # Blocking call resolution from coroutine
        if not temp:
            return 0
        b[:len(temp)] = temp
        return len(temp)