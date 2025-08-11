"""Http related parsers and protocol."""
import asyncio
import sys
import zlib
from typing import Any, Awaitable, Callable, Iterable, List, NamedTuple, Optional, Union
from multidict import CIMultiDict
from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .client_exceptions import ClientConnectionResetError
from .compression_utils import ZLibCompressor
from .helpers import NO_EXTENSIONS
__all__ = ('StreamWriter', 'HttpVersion', 'HttpVersion10', 'HttpVersion11')
MIN_PAYLOAD_FOR_WRITELINES = 2048
IS_PY313_BEFORE_313_2 = (3, 13, 0) <= sys.version_info < (3, 13, 2)
IS_PY_BEFORE_312_9 = sys.version_info < (3, 12, 9)
SKIP_WRITELINES = IS_PY313_BEFORE_313_2 or IS_PY_BEFORE_312_9

class HttpVersion(NamedTuple):
    pass
HttpVersion10 = HttpVersion(1, 0)
HttpVersion11 = HttpVersion(1, 1)
_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]
_T_OnHeadersSent = Optional[Callable[['CIMultiDict[str]'], Awaitable[None]]]

class StreamWriter(AbstractStreamWriter):
    length = None
    chunked = False
    _eof = False
    _compress = None

    def __init__(self, protocol: Union[int, dict[str, int], asyncio.AbstractEventLoop], loop: Union[asyncio.AbstractEventLoop, None], on_chunk_sent: Union[None, int, asyncio.AbstractEventLoop, str]=None, on_headers_sent: Union[None, str, bytes]=None) -> None:
        self._protocol = protocol
        self.loop = loop
        self._on_chunk_sent = on_chunk_sent
        self._on_headers_sent = on_headers_sent

    @property
    def transport(self):
        return self._protocol.transport

    @property
    def protocol(self):
        return self._protocol

    def enable_chunking(self) -> None:
        self.chunked = True

    def enable_compression(self, encoding: typing.Text='deflate', strategy: Any=zlib.Z_DEFAULT_STRATEGY) -> None:
        self._compress = ZLibCompressor(encoding=encoding, strategy=strategy)

    def _write(self, chunk: bytes) -> None:
        size = len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self._protocol.transport
        if transport is None or transport.is_closing():
            raise ClientConnectionResetError('Cannot write to closing transport')
        transport.write(chunk)

    def _writelines(self, chunks: Union[bytes, list[bytes], str]) -> None:
        size = 0
        for chunk in chunks:
            size += len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self._protocol.transport
        if transport is None or transport.is_closing():
            raise ClientConnectionResetError('Cannot write to closing transport')
        if SKIP_WRITELINES or size < MIN_PAYLOAD_FOR_WRITELINES:
            transport.write(b''.join(chunks))
        else:
            transport.writelines(chunks)

    async def write(self, chunk, *, drain=True, LIMIT=65536):
        """Writes chunk of data to a stream.

        write_eof() indicates end of stream.
        writer can't be used after write_eof() method being called.
        write() return drain future.
        """
        if self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)
        if isinstance(chunk, memoryview):
            if chunk.nbytes != len(chunk):
                chunk = chunk.cast('c')
        if self._compress is not None:
            chunk = await self._compress.compress(chunk)
            if not chunk:
                return
        if self.length is not None:
            chunk_len = len(chunk)
            if self.length >= chunk_len:
                self.length = self.length - chunk_len
            else:
                chunk = chunk[:self.length]
                self.length = 0
                if not chunk:
                    return
        if chunk:
            if self.chunked:
                self._writelines((f'{len(chunk):x}\r\n'.encode('ascii'), chunk, b'\r\n'))
            else:
                self._write(chunk)
            if self.buffer_size > LIMIT and drain:
                self.buffer_size = 0
                await self.drain()

    async def write_headers(self, status_line, headers):
        """Write request/response status and headers."""
        if self._on_headers_sent is not None:
            await self._on_headers_sent(headers)
        buf = _serialize_headers(status_line, headers)
        self._write(buf)

    def set_eof(self) -> None:
        """Indicate that the message is complete."""
        self._eof = True

    async def write_eof(self, chunk=b''):
        if self._eof:
            return
        if chunk and self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)
        if self._compress:
            chunks = []
            chunks_len = 0
            if chunk and (compressed_chunk := (await self._compress.compress(chunk))):
                chunks_len = len(compressed_chunk)
                chunks.append(compressed_chunk)
            flush_chunk = self._compress.flush()
            chunks_len += len(flush_chunk)
            chunks.append(flush_chunk)
            assert chunks_len
            if self.chunked:
                chunk_len_pre = f'{chunks_len:x}\r\n'.encode('ascii')
                self._writelines((chunk_len_pre, *chunks, b'\r\n0\r\n\r\n'))
            elif len(chunks) > 1:
                self._writelines(chunks)
            else:
                self._write(chunks[0])
        elif self.chunked:
            if chunk:
                chunk_len_pre = f'{len(chunk):x}\r\n'.encode('ascii')
                self._writelines((chunk_len_pre, chunk, b'\r\n0\r\n\r\n'))
            else:
                self._write(b'0\r\n\r\n')
        elif chunk:
            self._write(chunk)
        await self.drain()
        self._eof = True

    async def drain(self):
        """Flush the write buffer.

        The intended use is to write

          await w.write(data)
          await w.drain()
        """
        protocol = self._protocol
        if protocol.transport is not None and protocol._paused:
            await protocol._drain_helper()

def _safe_header(string: str) -> str:
    if '\r' in string or '\n' in string:
        raise ValueError('Newline or carriage return detected in headers. Potential header injection attack.')
    return string

def _py_serialize_headers(status_line: str, headers: dict[str, str]) -> bytes:
    headers_gen = (_safe_header(k) + ': ' + _safe_header(v) for k, v in headers.items())
    line = status_line + '\r\n' + '\r\n'.join(headers_gen) + '\r\n\r\n'
    return line.encode('utf-8')
_serialize_headers = _py_serialize_headers
try:
    import aiohttp._http_writer as _http_writer
    _c_serialize_headers = _http_writer._serialize_headers
    if not NO_EXTENSIONS:
        _serialize_headers = _c_serialize_headers
except ImportError:
    pass