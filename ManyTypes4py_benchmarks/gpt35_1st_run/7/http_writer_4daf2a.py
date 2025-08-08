from typing import Any, Awaitable, Callable, Iterable, List, NamedTuple, Optional, Union

_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]
_T_OnHeadersSent = Optional[Callable[['CIMultiDict[str]'], Awaitable[None]]

class HttpVersion(NamedTuple):
    major: int
    minor: int

HttpVersion10: HttpVersion = HttpVersion(1, 0)
HttpVersion11: HttpVersion = HttpVersion(1, 1)

class StreamWriter(AbstractStreamWriter):
    length: Optional[int] = None
    chunked: bool = False
    _eof: bool = False
    _compress: Optional[ZLibCompressor] = None

    def __init__(self, protocol: BaseProtocol, loop: asyncio.AbstractEventLoop, on_chunk_sent: _T_OnChunkSent = None, on_headers_sent: _T_OnHeadersSent = None) -> None:
        self._protocol = protocol
        self.loop = loop
        self._on_chunk_sent = on_chunk_sent
        self._on_headers_sent = on_headers_sent

    @property
    def transport(self) -> Any:
        return self._protocol.transport

    @property
    def protocol(self) -> BaseProtocol:
        return self._protocol

    def enable_chunking(self) -> None:
        self.chunked = True

    def enable_compression(self, encoding: str = 'deflate', strategy: int = zlib.Z_DEFAULT_STRATEGY) -> None:
        self._compress = ZLibCompressor(encoding=encoding, strategy=strategy)

    def _write(self, chunk: bytes) -> None:
        size = len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self._protocol.transport
        if transport is None or transport.is_closing():
            raise ClientConnectionResetError('Cannot write to closing transport')
        transport.write(chunk)

    def _writelines(self, chunks: Iterable[bytes]) -> None:
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

    async def write(self, chunk: Union[bytes, memoryview], *, drain: bool = True, LIMIT: int = 65536) -> None:
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

    async def write_headers(self, status_line: str, headers: CIMultiDict[str]) -> None:
        if self._on_headers_sent is not None:
            await self._on_headers_sent(headers)
        buf = _serialize_headers(status_line, headers)
        self._write(buf)

    def set_eof(self) -> None:
        self._eof = True

    async def write_eof(self, chunk: bytes = b'') -> None:
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

    async def drain(self) -> None:
        protocol = self._protocol
        if protocol.transport is not None and protocol._paused:
            await protocol._drain_helper()
