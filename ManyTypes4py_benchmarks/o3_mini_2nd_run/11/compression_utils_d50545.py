import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast

try:
    try:
        import brotlicffi as brotli  # type: ignore
    except ImportError:
        import brotli  # type: ignore
    HAS_BROTLI: bool = True
except ImportError:
    HAS_BROTLI = False

MAX_SYNC_CHUNK_SIZE: int = 1024

def encoding_to_mode(encoding: Optional[str] = None, suppress_deflate_header: bool = False) -> int:
    if encoding == 'gzip':
        return 16 + zlib.MAX_WBITS
    return -zlib.MAX_WBITS if suppress_deflate_header else zlib.MAX_WBITS

class ZlibBaseHandler:
    def __init__(
        self,
        mode: int,
        executor: Optional[Executor] = None,
        max_sync_chunk_size: int = MAX_SYNC_CHUNK_SIZE
    ) -> None:
        self._mode: int = mode
        self._executor: Optional[Executor] = executor
        self._max_sync_chunk_size: int = max_sync_chunk_size

class ZLibCompressor(ZlibBaseHandler):
    def __init__(
        self,
        encoding: Optional[str] = None,
        suppress_deflate_header: bool = False,
        level: Optional[int] = None,
        wbits: Optional[int] = None,
        strategy: int = zlib.Z_DEFAULT_STRATEGY,
        executor: Optional[Executor] = None,
        max_sync_chunk_size: int = MAX_SYNC_CHUNK_SIZE
    ) -> None:
        mode = encoding_to_mode(encoding, suppress_deflate_header) if wbits is None else wbits
        super().__init__(mode=mode, executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        if level is None:
            self._compressor: zlib.compressobj = zlib.compressobj(wbits=self._mode, strategy=strategy)
        else:
            self._compressor = zlib.compressobj(wbits=self._mode, strategy=strategy, level=level)
        self._compress_lock: asyncio.Lock = asyncio.Lock()

    def compress_sync(self, data: bytes) -> bytes:
        return self._compressor.compress(data)

    async def compress(self, data: bytes) -> bytes:
        """Compress the data and return the compressed bytes.
        
        Note that flush() must be called after the last call to compress()
        
        If the data size is larger than the max_sync_chunk_size, the compression
        will be done in the executor. Otherwise, the compression will be done
        in the event loop.
        """
        async with self._compress_lock:
            if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(self._executor, self._compressor.compress, data)
            return self.compress_sync(data)

    def flush(self, mode: int = zlib.Z_FINISH) -> bytes:
        return self._compressor.flush(mode)

class ZLibDecompressor(ZlibBaseHandler):
    def __init__(
        self,
        encoding: Optional[str] = None,
        suppress_deflate_header: bool = False,
        executor: Optional[Executor] = None,
        max_sync_chunk_size: int = MAX_SYNC_CHUNK_SIZE
    ) -> None:
        mode = encoding_to_mode(encoding, suppress_deflate_header)
        super().__init__(mode=mode, executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        self._decompressor: zlib.decompressobj = zlib.decompressobj(wbits=self._mode)

    def decompress_sync(self, data: bytes, max_length: int = 0) -> bytes:
        return self._decompressor.decompress(data, max_length)

    async def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        """Decompress the data and return the decompressed bytes.
        
        If the data size is larger than the max_sync_chunk_size, the decompression
        will be done in the executor. Otherwise, the decompression will be done
        in the event loop.
        """
        if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._decompressor.decompress, data, max_length)
        return self.decompress_sync(data, max_length)

    def flush(self, length: int = 0) -> bytes:
        if length > 0:
            return self._decompressor.flush(length)
        return self._decompressor.flush()

    @property
    def eof(self) -> bool:
        return self._decompressor.eof

    @property
    def unconsumed_tail(self) -> bytes:
        return self._decompressor.unconsumed_tail

    @property
    def unused_data(self) -> bytes:
        return self._decompressor.unused_data

class BrotliDecompressor:
    def __init__(self) -> None:
        if not HAS_BROTLI:
            raise RuntimeError('The brotli decompression is not available. Please install `Brotli` module')
        self._obj = brotli.Decompressor()

    def decompress_sync(self, data: bytes) -> bytes:
        if hasattr(self._obj, 'decompress'):
            return cast(bytes, self._obj.decompress(data))
        return cast(bytes, self._obj.process(data))

    def flush(self) -> bytes:
        if hasattr(self._obj, 'flush'):
            return cast(bytes, self._obj.flush())
        return b''