import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False
MAX_SYNC_CHUNK_SIZE = 1024

def encoding_to_mode(encoding: Union[None, bool, typing.Sequence[str]]=None, suppress_deflate_header: bool=False):
    if encoding == 'gzip':
        return 16 + zlib.MAX_WBITS
    return -zlib.MAX_WBITS if suppress_deflate_header else zlib.MAX_WBITS

class ZlibBaseHandler:

    def __init__(self, mode: Union[str, None, bool, list], executor: Union[None, bool, asyncio.AbstractEventLoop, asyncio.base_events.BaseEventLoop]=None, max_sync_chunk_size: Union[int, str]=MAX_SYNC_CHUNK_SIZE) -> None:
        self._mode = mode
        self._executor = executor
        self._max_sync_chunk_size = max_sync_chunk_size

class ZLibCompressor(ZlibBaseHandler):

    def __init__(self, encoding=None, suppress_deflate_header=False, level=None, wbits=None, strategy=zlib.Z_DEFAULT_STRATEGY, executor: Union[None, bool, asyncio.AbstractEventLoop, asyncio.base_events.BaseEventLoop]=None, max_sync_chunk_size: Union[int, str]=MAX_SYNC_CHUNK_SIZE) -> None:
        super().__init__(mode=encoding_to_mode(encoding, suppress_deflate_header) if wbits is None else wbits, executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        if level is None:
            self._compressor = zlib.compressobj(wbits=self._mode, strategy=strategy)
        else:
            self._compressor = zlib.compressobj(wbits=self._mode, strategy=strategy, level=level)
        self._compress_lock = asyncio.Lock()

    def compress_sync(self, data: bytes) -> Union[typing.TextIO, dict, typing.IO]:
        return self._compressor.compress(data)

    async def compress(self, data):
        """Compress the data and returned the compressed bytes.

        Note that flush() must be called after the last call to compress()

        If the data size is large than the max_sync_chunk_size, the compression
        will be done in the executor. Otherwise, the compression will be done
        in the event loop.
        """
        async with self._compress_lock:
            if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
                return await asyncio.get_running_loop().run_in_executor(self._executor, self._compressor.compress, data)
            return self.compress_sync(data)

    def flush(self, mode=zlib.Z_FINISH) -> Union[str, int]:
        return self._compressor.flush(mode)

class ZLibDecompressor(ZlibBaseHandler):

    def __init__(self, encoding=None, suppress_deflate_header=False, executor: Union[None, bool, asyncio.AbstractEventLoop, asyncio.base_events.BaseEventLoop]=None, max_sync_chunk_size: Union[int, str]=MAX_SYNC_CHUNK_SIZE) -> None:
        super().__init__(mode=encoding_to_mode(encoding, suppress_deflate_header), executor=executor, max_sync_chunk_size=max_sync_chunk_size)
        self._decompressor = zlib.decompressobj(wbits=self._mode)

    def decompress_sync(self, data: Union[bytes, bytearray, memoryview, int], max_length: int=0) -> Union[str, typing.BinaryIO, bytes]:
        return self._decompressor.decompress(data, max_length)

    async def decompress(self, data, max_length=0):
        """Decompress the data and return the decompressed bytes.

        If the data size is large than the max_sync_chunk_size, the decompression
        will be done in the executor. Otherwise, the decompression will be done
        in the event loop.
        """
        if self._max_sync_chunk_size is not None and len(data) > self._max_sync_chunk_size:
            return await asyncio.get_running_loop().run_in_executor(self._executor, self._decompressor.decompress, data, max_length)
        return self.decompress_sync(data, max_length)

    def flush(self, length: int=0) -> Union[str, int]:
        return self._decompressor.flush(length) if length > 0 else self._decompressor.flush()

    @property
    def eof(self):
        return self._decompressor.eof

    @property
    def unconsumed_tail(self):
        return self._decompressor.unconsumed_tail

    @property
    def unused_data(self):
        return self._decompressor.unused_data

class BrotliDecompressor:

    def __init__(self) -> None:
        if not HAS_BROTLI:
            raise RuntimeError('The brotli decompression is not available. Please install `Brotli` module')
        self._obj = brotli.Decompressor()

    def decompress_sync(self, data: Union[bytes, bytearray, memoryview, int]) -> Union[str, typing.BinaryIO, bytes]:
        if hasattr(self._obj, 'decompress'):
            return cast(bytes, self._obj.decompress(data))
        return cast(bytes, self._obj.process(data))

    def flush(self) -> Union[str, int]:
        if hasattr(self._obj, 'flush'):
            return cast(bytes, self._obj.flush())
        return b''