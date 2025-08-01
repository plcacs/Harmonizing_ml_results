import asyncio
import collections
import warnings
from typing import Awaitable, Callable, Deque, Final, Generic, List, Optional, Tuple, Type, TypeVar, Union, Any, cast
from .base_protocol import BaseProtocol
from .helpers import _EXC_SENTINEL, BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
__all__ = ('EMPTY_PAYLOAD', 'EofStream', 'StreamReader', 'DataQueue')
_T = TypeVar('_T')

class EofStream(Exception):
    """eof stream indication."""

class AsyncStreamIterator(Generic[_T]):
    __slots__ = ('read_func',)

    def __init__(self, read_func: Callable[[], Awaitable[_T]]) -> None:
        self.read_func = read_func

    def __aiter__(self) -> 'AsyncStreamIterator[_T]':
        return self

    async def __anext__(self) -> _T:
        try:
            rv = await self.read_func()
        except EofStream:
            raise StopAsyncIteration
        if rv == b'':
            raise StopAsyncIteration
        return rv

class ChunkTupleAsyncStreamIterator:
    __slots__ = ('_stream',)

    def __init__(self, stream: 'StreamReader') -> None:
        self._stream = stream

    def __aiter__(self) -> 'ChunkTupleAsyncStreamIterator':
        return self

    async def __anext__(self) -> Tuple[bytes, bool]:
        rv = await self._stream.readchunk()
        if rv == (b'', False):
            raise StopAsyncIteration
        return rv

class AsyncStreamReaderMixin:
    __slots__ = ()

    def __aiter__(self) -> AsyncStreamIterator[bytes]:
        return AsyncStreamIterator(self.readline)

    def iter_chunked(self, n: int) -> AsyncStreamIterator[bytes]:
        """Returns an asynchronous iterator that yields chunks of size n."""
        return AsyncStreamIterator(lambda: self.read(n))

    def iter_any(self) -> AsyncStreamIterator[bytes]:
        """Yield all available data as soon as it is received."""
        return AsyncStreamIterator(self.readany)

    def iter_chunks(self) -> ChunkTupleAsyncStreamIterator:
        """Yield chunks of data as they are received by the server.

        The yielded objects are tuples
        of (bytes, bool) as returned by the StreamReader.readchunk method.
        """
        return ChunkTupleAsyncStreamIterator(self)

class StreamReader(AsyncStreamReaderMixin):
    """An enhancement of asyncio.StreamReader.

    Supports asynchronous iteration by line, chunk or as available::

        async for line in reader:
            ...
        async for chunk in reader.iter_chunked(1024):
            ...
        async for slice in reader.iter_any():
            ...

    """
    __slots__ = ('_protocol', '_low_water', '_high_water', '_loop', '_size', '_cursor', '_http_chunk_splits', '_buffer', '_buffer_offset', '_eof', '_waiter', '_eof_waiter', '_exception', '_timer', '_eof_callbacks', '_eof_counter', 'total_bytes')

    def __init__(self, protocol: BaseProtocol, limit: int, *, timer: Optional[BaseTimerContext] = None, loop: asyncio.AbstractEventLoop) -> None:
        self._protocol = protocol
        self._low_water = limit
        self._high_water = limit * 2
        self._loop = loop
        self._size = 0
        self._cursor = 0
        self._http_chunk_splits: Optional[List[int]] = None
        self._buffer: Deque[bytes] = collections.deque()
        self._buffer_offset = 0
        self._eof = False
        self._waiter: Optional[asyncio.Future[None]] = None
        self._eof_waiter: Optional[asyncio.Future[None]] = None
        self._exception: Optional[Exception] = None
        self._timer = TimerNoop() if timer is None else timer
        self._eof_callbacks: List[Callable[[], None]] = []
        self._eof_counter = 0
        self.total_bytes = 0

    def __repr__(self) -> str:
        info = [self.__class__.__name__]
        if self._size:
            info.append('%d bytes' % self._size)
        if self._eof:
            info.append('eof')
        if self._low_water != 2 ** 16:
            info.append('low=%d high=%d' % (self._low_water, self._high_water))
        if self._waiter:
            info.append('w=%r' % self._waiter)
        if self._exception:
            info.append('e=%r' % self._exception)
        return '<%s>' % ' '.join(info)

    def get_read_buffer_limits(self) -> Tuple[int, int]:
        return (self._low_water, self._high_water)

    def exception(self) -> Optional[Exception]:
        return self._exception

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        self._exception = exc
        self._eof_callbacks.clear()
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_exception(waiter, exc, exc_cause)
        waiter = self._eof_waiter
        if waiter is not None:
            self._eof_waiter = None
            set_exception(waiter, exc, exc_cause)

    def on_eof(self, callback: Callable[[], None]) -> None:
        if self._eof:
            try:
                callback()
            except Exception:
                internal_logger.exception('Exception in eof callback')
        else:
            self._eof_callbacks.append(callback)

    def feed_eof(self) -> None:
        self._eof = True
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)
        waiter = self._eof_waiter
        if waiter is not None:
            self._eof_waiter = None
            set_result(waiter, None)
        if self._protocol._reading_paused:
            self._protocol.resume_reading()
        for cb in self._eof_callbacks:
            try:
                cb()
            except Exception:
                internal_logger.exception('Exception in eof callback')
        self._eof_callbacks.clear()

    def is_eof(self) -> bool:
        """Return True if  'feed_eof' was called."""
        return self._eof

    def at_eof(self) -> bool:
        """Return True if the buffer is empty and 'feed_eof' was called."""
        return self._eof and (not self._buffer)

    async def wait_eof(self) -> None:
        if self._eof:
            return
        assert self._eof_waiter is None
        self._eof_waiter = self._loop.create_future()
        try:
            await self._eof_waiter
        finally:
            self._eof_waiter = None

    def unread_data(self, data: bytes) -> None:
        """rollback reading some data from stream, inserting it to buffer head."""
        warnings.warn('unread_data() is deprecated and will be removed in future releases (#3260)', DeprecationWarning, stacklevel=2)
        if not data:
            return
        if self._buffer_offset:
            self._buffer[0] = self._buffer[0][self._buffer_offset:]
            self._buffer_offset = 0
        self._size += len(data)
        self._cursor -= len(data)
        self._buffer.appendleft(data)
        self._eof_counter = 0

    def feed_data(self, data: bytes) -> None:
        assert not self._eof, 'feed_data after feed_eof'
        if not data:
            return
        data_len = len(data)
        self._size += data_len
        self._buffer.append(data)
        self.total_bytes += data_len
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)
        if self._size > self._high_water and (not self._protocol._reading_paused):
            self._protocol.pause_reading()

    def begin_http_chunk_receiving(self) -> None:
        if self._http_chunk_splits is None:
            if self.total_bytes:
                raise RuntimeError('Called begin_http_chunk_receiving when some data was already fed')
            self._http_chunk_splits = []

    def end_http_chunk_receiving(self) -> None:
        if self._http_chunk_splits is None:
            raise RuntimeError('Called end_chunk_receiving without calling begin_chunk_receiving first')
        pos = self._http_chunk_splits[-1] if self._http_chunk_splits else 0
        if self.total_bytes == pos:
            return
        self._http_chunk_splits.append(self.total_bytes)
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)

    async def _wait(self, func_name: str) -> None:
        if not self._protocol.connected:
            raise RuntimeError('Connection closed.')
        if self._waiter is not None:
            raise RuntimeError('%s() called while another coroutine is already waiting for incoming data' % func_name)
        waiter = self._waiter = self._loop.create_future()
        try:
            with self._timer:
                await waiter
        finally:
            self._waiter = None

    async def readline(self) -> bytes:
        return await self.readuntil()

    async def readuntil(self, separator: bytes = b'\n') -> bytes:
        seplen = len(separator)
        if seplen == 0:
            raise ValueError('Separator should be at least one-byte string')
        if self._exception is not None:
            raise self._exception
        chunk = b''
        chunk_size = 0
        not_enough = True
        while not_enough:
            while self._buffer and not_enough:
                offset = self._buffer_offset
                ichar = self._buffer[0].find(separator, offset) + 1
                data = self._read_nowait_chunk(ichar - offset + seplen - 1 if ichar else -1)
                chunk += data
                chunk_size += len(data)
                if ichar:
                    not_enough = False
                if chunk_size > self._high_water:
                    raise ValueError('Chunk too big')
            if self._eof:
                break
            if not_enough:
                await self._wait('readuntil')
        return chunk

    async def read(self, n: int = -1) -> bytes:
        if self._exception is not None:
            raise self._exception
        if not n:
            return b''
        if n < 0:
            blocks = []
            while True:
                block = await self.readany()
                if not block:
                    break
                blocks.append(block)
            return b''.join(blocks)
        while not self._buffer and (not self._eof):
            await self._wait('read')
        return self._read_nowait(n)

    async def readany(self) -> bytes:
        if self._exception is not None:
            raise self._exception
        while not self._buffer and (not self._eof):
            await self._wait('readany')
        return self._read_nowait(-1)

    async def readchunk(self) -> Tuple[bytes, bool]:
        """Returns a tuple of (data, end_of_http_chunk).

        When chunked transfer
        encoding is used, end_of_http_chunk is a boolean indicating if the end
        of the data corresponds to the end of a HTTP chunk , otherwise it is
        always False.
        """
        while True:
            if self._exception is not None:
                raise self._exception
            while self._http_chunk_splits:
                pos = self._http_chunk_splits.pop(0)
                if pos == self._cursor:
                    return (b'', True)
                if pos > self._cursor:
                    return (self._read_nowait(pos - self._cursor), True)
                internal_logger.warning('Skipping HTTP chunk end due to data consumption beyond chunk boundary')
            if self._buffer:
                return (self._read_nowait_chunk(-1), False)
            if self._eof:
                return (b'', False)
            await self._wait('readchunk')

    async def readexactly(self, n: int) -> bytes:
        if self._exception is not None:
            raise self._exception
        blocks = []
        while n > 0:
            block = await self.read(n)
            if not block:
                partial = b''.join(blocks)
                raise asyncio.IncompleteReadError(partial, len(partial) + n)
            blocks.append(block)
            n -= len(block)
        return b''.join(blocks)

    def read_nowait(self, n: int = -1) -> bytes:
        if self._exception is not None:
            raise self._exception
        if self._waiter and (not self._waiter.done()):
            raise RuntimeError('Called while some coroutine is waiting for incoming data.')
        return self._read_nowait(n)

    def _read_nowait_chunk(self, n: int) -> bytes:
        first_buffer = self._buffer[0]
        offset = self._buffer_offset
        if n != -1 and len(first_buffer) - offset > n:
            data = first_buffer[offset:offset + n]
            self._buffer_offset += n
        elif offset:
            self._buffer.popleft()
            data = first_buffer[offset:]
            self._buffer_offset = 0
        else:
            data = self._buffer.popleft()
        data_len = len(data)
        self._size -= data_len
        self._cursor += data_len
        chunk_splits = self._http_chunk_splits
        while chunk_splits and chunk_splits[0] < self._cursor:
            chunk_splits.pop(0)
        if self._size < self._low_water and self._protocol._reading_paused:
            self._protocol.resume_reading()
        return data

    def _read_nowait(self, n: int) -> bytes:
        """Read not more than n bytes, or whole buffer if n == -1"""
        self._timer.assert_timeout()
        chunks = []
        while self._buffer:
            chunk = self._read_nowait_chunk(n)
            chunks.append(chunk)
            if n != -1:
                n -= len(chunk)
                if n == 0:
                    break
        return b''.join(chunks) if chunks else b''

class EmptyStreamReader(StreamReader):
    __slots__ = ('_read_eof_chunk',)

    def __init__(self) -> None:
        self._read_eof_chunk = False
        self.total_bytes = 0

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def exception(self) -> None:
        return None

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        pass

    def on_eof(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            internal_logger.exception('Exception in eof callback')

    def feed_eof(self) -> None:
        pass

    def is_eof(self) -> bool:
        return True

    def at_eof(self) -> bool:
        return True

    async def wait_eof(self) -> None:
        return

    def feed_data(self, data: bytes) -> None:
        pass

    async def readline(self) -> bytes:
        return b''

    async def read(self, n: int = -1) -> bytes:
        return b''

    async def readany(self) -> bytes:
        return b''

    async def readchunk(self) -> Tuple[bytes, bool]:
        if not self._read_eof_chunk:
            self._read_eof_chunk = True
            return (b'', False)
        return (b'', True)

    async def readexactly(self, n: int) -> bytes:
        raise asyncio.IncompleteReadError(b'', n)

    def read_nowait(self, n: int = -1) -> bytes:
        return b''
EMPTY_PAYLOAD = EmptyStreamReader()

class DataQueue(Generic[_T]):
    """DataQueue is a general-purpose blocking queue with one reader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._eof = False
        self._waiter: Optional[asyncio.Future[None]] = None
        self._exception: Optional[Exception] = None
        self._buffer: Deque[_T] = collections.deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_eof(self) -> bool:
        return self._eof

    def at_eof(self) -> bool:
        return self._eof and (not self._buffer)

    def exception(self) -> Optional[Exception]:
        return self._exception

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        self._eof = True
        self._exception = exc
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_exception(waiter, exc, exc_cause)

    def feed_data(self, data: _T) -> None:
        self._buffer.append(data)
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_result(waiter, None)

    def feed_eof(self) -> None:
        self._e