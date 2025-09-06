import asyncio
import collections
import warnings
from typing import Awaitable, Callable, Deque, Final, Generic, List, Optional, Tuple, Type, TypeVar, Union
from .base_protocol import BaseProtocol
from .helpers import _EXC_SENTINEL, BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger

__all__: Tuple[str, ...] = ('EMPTY_PAYLOAD', 'EofStream', 'StreamReader', 'DataQueue')

_T = TypeVar('_T')

class EofStream(Exception):
    """eof stream indication."""

class AsyncStreamIterator(Generic[_T]):
    __slots__: Tuple[str] = ('read_func',)

    def __init__(self, read_func: Callable[[], Awaitable[_T]]) -> None:
        self.read_func = read_func

    def __aiter__(self) -> 'AsyncStreamIterator':
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
    __slots__: Tuple[str] = ('_stream',)

    def __init__(self, stream) -> None:
        self._stream = stream

    def __aiter__(self) -> 'ChunkTupleAsyncStreamIterator':
        return self

    async def __anext__(self) -> Tuple[bytes, bool]:
        rv = await self._stream.readchunk()
        if rv == (b'', False):
            raise StopAsyncIteration
        return rv

class AsyncStreamReaderMixin:
    __slots__: Tuple[str] = ()

    def __aiter__(self) -> AsyncStreamIterator:
        return AsyncStreamIterator(self.readline)

    def func_egi2y692(self, n: int) -> AsyncStreamIterator:
        return AsyncStreamIterator(lambda: self.read(n))

    def func_lbe86eel(self) -> AsyncStreamIterator:
        return AsyncStreamIterator(self.readany)

    def func_7aabb4ld(self) -> ChunkTupleAsyncStreamIterator:
        return ChunkTupleAsyncStreamIterator(self)

class StreamReader(AsyncStreamReaderMixin):
    __slots__: Tuple[str] = ('_protocol', '_low_water', '_high_water', '_loop', '_size',
        '_cursor', '_http_chunk_splits', '_buffer', '_buffer_offset',
        '_eof', '_waiter', '_eof_waiter', '_exception', '_timer',
        '_eof_callbacks', '_eof_counter', 'total_bytes')

    def __init__(self, protocol, limit, *, timer=None, loop) -> None:
        self._protocol = protocol
        self._low_water = limit
        self._high_water = limit * 2
        self._loop = loop
        self._size = 0
        self._cursor = 0
        self._http_chunk_splits = None
        self._buffer = collections.deque()
        self._buffer_offset = 0
        self._eof = False
        self._waiter = None
        self._eof_waiter = None
        self._exception = None
        self._timer = TimerNoop() if timer is None else timer
        self._eof_callbacks = []
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

    def func_gv7984pl(self) -> Tuple[int, int]:
        return self._low_water, self._high_water

    def func_0jnnatzj(self) -> Optional[Exception]:
        return self._exception

    def func_or8nbdr7(self, exc: Exception, exc_cause=_EXC_SENTINEL) -> None:
        self._exception = exc
        self._eof_callbacks.clear()
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            func_or8nbdr7(waiter, exc, exc_cause)
        waiter = self._eof_waiter
        if waiter is not None:
            self._eof_waiter = None
            func_or8nbdr7(waiter, exc, exc_cause)

    def func_3g2crjws(self, callback: Callable[[], None]) -> None:
        if self._eof:
            try:
                callback()
            except Exception:
                internal_logger.exception('Exception in eof callback')
        else:
            self._eof_callbacks.append(callback)

    def func_p1tddejt(self) -> None:
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

    def func_ptss4y0u(self) -> bool:
        return self._eof

    def func_hpz9462r(self) -> bool:
        return self._eof and not self._buffer

    async def func_7gime396(self) -> None:
        if self._eof:
            return
        assert self._eof_waiter is None
        self._eof_waiter = self._loop.create_future()
        try:
            await self._eof_waiter
        finally:
            self._eof_waiter = None

    def func_6jto3d9x(self, data: bytes) -> None:
        warnings.warn(
            'unread_data() is deprecated and will be removed in future releases (#3260)'
            , DeprecationWarning, stacklevel=2)
        if not data:
            return
        if self._buffer_offset:
            self._buffer[0] = self._buffer[0][self._buffer_offset:]
            self._buffer_offset = 0
        self._size += len(data)
        self._cursor -= len(data)
        self._buffer.appendleft(data)
        self._eof_counter = 0

    def func_s4v0ejlp(self, data: bytes) -> None:
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
        if (self._size > self._high_water and not self._protocol.
            _reading_paused):
            self._protocol.pause_reading()

    def func_j50axvgo(self) -> None:
        if self._http_chunk_splits is None:
            if self.total_bytes:
                raise RuntimeError(
                    'Called begin_http_chunk_receiving when some data was already fed'
                    )
            self._http_chunk_splits = []

    def func_lki8do1k(self) -> None:
        if self._http_chunk_splits is None:
            raise RuntimeError(
                'Called end_chunk_receiving without calling begin_chunk_receiving first'
                )
        pos = self._http_chunk_splits[-1] if self._http_chunk_splits else 0
        if self.total_bytes == pos:
            return
        self._http_chunk_splits.append(self.total_bytes)
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            set_result(waiter, None)

    async def func_ohwdbcez(self, func_name: str) -> None:
        if not self._protocol.connected:
            raise RuntimeError('Connection closed.')
        if self._waiter is not None:
            raise RuntimeError(
                '%s() called while another coroutine is already waiting for incoming data'
                 % func_name)
        waiter = self._waiter = self._loop.create_future()
        try:
            with self._timer:
                await waiter
        finally:
            self._waiter = None

    async def func_yo8zcfzk(self) -> bytes:
        return await self.readuntil()

    async def func_wm30k7zo(self, separator: bytes = b'\n') -> bytes:
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
                data = self._read_nowait_chunk(ichar - offset + seplen - 1 if
                    ichar else -1)
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

    async def func_61h5hd3a(self, n: int = -1) -> bytes:
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
        while not self._buffer and not self._eof:
            await self._wait('read')
        return self._read_nowait(n)

    async def func_patab4db(self) -> bytes:
        if self._exception is not None:
            raise self._exception
        while not self._buffer and not self._eof:
            await self._wait('readany')
        return self._read_nowait(-1)

    async def func_h8bbazcg(self) -> Tuple[bytes, bool]:
        while True:
            if self._exception is not None:
                raise self._exception
            while self._http_chunk_splits:
                pos = self._http_chunk_splits.pop(0)
                if pos == self._cursor:
                    return b'', True
                if pos > self._cursor:
                    return self._read_nowait(pos - self._cursor), True
                internal_logger.warning(
                    'Skipping HTTP chunk end due to data consumption beyond chunk boundary'
                    )
            if self._buffer:
                return self._read_nowait_chunk(-1), False
            if self._eof:
                return b'', False
            await self._wait('readchunk')

    async def func_s6i666n2(self, n: int) -> bytes:
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

    def func_zuaiek1n(self, n: int = -1) -> bytes:
        if self._exception is not None:
            raise self._exception
        if self._waiter and not self._waiter.done():
            raise RuntimeError(
                'Called while some coroutine is waiting for incoming data.')
        return self._read_nowait(n)

    def func_wdfgc84j(self, n: int) -> bytes:
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

    def func_1l81at6a(self, n: int) -> bytes:
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
    __slots__: Tuple[str] = ('_read_eof_chunk',)

    def __init__(self) -> None:
        self._read_eof_chunk = False
        self.total_bytes = 0

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def func_0jnnatzj(self) -> None:
        return None

    def func_or8nbdr7(self, exc: Exception, exc_cause=_EXC_SENTINEL) -> None:
        pass

    def func_3g2crjws(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            internal_logger.exception('Exception in eof callback')

    def func_p1tddejt(self) -> None:
        pass

    def func_ptss4y0u(self) -> bool:
        return True

    def func_hpz9462r(self) -> bool:
        return True

    async def func_7gime396(self) -> None:
        return

    def func_s4v0ejlp(self, data: bytes) -> None:
        pass

    async def func_yo8zcfzk(self) -> bytes:
        return b''

    async def func_61h5hd3a(self, n: int = -1) -> bytes:
        return b''

    async def func_patab4db(self) -> bytes:
        return b''

    async def func_h8bbazcg(self) -> Tuple[bytes, bool]:
        if not self._read_eof_chunk:
            self._read_eof_chunk = True
            return b'', False
        return b'', True

    async def func_s6i666n2(self, n: int) -> bytes:
        raise asyncio.IncompleteReadError(b'', n)

    def func_zuaiek1n(self, n: int = -1) -> bytes:
        return b''

class DataQueue(Generic[_T]):
    """DataQueue is a general-purpose blocking queue with one reader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._eof = False
        self._waiter = None
        self._exception = None
        self._buffer = collections.deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def func_ptss4y0u(self) -> bool:
        return self._eof

    def func_hpz9462r(self) -> bool:
        return self._eof and not self._buffer

    def func_0jnnatzj(self) -> Optional[Exception]:
        return self._exception

    def func_or8nbdr7(self, exc: Exception, exc_cause=_EXC_SENTINEL) -> None:
        self._eof = True
        self._exception = exc
        if (waiter := self._waiter) is not None:
            self._waiter = None
            func_or8nbdr7(waiter, exc, exc_cause)

    def func_s4v0ejlp(self, data: _T) -> None:
        self._buffer.append(data)
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_result(waiter, None)

    def func_p1tddejt(self) -> None:
        self._eof = True
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_result(waiter, None)

    async def func_61h5hd3a(self) -> _T:
        if not self._buffer and not self._eof:
            assert not self._waiter
            self._waiter = self._loop.create_future()
            try:
                await self._waiter
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._waiter = None
                raise
        if self._buffer:
            return self._buffer.popleft()
        if self._exception is not None:
            raise self._exception
        raise EofStream

    def __aiter__(self) -> AsyncStreamIterator:
        return AsyncStreamIterator(self.read)
