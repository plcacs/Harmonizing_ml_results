import asyncio
import collections
import warnings
from asyncio import IncompleteReadError
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Final,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

class EofStream(Exception):
    ...

class AsyncStreamIterator(Generic[_T]):
    __slots__ = ('read_func',)
    read_func: Callable[[], Awaitable[_T]]

    def __init__(self, read_func: Callable[[], Awaitable[_T]]) -> None:
        ...

    def __aiter__(self) -> AsyncStreamIterator[_T]:
        ...

    async def __anext__(self) -> _T:
        ...

class ChunkTupleAsyncStreamIterator:
    __slots__ = ('_stream',)
    _stream: StreamReader

    def __init__(self, stream: StreamReader) -> None:
        ...

    def __aiter__(self) -> ChunkTupleAsyncStreamIterator:
        ...

    async def __anext__(self) -> Tuple[bytes, bool]:
        ...

class AsyncStreamReaderMixin:
    __slots__ = ()

    def __aiter__(self) -> AsyncStreamIterator[bytes]:
        ...

    def iter_chunked(self, n: int) -> AsyncStreamIterator[bytes]:
        ...

    def iter_any(self) -> AsyncStreamIterator[bytes]:
        ...

    def iter_chunks(self) -> ChunkTupleAsyncStreamIterator:
        ...

class StreamReader(AsyncStreamReaderMixin):
    __slots__ = (
        '_protocol',
        '_low_water',
        '_high_water',
        '_loop',
        '_size',
        '_cursor',
        '_http_chunk_splits',
        '_buffer',
        '_buffer_offset',
        '_eof',
        '_waiter',
        '_eof_waiter',
        '_exception',
        '_timer',
        '_eof_callbacks',
        '_eof_counter',
        'total_bytes',
    )

    _protocol: BaseProtocol
    _low_water: int
    _high_water: int
    _loop: asyncio.AbstractEventLoop
    _size: int
    _cursor: int
    _http_chunk_splits: Optional[List[int]]
    _buffer: Deque[bytes]
    _buffer_offset: int
    _eof: bool
    _waiter: Optional[asyncio.Future[None]]
    _eof_waiter: Optional[asyncio.Future[None]]
    _exception: Optional[Exception]
    _timer: BaseTimerContext
    _eof_callbacks: List[Callable[[], None]]
    _eof_counter: int
    total_bytes: int

    def __init__(self, protocol: BaseProtocol, limit: int, *, timer: Optional[BaseTimerContext] = None, loop: asyncio.AbstractEventLoop) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def get_read_buffer_limits(self) -> Tuple[int, int]:
        ...

    def exception(self) -> Optional[Exception]:
        ...

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        ...

    def on_eof(self, callback: Callable[[], None]) -> None:
        ...

    def feed_eof(self) -> None:
        ...

    def is_eof(self) -> bool:
        ...

    def at_eof(self) -> bool:
        ...

    async def wait_eof(self) -> None:
        ...

    def unread_data(self, data: bytes) -> None:
        ...

    def feed_data(self, data: bytes) -> None:
        ...

    def begin_http_chunk_receiving(self) -> None:
        ...

    def end_http_chunk_receiving(self) -> None:
        ...

    async def _wait(self, func_name: str) -> None:
        ...

    async def readline(self) -> bytes:
        ...

    async def readuntil(self, separator: bytes = b'\n') -> bytes:
        ...

    async def read(self, n: int = -1) -> bytes:
        ...

    async def readany(self) -> bytes:
        ...

    async def readchunk(self) -> Tuple[bytes, bool]:
        ...

    async def readexactly(self, n: int) -> bytes:
        ...

    def read_nowait(self, n: int = -1) -> bytes:
        ...

    def _read_nowait_chunk(self, n: int) -> bytes:
        ...

    def _read_nowait(self, n: int) -> bytes:
        ...

class EmptyStreamReader(StreamReader):
    __slots__ = ('_read_eof_chunk',)
    _read_eof_chunk: bool

    def __init__(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def exception(self) -> None:
        ...

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        ...

    def on_eof(self, callback: Callable[[], None]) -> None:
        ...

    def feed_eof(self) -> None:
        ...

    def is_eof(self) -> bool:
        ...

    def at_eof(self) -> bool:
        ...

    async def wait_eof(self) -> None:
        ...

    def feed_data(self, data: bytes) -> None:
        ...

    async def readline(self) -> bytes:
        ...

    async def read(self, n: int = -1) -> bytes:
        ...

    async def readany(self) -> bytes:
        ...

    async def readchunk(self) -> Tuple[bytes, bool]:
        ...

    async def readexactly(self, n: int) -> bytes:
        ...

    def read_nowait(self, n: int = -1) -> bytes:
        ...

EMPTY_PAYLOAD = EmptyStreamReader()

class DataQueue(Generic[_T]):
    __slots__ = (
        '_loop',
        '_eof',
        '_waiter',
        '_exception',
        '_buffer',
    )

    _loop: asyncio.AbstractEventLoop
    _eof: bool
    _waiter: Optional[asyncio.Future[None]]
    _exception: Optional[Exception]
    _buffer: Deque[_T]

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        ...

    def __len__(self) -> int:
        ...

    def is_eof(self) -> bool:
        ...

    def at_eof(self) -> bool:
        ...

    def exception(self) -> Optional[Exception]:
        ...

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        ...

    def feed_data(self, data: _T) -> None:
        ...

    def feed_eof(self) -> None:
        ...

    async def read(self) -> _T:
        ...

    def __aiter__(self) -> AsyncStreamIterator[_T]:
        ...