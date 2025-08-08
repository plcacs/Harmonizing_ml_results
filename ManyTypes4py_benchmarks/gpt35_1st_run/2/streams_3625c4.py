import asyncio
import os
import reprlib
import typing
import weakref
from asyncio import CancelledError
from contextvars import ContextVar
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Mapping, MutableSequence, NamedTuple, Optional, Sequence, Set, Tuple, Union, cast
from mode import Seconds, Service, get_logger, shortlabel, want_seconds
from mode.utils.aiter import aenumerate, aiter
from mode.utils.futures import current_task, maybe_async, notify
from mode.utils.queues import ThrowableQueue
from mode.utils.typing import Deque
from mode.utils.types.trees import NodeT
from . import joins
from .exceptions import ImproperlyConfigured, Skip
from .types import AppT, ConsumerT, EventT, K, ModelArg, ModelT, TP, TopicT
from .types.joins import JoinT
from .types.models import FieldDescriptorT
from .types.serializers import SchemaT
from .types.streams import GroupByKeyArg, JoinableT, Processor, StreamT, T, T_co, T_contra
from .types.topics import ChannelT
from .types.tuples import Message

NO_CYTHON: bool = bool(os.environ.get('NO_CYTHON', False))
if not NO_CYTHON:
    try:
        from ._cython.streams import StreamIterator as _CStreamIterator
    except ImportError:
        _CStreamIterator = None
else:
    _CStreamIterator = None

__all__: List[str] = ['Stream', 'current_event']
logger: Any = get_logger(__name__)

if typing.TYPE_CHECKING:
    _current_event: ContextVar = ContextVar('current_event')

def current_event() -> Optional[EventT]:
    eventref = _current_event.get(None)
    return eventref() if eventref is not None else None

async def maybe_forward(value: Union[EventT, Any], channel: ChannelT) -> Any:
    if isinstance(value, EventT):
        await value.forward(channel)
    else:
        await channel.send(value=value)
    return value

class _LinkedListDirection(NamedTuple):
    pass

_LinkedListDirectionFwd: _LinkedListDirection = _LinkedListDirection('_next', lambda n: n._next)
_LinkedListDirectionBwd: _LinkedListDirection = _LinkedListDirection('_prev', lambda n: n._prev)

class Stream(StreamT[T_co], Service):
    logger: Any = logger
    mundane_level: str = 'debug'
    events_total: int = 0
    _anext_started: bool = False
    _passive: bool = False
    _finalized: bool = False

    def __init__(self, channel: ChannelT, *, app: AppT, processors: Optional[List[Processor]] = None, combined: Optional[List[JoinableT]] = None, on_start: Optional[Callable] = None, join_strategy: Optional[JoinT] = None, beacon: Any = None, concurrency_index: Any = None, prev: Any = None, active_partitions: Any = None, enable_acks: bool = True, prefix: str = '', loop: Any = None) -> None:
        Service.__init__(self, loop=loop, beacon=beacon)
        self.app: AppT = app
        self.channel: ChannelT = channel
        self.outbox: Any = self.app.FlowControlQueue(maxsize=self.app.conf.stream_buffer_maxsize, loop=self.loop, clear_on_resume=True)
        self._passive_started: asyncio.Event = asyncio.Event(loop=self.loop)
        self.join_strategy: Optional[JoinT] = join_strategy
        self.combined: List[JoinableT] = combined if combined is not None else []
        self.concurrency_index: Any = concurrency_index
        self._prev: Any = prev
        self.active_partitions: Any = active_partitions
        self.enable_acks: bool = enable_acks
        self.prefix: str = prefix
        self._processors: List[Processor] = list(processors) if processors else []
        self._on_start: Optional[Callable] = on_start
        task: Any = current_task(loop=self.loop)
        if task is not None:
            self.task_owner: Any = task
        self._on_stream_event_in: Any = self.app.sensors.on_stream_event_in
        self._on_stream_event_out: Any = self.app.sensors.on_stream_event_out
        self._on_message_in: Any = self.app.sensors.on_message_in
        self._on_message_out: Any = self.app.sensors.on_message_out
        self._skipped_value: Any = object()

    def get_active_stream(self) -> List['Stream']:
        return list(self._iter_ll_forwards())[-1]

    def get_root_stream(self) -> List['Stream']:
        return list(self._iter_ll_backwards())[-1]

    def _iter_ll_forwards(self) -> Iterator['Stream']:
        return self._iter_ll(_LinkedListDirectionFwd)

    def _iter_ll_backwards(self) -> Iterator['Stream']:
        return self._iter_ll(_LinkedListDirectionBwd)

    def _iter_ll(self, dir_: _LinkedListDirection) -> Iterator['Stream']:
        node: 'Stream' = self
        seen: Set['Stream'] = set()
        while node:
            if node in seen:
                raise RuntimeError('Loop in Stream.{dir_.attr}: Call support!')
            seen.add(node)
            yield node
            node = dir_.getter(node)

    def add_processor(self, processor: Processor) -> None:
        self._processors.append(processor)

    def info(self) -> Dict[str, Any]:
        return {'app': self.app, 'channel': self.channel, 'processors': self._processors, 'on_start': self._on_start, 'loop': self.loop, 'combined': self.combined, 'beacon': self.beacon, 'concurrency_index': self.concurrency_index, 'prev': self._prev, 'active_partitions': self.active_partitions}

    def clone(self, **kwargs) -> 'Stream':
        return self.__class__(**{**self.info(), **kwargs})

    def _chain(self, **kwargs) -> 'Stream':
        assert not self._finalized
        self._next = new_stream = self.clone(on_start=self.maybe_start, prev=self, processors=list(self._processors), **kwargs)
        self._processors.clear()
        return new_stream

    def noack(self) -> 'Stream':
        self._next = new_stream = self.clone(enable_acks=False)
        return new_stream

    async def items(self) -> AsyncIterator[Tuple[K, T_co]]:
        async for event in self.events():
            yield (event.key, cast(T_co, event.value))

    async def events(self) -> AsyncIterator[EventT]:
        async for _ in self:
            if self.current_event is not None:
                yield self.current_event

    async def take(self, max_: int, within: Union[int, float]) -> AsyncIterator[List[T_co]]:
        buffer: List[T_co] = []
        events: List[EventT] = []
        buffer_add = buffer.append
        event_add = events.append
        buffer_size = buffer.__len__
        buffer_full = asyncio.Event(loop=self.loop)
        buffer_consumed = asyncio.Event(loop=self.loop)
        timeout = want_seconds(within) if within else None
        stream_enable_acks = self.enable_acks
        buffer_consuming = None
        channel_it = aiter(self.channel)

        async def add_to_buffer(value: T_co) -> T_co:
            try:
                nonlocal buffer_consuming
                if buffer_consuming is not None:
                    try:
                        await buffer_consuming
                    finally:
                        buffer_consuming = None
                buffer_add(cast(T_co, value))
                event = self.current_event
                if event is None:
                    raise RuntimeError('Take buffer found current_event is None')
                event_add(event)
                if buffer_size() >= max_:
                    buffer_full.set()
                    buffer_consumed.clear()
                    await self.wait(buffer_consumed)
            except CancelledError:
                raise
            except Exception as exc:
                self.log.exception('Error adding to take buffer: %r', exc)
                await self.crash(exc)
            return value
        self.enable_acks = False
        self.add_processor(add_to_buffer)
        self._enable_passive(cast(ChannelT, channel_it)
