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
        ...

    def enumerate(self, start: int = 0) -> AsyncIterator[Tuple[int, T_co]]:
        return aenumerate(self, start)

    def through(self, channel: Union[str, ChannelT]) -> 'Stream':
        ...

    def _enable_passive(self, channel: ChannelT, declare: bool = False) -> None:
        ...

    async def _passive_drainer(self, channel: ChannelT, declare: bool = False) -> None:
        ...

    def _channel_stop_iteration(self, channel: ChannelT) -> None:
        ...

    def echo(self, *channels: ChannelT) -> 'Stream':
        ...

    def group_by(self, key: Union[FieldDescriptorT, Callable[[T], Any], Callable[[T], Any]], *, name: str = None, topic: Optional[TopicT] = None, partitions: Optional[int] = None) -> 'Stream':
        ...

    def filter(self, fun: Callable[[T], Union[bool, Awaitable[bool]]]) -> 'Stream':
        ...

    async def _format_key(self, key: Union[FieldDescriptorT, Callable[[T], Any]], value: T) -> Any:
        ...

    def derive_topic(self, name: str, *, schema: Optional[SchemaT] = None, key_type: Any = None, value_type: Any = None, prefix: str = '', suffix: str = '') -> TopicT:
        ...

    async def throw(self, exc: Exception) -> None:
        ...

    def combine(self, *nodes: JoinableT, **kwargs) -> 'Stream':
        ...

    def contribute_to_stream(self, active: 'Stream') -> None:
        ...

    async def remove_from_stream(self, stream: 'Stream') -> None:
        ...

    def join(self, *fields: Any) -> 'Stream':
        ...

    def left_join(self, *fields: Any) -> 'Stream':
        ...

    def inner_join(self, *fields: Any) -> 'Stream':
        ...

    def outer_join(self, *fields: Any) -> 'Stream':
        ...

    def _join(self, join_strategy: JoinT) -> 'Stream':
        ...

    async def on_merge(self, value: Any = None) -> Any:
        ...

    async def on_start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def on_stop(self) -> None:
        ...

    def __iter__(self) -> 'Stream':
        ...

    def __next__(self) -> None:
        ...

    def __aiter__(self) -> AsyncIterator[T_co]:
        ...

    async def _c_aiter(self) -> None:
        ...

    async def _py_aiter(self) -> None:
        ...

    async def __anext__(self) -> None:
        ...

    async def ack(self, event: EventT) -> bool:
        ...

    def __and__(self, other: 'Stream') -> 'Stream':
        ...

    def __copy__(self) -> 'Stream':
        ...

    def _repr_info(self) -> str:
        ...

    @property
    def label(self) -> str:
        ...

    def _repr_channel(self) -> str:
        ...

    @property
    def shortlabel(self) -> str:
        ...

    def _human_channel(self) -> str:
        ...

