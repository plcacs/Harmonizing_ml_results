from asyncio import CancelledError, create_future, sleep
from contextvars import ContextVar
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Mapping, MutableSequence, NamedTuple, Optional, Sequence, Set, Tuple, Union
from mode import Seconds, Service, get_logger, shortlabel, want_seconds
from mode.utils.aiter import aenumerate, aiter
from mode.utils.futures import current_task, maybe_async, notify
from mode.utils.queues import ThrowableQueue
from mode.utils.typing import Deque
from mode.utils.types.trees import NodeT

class Stream(StreamT[T_co], Service):
    logger: Logger
    mundane_level: str
    events_total: int
    _anext_started: bool
    _passive: bool
    _finalized: bool

    def __init__(self, channel: ChannelT, app: AppT, processors: Optional[List[Callable[[T_co], T_co]]] = None, 
                 combined: Optional[List['Stream']] = None, on_start: Optional[Callable[[], None]] = None, 
                 join_strategy: Optional[JoinT] = None, beacon: Optional[Seconds] = None, concurrency_index: Optional[int] = None, 
                 prev: Optional['Stream'] = None, active_partitions: Optional[List[int]] = None, enable_acks: bool = True, 
                 prefix: str = '', loop: Optional[asyncio.BaseEventLoop] = None) -> None:
        ...

    def get_active_stream(self) -> 'Stream':
        ...

    def get_root_stream(self) -> 'Stream':
        ...

    def add_processor(self, processor: Callable[[T_co], T_co]) -> None:
        ...

    def info(self) -> Dict[str, Any]:
        ...

    def clone(self, **kwargs: Any) -> 'Stream':
        ...

    def noack(self) -> 'Stream':
        ...

    async def items(self) -> AsyncIterable[Tuple[K, T_co]]:
        ...

    async def events(self) -> AsyncIterable[EventT]:
        ...

    async def take(self, max_: int, within: Seconds) -> AsyncIterable[T_co]:
        ...

    def through(self, channel: ChannelT) -> 'Stream':
        ...

    def group_by(self, key: Union[FieldDescriptorT, Callable[[T_co], K], str], name: Optional[str] = None, topic: Optional[TopicT] = None, 
                 partitions: Optional[List[int]] = None) -> 'Stream':
        ...

    def filter(self, fun: Callable[[T_co], bool]) -> 'Stream':
        ...

    async def _format_key(self, key: Union[FieldDescriptorT, Callable[[T_co], K], str], value: T_co) -> K:
        ...

    def derive_topic(self, name: str, schema: Optional[SchemaT] = None, key_type: Optional[type] = None, value_type: Optional[type] = None, 
                     prefix: str = '', suffix: str = '') -> TopicT:
        ...

    async def on_merge(self, value: T_co) -> T_co:
        ...

    async def on_start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def on_stop(self) -> None:
        ...

    def __iter__(self) -> Iterator[T_co]:
        ...

    async def __anext__(self) -> T_co:
        ...

    async def ack(self, event: EventT) -> None:
        ...
