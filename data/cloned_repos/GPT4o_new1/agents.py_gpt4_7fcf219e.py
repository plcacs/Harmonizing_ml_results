import abc
import asyncio
import typing
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

from mode import ServiceT, SupervisorStrategyT
from mode.utils.collections import ManagedUserDict

from .codecs import CodecArg
from .core import HeadersArg, K, V
from .events import EventT
from .models import ModelArg
from .serializers import SchemaT
from .streams import StreamT
from .topics import ChannelT
from .tuples import Message, RecordMetadata, TP

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
else:
    class _AppT: ...          # noqa

__all__ = [
    'AgentErrorHandler',
    'AgentFun',
    'ActorT',
    'ActorRefT',
    'AgentManagerT',
    'AgentT',
    'AgentTestWrapperT',
    'AsyncIterableActorT',
    'AwaitableActorT',
    'ReplyToArg',
    'SinkT',
]

_T = TypeVar('_T')
AgentErrorHandler = Callable[['AgentT', BaseException], Awaitable[None]]
AgentFun = Callable[
    [StreamT[_T]],
    Union[Coroutine[Any, Any, None], Awaitable[None], AsyncIterable[_T]],
]

SinkT = Union['AgentT', ChannelT, Callable[[Any], Union[Awaitable[None], None]]]

ReplyToArg = Union['AgentT', ChannelT, str]


class ActorT(ServiceT, Generic[_T]):

    agent: 'AgentT'
    stream: StreamT
    it: _T
    actor_task: Optional[asyncio.Task[None]]
    active_partitions: Optional[Set[TP]]

    index: Optional[int] = None

    @abc.abstractmethod
    def __init__(self, agent: 'AgentT', stream: StreamT, it: _T,
                 active_partitions: Optional[Set[TP]] = None,
                 **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def cancel(self) -> None:
        ...

    @abc.abstractmethod
    async def on_isolated_partition_revoked(self, tp: TP) -> None:
        ...

    @abc.abstractmethod
    async def on_isolated_partition_assigned(self, tp: TP) -> None:
        ...

    @abc.abstractmethod
    def traceback(self) -> str:
        ...


class AsyncIterableActorT(ActorT[AsyncIterable[_T]], AsyncIterable[_T]):
    """Used for agent function that yields."""


class AwaitableActorT(ActorT[Awaitable[None]], Awaitable[None]):
    """Used for agent function that do not yield."""


ActorRefT = ActorT[Union[AsyncIterable[Any], Awaitable[None]]]


class AgentT(ServiceT, Generic[_T]):

    name: str
    app: _AppT
    concurrency: int
    help: str
    supervisor_strategy: Optional[Type[SupervisorStrategyT]]
    isolated_partitions: bool

    @abc.abstractmethod
    def __init__(self,
                 fun: AgentFun,
                 *,
                 name: Optional[str] = None,
                 app: Optional[_AppT] = None,
                 channel: Optional[Union[str, ChannelT]] = None,
                 concurrency: int = 1,
                 sink: Optional[Iterable[SinkT]] = None,
                 on_error: Optional[AgentErrorHandler] = None,
                 supervisor_strategy: Optional[Type[SupervisorStrategyT]] = None,
                 help: Optional[str] = None,
                 schema: Optional[SchemaT] = None,
                 key_type: Optional[ModelArg] = None,
                 value_type: Optional[ModelArg] = None,
                 isolated_partitions: bool = False,
                 **kwargs: Any) -> None:
        self.fun: AgentFun = fun

    @abc.abstractmethod
    def actor_tracebacks(self) -> List[str]:
        ...

    @abc.abstractmethod
    def __call__(self, *,
                 index: Optional[int] = None,
                 active_partitions: Optional[Set[TP]] = None,
                 stream: Optional[StreamT] = None,
                 channel: Optional[ChannelT] = None) -> ActorRefT:
        ...

    @abc.abstractmethod
    def test_context(self,
                     channel: Optional[ChannelT] = None,
                     supervisor_strategy: Optional[SupervisorStrategyT] = None,
                     **kwargs: Any) -> 'AgentTestWrapperT':
        ...

    @abc.abstractmethod
    def add_sink(self, sink: SinkT) -> None:
        ...

    @abc.abstractmethod
    def stream(self, **kwargs: Any) -> StreamT:
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def cast(self,
                   value: Optional[V] = None,
                   *,
                   key: Optional[K] = None,
                   partition: Optional[int] = None,
                   timestamp: Optional[float] = None,
                   headers: Optional[HeadersArg] = None) -> None:
        ...

    @abc.abstractmethod
    async def ask(self,
                  value: Optional[V] = None,
                  *,
                  key: Optional[K] = None,
                  partition: Optional[int] = None,
                  timestamp: Optional[float] = None,
                  headers: Optional[HeadersArg] = None,
                  reply_to: Optional[ReplyToArg] = None,
                  correlation_id: Optional[str] = None) -> Any:
        ...

    @abc.abstractmethod
    async def send(self,
                   *,
                   key: Optional[K] = None,
                   value: Optional[V] = None,
                   partition: Optional[int] = None,
                   timestamp: Optional[float] = None,
                   headers: Optional[HeadersArg] = None,
                   key_serializer: Optional[CodecArg] = None,
                   value_serializer: Optional[CodecArg] = None,
                   reply_to: Optional[ReplyToArg] = None,
                   correlation_id: Optional[str] = None) -> Awaitable[RecordMetadata]:
        ...

    @abc.abstractmethod
    @no_type_check
    async def map(self,
                  values: Union[AsyncIterable[_T], Iterable[_T]],
                  key: Optional[K] = None,
                  reply_to: Optional[ReplyToArg] = None) -> AsyncIterator[_T]:
        ...

    @abc.abstractmethod
    @no_type_check
    async def kvmap(
            self,
            items: Union[AsyncIterable[Tuple[K, V]], Iterable[Tuple[K, V]]],
            reply_to: Optional[ReplyToArg] = None) -> AsyncIterator[str]:
        ...

    @abc.abstractmethod
    async def join(self,
                   values: Union[AsyncIterable[V], Iterable[V]],
                   key: Optional[K] = None,
                   reply_to: Optional[ReplyToArg] = None) -> List[Any]:
        ...

    @abc.abstractmethod
    async def kvjoin(
            self,
            items: Union[AsyncIterable[Tuple[K, V]], Iterable[Tuple[K, V]]],
            reply_to: Optional[ReplyToArg] = None) -> List[Any]:
        ...

    @abc.abstractmethod
    def info(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def clone(self, *, cls: Optional[Type['AgentT']] = None, **kwargs: Any) -> 'AgentT':
        ...

    @abc.abstractmethod
    def get_topic_names(self) -> Iterable[str]:
        ...

    @property
    @abc.abstractmethod
    def channel(self) -> ChannelT:
        ...

    @channel.setter
    def channel(self, channel: ChannelT) -> None:
        ...

    @property
    @abc.abstractmethod
    def channel_iterator(self) -> AsyncIterator[Any]:
        ...

    @channel_iterator.setter
    def channel_iterator(self, channel: AsyncIterator[Any]) -> None:
        ...

    @abc.abstractmethod
    def _agent_label(self, name_suffix: str = '') -> str:
        ...


class AgentManagerT(ServiceT, ManagedUserDict[str, AgentT]):

    app: _AppT

    @abc.abstractmethod
    async def wait_until_agents_started(self) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self,
                           revoked: Set[TP],
                           newly_assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    def actor_tracebacks(self) -> Mapping[str, List[str]]:
        ...

    @abc.abstractmethod
    def human_tracebacks(self) -> str:
        ...


class AgentTestWrapperT(AgentT, AsyncIterable[Any]):

    new_value_processed: asyncio.Condition
    original_channel: ChannelT
    results: MutableMapping[int, Any]
    sent_offset: int = 0
    processed_offset: int = 0

    @abc.abstractmethod
    def __init__(self,
                 *args: Any,
                 original_channel: Optional[ChannelT] = None,
                 **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def put(self,
                  value: Optional[V] = None,
                  key: Optional[K] = None,
                  partition: Optional[int] = None,
                  timestamp: Optional[float] = None,
                  headers: Optional[HeadersArg] = None,
                  key_serializer: Optional[CodecArg] = None,
                  value_serializer: Optional[CodecArg] = None,
                  *,
                  reply_to: Optional[ReplyToArg] = None,
                  correlation_id: Optional[str] = None,
                  wait: bool = True) -> EventT:
        ...

    @abc.abstractmethod
    def to_message(self,
                   key: K,
                   value: V,
                   *,
                   partition: int = 0,
                   offset: int = 0,
                   timestamp: Optional[float] = None,
                   timestamp_type: int = 0,
                   headers: Optional[HeadersArg] = None) -> Message:
        ...

    @abc.abstractmethod
    async def throw(self, exc: BaseException) -> None:
        ...
