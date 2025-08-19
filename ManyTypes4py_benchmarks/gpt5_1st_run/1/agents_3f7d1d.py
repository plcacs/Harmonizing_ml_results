import abc
import asyncio
import typing
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union, no_type_check
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

    class _AppT:
        ...
__all__ = ['AgentErrorHandler', 'AgentFun', 'ActorT', 'ActorRefT', 'AgentManagerT', 'AgentT', 'AgentTestWrapperT', 'AsyncIterableActorT', 'AwaitableActorT', 'ReplyToArg', 'SinkT']
_T = TypeVar('_T')
AgentErrorHandler = Callable[['AgentT[Any]', BaseException], Awaitable[None]]
AgentFun = Callable[[StreamT[_T]], Union[Coroutine[Any, Any, None], Awaitable[None], AsyncIterable[Any]]]
SinkT = Union['AgentT[Any]', ChannelT, Callable[[Any], Union[Awaitable[Any], None]]]
ReplyToArg = Union['AgentT[Any]', ChannelT, str]

class ActorT(ServiceT, Generic[_T]):
    index: Optional[int] = None

    @abc.abstractmethod
    def __init__(self, agent: 'AgentT[Any]', stream: StreamT[Any], it: _T, active_partitions: Optional[Set[TP]] = None, **kwargs: Any) -> None:
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
    def traceback(self) -> List[str]:
        ...

class AsyncIterableActorT(ActorT[AsyncIterable[Any]], AsyncIterable[Any]):
    """Used for agent function that yields."""

class AwaitableActorT(ActorT[Awaitable[None]], Awaitable[None]):
    """Used for agent function that do not yield."""
ActorRefT = ActorT[Union[AsyncIterable[Any], Awaitable[None]]]

class AgentT(ServiceT, Generic[_T]):

    @abc.abstractmethod
    def __init__(
        self,
        fun: AgentFun[_T],
        *,
        name: Optional[str] = None,
        app: Optional[_AppT] = None,
        channel: Optional[ChannelT] = None,
        concurrency: int = 1,
        sink: Optional[SinkT] = None,
        on_error: Optional[AgentErrorHandler] = None,
        supervisor_strategy: Optional[SupervisorStrategyT] = None,
        help: Optional[str] = None,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        isolated_partitions: bool = False,
        **kwargs: Any
    ) -> None:
        self.fun: AgentFun[_T] = fun

    @abc.abstractmethod
    def actor_tracebacks(self) -> Mapping[Optional[int], List[str]]:
        ...

    @abc.abstractmethod
    def __call__(
        self,
        *,
        index: Optional[int] = None,
        active_partitions: Optional[Set[TP]] = None,
        stream: Optional[StreamT[Any]] = None,
        channel: Optional[ChannelT] = None
    ) -> ActorRefT:
        ...

    @abc.abstractmethod
    def test_context(
        self,
        channel: Optional[ChannelT] = None,
        supervisor_strategy: Optional[SupervisorStrategyT] = None,
        **kwargs: Any
    ) -> 'AgentTestWrapperT[Any]':
        ...

    @abc.abstractmethod
    def add_sink(self, sink: SinkT) -> None:
        ...

    @abc.abstractmethod
    def stream(self, **kwargs: Any) -> StreamT[Any]:
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def cast(
        self,
        value: Optional[V] = None,
        *,
        key: Optional[K] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None
    ) -> RecordMetadata:
        ...

    @abc.abstractmethod
    async def ask(
        self,
        value: Optional[V] = None,
        *,
        key: Optional[K] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        reply_to: Optional[ReplyToArg] = None,
        correlation_id: Optional[str] = None
    ) -> EventT[Any]:
        ...

    @abc.abstractmethod
    async def send(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        reply_to: Optional[ReplyToArg] = None,
        correlation_id: Optional[str] = None
    ) -> RecordMetadata:
        ...

    @abc.abstractmethod
    @no_type_check
    async def map(
        self,
        values: Iterable[V],
        key: Optional[K] = None,
        reply_to: Optional[ReplyToArg] = None
    ) -> List[RecordMetadata]:
        ...

    @abc.abstractmethod
    @no_type_check
    async def kvmap(
        self,
        items: Iterable[Tuple[K, V]],
        reply_to: Optional[ReplyToArg] = None
    ) -> List[RecordMetadata]:
        ...

    @abc.abstractmethod
    async def join(
        self,
        values: Iterable[V],
        key: Optional[K] = None,
        reply_to: Optional[ReplyToArg] = None
    ) -> List[EventT[Any]]:
        ...

    @abc.abstractmethod
    async def kvjoin(
        self,
        items: Iterable[Tuple[K, V]],
        reply_to: Optional[ReplyToArg] = None
    ) -> List[EventT[Any]]:
        ...

    @abc.abstractmethod
    def info(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def clone(self, *, cls: Optional[Type['AgentT[Any]']] = None, **kwargs: Any) -> 'AgentT[Any]':
        ...

    @abc.abstractmethod
    def get_topic_names(self) -> Set[str]:
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
    def channel_iterator(self) -> AsyncIterator[EventT[Any]]:
        ...

    @channel_iterator.setter
    def channel_iterator(self, channel: AsyncIterator[EventT[Any]]) -> None:
        ...

    @abc.abstractmethod
    def _agent_label(self, name_suffix: str = '') -> str:
        ...

class AgentManagerT(ServiceT, ManagedUserDict[str, 'AgentT[Any]']):

    @abc.abstractmethod
    async def wait_until_agents_started(self) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    def actor_tracebacks(self) -> Mapping[str, Mapping[Optional[int], List[str]]]:
        ...

    @abc.abstractmethod
    def human_tracebacks(self) -> str:
        ...

class AgentTestWrapperT(AgentT[_T], AsyncIterable[Any]):
    sent_offset: int = 0
    processed_offset: int = 0

    @abc.abstractmethod
    def __init__(self, *args: Any, original_channel: Optional[ChannelT] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def put(
        self,
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
        wait: bool = True
    ) -> Optional[RecordMetadata]:
        ...

    @abc.abstractmethod
    def to_message(
        self,
        key: Optional[K],
        value: Optional[V],
        *,
        partition: int = 0,
        offset: int = 0,
        timestamp: Optional[float] = None,
        timestamp_type: int = 0,
        headers: Optional[HeadersArg] = None
    ) -> Message:
        ...

    @abc.abstractmethod
    async def throw(self, exc: BaseException) -> None:
        ...