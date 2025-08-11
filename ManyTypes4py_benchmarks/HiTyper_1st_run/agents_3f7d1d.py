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
AgentErrorHandler = Callable[['AgentT', BaseException], Awaitable]
AgentFun = Callable[[StreamT[_T]], Union[Coroutine[Any, Any, None], Awaitable[None], AsyncIterable]]
SinkT = Union['AgentT', ChannelT, Callable[[Any], Union[Awaitable, None]]]
ReplyToArg = Union['AgentT', ChannelT, str]

class ActorT(ServiceT, Generic[_T]):
    index = None

    @abc.abstractmethod
    def __init__(self, agent, stream, it, active_partitions=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def cancel(self) -> None:
        ...

    @abc.abstractmethod
    async def on_isolated_partition_revoked(self, tp):
        ...

    @abc.abstractmethod
    async def on_isolated_partition_assigned(self, tp):
        ...

    @abc.abstractmethod
    def traceback(self) -> None:
        ...

class AsyncIterableActorT(ActorT[AsyncIterable], AsyncIterable):
    """Used for agent function that yields."""

class AwaitableActorT(ActorT[Awaitable], Awaitable):
    """Used for agent function that do not yield."""
ActorRefT = ActorT[Union[AsyncIterable, Awaitable]]

class AgentT(ServiceT, Generic[_T]):

    @abc.abstractmethod
    def __init__(self, fun, *, name=None, app=None, channel=None, concurrency=1, sink=None, on_error=None, supervisor_strategy=None, help=None, schema=None, key_type=None, value_type=None, isolated_partitions=False, **kwargs) -> None:
        self.fun = fun

    @abc.abstractmethod
    def actor_tracebacks(self) -> None:
        ...

    @abc.abstractmethod
    def __call__(self, *, index: Union[None, int, set[tuples.TP], streams.StreamT]=None, active_partitions: Union[None, int, set[tuples.TP], streams.StreamT]=None, stream: Union[None, int, set[tuples.TP], streams.StreamT]=None, channel: Union[None, int, set[tuples.TP], streams.StreamT]=None) -> None:
        ...

    @abc.abstractmethod
    def test_context(self, channel: Union[None, collections.abc.Awaitable, typing.Optional, typing.Callable]=None, supervisor_strategy: Union[None, collections.abc.Awaitable, typing.Optional, typing.Callable]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def add_sink(self, sink: Union[list, bytes, str]) -> None:
        ...

    @abc.abstractmethod
    def stream(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned):
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked):
        ...

    @abc.abstractmethod
    async def cast(self, value=None, *, key=None, partition=None, timestamp=None, headers=None):
        ...

    @abc.abstractmethod
    async def ask(self, value=None, *, key=None, partition=None, timestamp=None, headers=None, reply_to=None, correlation_id=None):
        ...

    @abc.abstractmethod
    async def send(self, *, key=None, value=None, partition=None, timestamp=None, headers=None, key_serializer=None, value_serializer=None, reply_to=None, correlation_id=None):
        ...

    @abc.abstractmethod
    @no_type_check
    async def map(self, values, key=None, reply_to=None):
        ...

    @abc.abstractmethod
    @no_type_check
    async def kvmap(self, items, reply_to=None):
        ...

    @abc.abstractmethod
    async def join(self, values, key=None, reply_to=None):
        ...

    @abc.abstractmethod
    async def kvjoin(self, items, reply_to=None):
        ...

    @abc.abstractmethod
    def info(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, *, cls: Union[None, typing.Type]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def get_topic_names(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def channel(self) -> None:
        ...

    @channel.setter
    def channel(self, channel) -> None:
        ...

    @property
    @abc.abstractmethod
    def channel_iterator(self) -> None:
        ...

    @channel_iterator.setter
    def channel_iterator(self, channel) -> None:
        ...

    @abc.abstractmethod
    def _agent_label(self, name_suffix: typing.Text='') -> None:
        ...

class AgentManagerT(ServiceT, ManagedUserDict[str, AgentT]):

    @abc.abstractmethod
    async def wait_until_agents_started(self):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    def actor_tracebacks(self) -> None:
        ...

    @abc.abstractmethod
    def human_tracebacks(self) -> None:
        ...

class AgentTestWrapperT(AgentT, AsyncIterable):
    sent_offset = 0
    processed_offset = 0

    @abc.abstractmethod
    def __init__(self, *args, original_channel: Union[None, bool, str]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    async def put(self, value=None, key=None, partition=None, timestamp=None, headers=None, key_serializer=None, value_serializer=None, *, reply_to=None, correlation_id=None, wait=True):
        ...

    @abc.abstractmethod
    def to_message(self, key: Union[int, core.K, core.V], value: Union[int, core.K, core.V], *, partition: int=0, offset: int=0, timestamp: Union[None, int, core.K, core.V]=None, timestamp_type: int=0, headers: Union[None, int, core.K, core.V]=None) -> None:
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...