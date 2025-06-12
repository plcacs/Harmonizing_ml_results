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
    def __init__(self, agent, stream, it, active_partitions=None, **kwargs):
        ...

    @abc.abstractmethod
    def cancel(self):
        ...

    @abc.abstractmethod
    async def on_isolated_partition_revoked(self, tp):
        ...

    @abc.abstractmethod
    async def on_isolated_partition_assigned(self, tp):
        ...

    @abc.abstractmethod
    def traceback(self):
        ...

class AsyncIterableActorT(ActorT[AsyncIterable], AsyncIterable):
    """Used for agent function that yields."""

class AwaitableActorT(ActorT[Awaitable], Awaitable):
    """Used for agent function that do not yield."""
ActorRefT = ActorT[Union[AsyncIterable, Awaitable]]

class AgentT(ServiceT, Generic[_T]):

    @abc.abstractmethod
    def __init__(self, fun, *, name=None, app=None, channel=None, concurrency=1, sink=None, on_error=None, supervisor_strategy=None, help=None, schema=None, key_type=None, value_type=None, isolated_partitions=False, **kwargs):
        self.fun = fun

    @abc.abstractmethod
    def actor_tracebacks(self):
        ...

    @abc.abstractmethod
    def __call__(self, *, index=None, active_partitions=None, stream=None, channel=None):
        ...

    @abc.abstractmethod
    def test_context(self, channel=None, supervisor_strategy=None, **kwargs):
        ...

    @abc.abstractmethod
    def add_sink(self, sink):
        ...

    @abc.abstractmethod
    def stream(self, **kwargs):
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
    def info(self):
        ...

    @abc.abstractmethod
    def clone(self, *, cls=None, **kwargs):
        ...

    @abc.abstractmethod
    def get_topic_names(self):
        ...

    @property
    @abc.abstractmethod
    def channel(self):
        ...

    @channel.setter
    def channel(self, channel):
        ...

    @property
    @abc.abstractmethod
    def channel_iterator(self):
        ...

    @channel_iterator.setter
    def channel_iterator(self, channel):
        ...

    @abc.abstractmethod
    def _agent_label(self, name_suffix=''):
        ...

class AgentManagerT(ServiceT, ManagedUserDict[str, AgentT]):

    @abc.abstractmethod
    async def wait_until_agents_started(self):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    def actor_tracebacks(self):
        ...

    @abc.abstractmethod
    def human_tracebacks(self):
        ...

class AgentTestWrapperT(AgentT, AsyncIterable):
    sent_offset = 0
    processed_offset = 0

    @abc.abstractmethod
    def __init__(self, *args, original_channel=None, **kwargs):
        ...

    @abc.abstractmethod
    async def put(self, value=None, key=None, partition=None, timestamp=None, headers=None, key_serializer=None, value_serializer=None, *, reply_to=None, correlation_id=None, wait=True):
        ...

    @abc.abstractmethod
    def to_message(self, key, value, *, partition=0, offset=0, timestamp=None, timestamp_type=0, headers=None):
        ...

    @abc.abstractmethod
    async def throw(self, exc):
        ...