import abc
import asyncio
import typing
from typing import Any, AsyncIterator, Awaitable, Generic, Optional, Set, TypeVar

from mode import Seconds
from mode.utils.futures import stampede
from mode.utils.queues import ThrowableQueue
from .codecs import CodecArg
from .core import HeadersArg, K, V
from .tuples import FutureMessage, Message, MessageSentCallback, RecordMetadata, TP

_T = TypeVar('_T')
_T_contra = TypeVar('_T_contra', contravariant=True)

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .events import EventT as _EventT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
    from .streams import StreamT as _StreamT
else:

    class _AppT:
        ...

    class _EventT(Generic[_T]):
        ...

    class _ModelArg:
        ...

    class _SchemaT:
        ...

    class _StreamT:
        ...


class ChannelT(AsyncIterator['_EventT[_T]'], Generic[_T], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(
        self,
        app: _AppT,
        *,
        schema: Optional['_SchemaT'] = None,
        key_type: Optional[K] = None,
        value_type: Optional[V] = None,
        is_iterator: bool = False,
        queue: Optional[ThrowableQueue] = None,
        maxsize: Optional[int] = None,
        root: Optional[Any] = None,
        active_partitions: Optional[Set[int]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> "ChannelT[_T]":
        ...

    @abc.abstractmethod
    def clone_using_queue(self, queue: ThrowableQueue) -> "ChannelT[_T]":
        ...

    @abc.abstractmethod
    def stream(self, **kwargs: Any) -> AsyncIterator['_EventT[_T]']:
        ...

    @abc.abstractmethod
    def get_topic_name(self) -> str:
        ...

    @abc.abstractmethod
    async def send(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional['_SchemaT'] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
    ) -> RecordMetadata:
        ...

    @abc.abstractmethod
    def send_soon(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional['_SchemaT'] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False,
    ) -> FutureMessage:
        ...

    @abc.abstractmethod
    def as_future_message(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional['_SchemaT'] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        callback: Optional[MessageSentCallback] = None,
        eager_partitioning: bool = False,
    ) -> FutureMessage:
        ...

    @abc.abstractmethod
    async def publish_message(self, fut: Awaitable[RecordMetadata], wait: bool = True) -> RecordMetadata:
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_declare(self) -> None:
        ...

    @abc.abstractmethod
    async def declare(self) -> None:
        ...

    @abc.abstractmethod
    def prepare_key(self, key: Any, key_serializer: Optional[Any], schema: Optional['_SchemaT'] = None) -> Any:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Any, value_serializer: Optional[Any], schema: Optional['_SchemaT'] = None) -> Any:
        ...

    @abc.abstractmethod
    async def decode(self, message: Message, *, propagate: bool = False) -> "_EventT[_T]":
        ...

    @abc.abstractmethod
    async def deliver(self, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def put(self, value: _T) -> None:
        ...

    @abc.abstractmethod
    async def get(self, *, timeout: Optional[Seconds] = None) -> "_EventT[_T]":
        ...

    @abc.abstractmethod
    def empty(self) -> bool:
        ...

    @abc.abstractmethod
    async def on_key_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def on_value_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def on_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    @abc.abstractmethod
    def on_stop_iteration(self) -> StopAsyncIteration:
        ...

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator["_EventT[_T]"]:
        ...

    @abc.abstractmethod
    def __anext__(self) -> Awaitable["_EventT[_T]"]:
        ...

    @abc.abstractmethod
    async def throw(self, exc: Exception) -> None:
        ...

    @abc.abstractmethod
    def _throw(self, exc: Exception) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs: Any) -> "ChannelT[_T]":
        ...

    @property
    @abc.abstractmethod
    def subscriber_count(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def queue(self) -> ThrowableQueue:
        ...