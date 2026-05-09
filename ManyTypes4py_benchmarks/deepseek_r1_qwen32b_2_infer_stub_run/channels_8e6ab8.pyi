"""Channel.

A channel is used to send values to streams.

The stream will iterate over incoming events in the channel.
"""

from __future__ import annotations
import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from weakref import WeakSet
from mode import Seconds
from mode.utils.futures import maybe_async
from mode.utils.queues import ThrowableQueue
from .types import (
    AppT,
    ChannelT,
    CodecArg,
    EventT,
    FutureMessage,
    K,
    Message,
    MessageSentCallback,
    ModelArg,
    PendingMessage,
    RecordMetadata,
    SchemaT,
    StreamT,
    TP,
    V,
)
from .types.core import HeadersArg, OpenHeadersArg

__all__ = ['Channel', 'SerializedChannel']

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T], Generic[T]):
    """Create new channel."""

    app: AppT
    loop: Optional[asyncio.AbstractEventLoop]
    is_iterator: bool
    _queue: Optional[ThrowableQueue[EventT]]
    maxsize: Optional[int]
    Deliver: Callable[[Message], Awaitable[None]]
    _root: Channel
    active_partitions: Optional[Set[TP]]
    _subscribers: WeakSet[Channel]
    schema: SchemaT
    key_type: ModelArg
    value_type: ModelArg

    def __init__(self, app: AppT, *, schema: Optional[SchemaT] = None, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, is_iterator: bool = False, queue: Optional[ThrowableQueue[EventT]] = None, maxsize: Optional[int] = None, root: Optional[Channel] = None, active_partitions: Optional[Set[TP]] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        ...

    def _get_default_schema(self, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None) -> SchemaT:
        ...

    @property
    def queue(self) -> ThrowableQueue[EventT]:
        ...

    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> ChannelT[T]:
        ...

    def clone_using_queue(self, queue: ThrowableQueue[EventT]) -> ChannelT[T]:
        ...

    def _clone(self, **kwargs: Any) -> Channel:
        ...

    def _clone_args(self) -> Dict[str, Any]:
        ...

    def stream(self, **kwargs: Any) -> StreamT:
        ...

    def get_topic_name(self) -> str:
        ...

    async def send(self, *, key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, callback: Optional[MessageSentCallback] = None, force: bool = False) -> FutureMessage:
        ...

    def send_soon(self, *, key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, callback: Optional[MessageSentCallback] = None, force: bool = False, eager_partitioning: bool = False) -> None:
        ...

    def as_future_message(self, key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, callback: Optional[MessageSentCallback] = None, eager_partitioning: bool = False) -> FutureMessage:
        ...

    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg:
        ...

    async def _send_now(self, key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, callback: Optional[MessageSentCallback] = None) -> FutureMessage:
        ...

    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> FutureMessage:
        ...

    def _future_message_to_event(self, fut: FutureMessage) -> EventT:
        ...

    async def _finalize_message(self, fut: FutureMessage, result: RecordMetadata) -> FutureMessage:
        ...

    @stampede
    async def maybe_declare(self) -> None:
        ...

    async def declare(self) -> None:
        ...

    def prepare_key(self, key: Optional[K], key_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, HeadersArg]:
        ...

    def prepare_value(self, value: Optional[V], value_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, HeadersArg]:
        ...

    async def decode(self, message: Message, *, propagate: bool = False) -> EventT:
        ...

    async def deliver(self, message: Message) -> None:
        ...

    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]:
        ...

    def _create_event(self, key: Any, value: Any, headers: HeadersArg, message: Message) -> EventT:
        ...

    async def put(self, value: Any) -> None:
        ...

    async def get(self, *, timeout: Optional[Seconds] = None) -> EventT:
        ...

    def empty(self) -> bool:
        ...

    async def on_key_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    async def on_value_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    async def on_decode_error(self, exc: Exception, message: Message) -> None:
        ...

    def on_stop_iteration(self) -> None:
        ...

    def derive(self, **kwargs: Any) -> Channel:
        ...

    def __aiter__(self) -> Channel:
        ...

    async def __anext__(self) -> EventT:
        ...

    async def throw(self, exc: BaseException) -> None:
        ...

    def _throw(self, exc: BaseException) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def _object_id_as_hex(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    @property
    def subscriber_count(self) -> int:
        ...

    @property
    def label(self) -> str:
        ...

class SerializedChannel(Channel[T], Generic[T]):
    """Channel with serialization support."""

    key_serializer: CodecArg
    value_serializer: CodecArg
    allow_empty: Optional[bool]

    def __init__(self, app: AppT, *, schema: Optional[SchemaT] = None, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, allow_empty: Optional[bool] = None, **kwargs: Any) -> None:
        ...

    def _contribute_to_schema(self, schema: SchemaT, *, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    def _get_default_schema(self, key_type: Optional[ModelArg] = None, value_type: Optional[ModelArg] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, allow_empty: Optional[bool] = None) -> SchemaT:
        ...

    def _clone_args(self) -> Dict[str, Any]:
        ...

    def prepare_key(self, key: Optional[K], key_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, HeadersArg]:
        ...

    def prepare_value(self, value: Optional[V], value_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, HeadersArg]:
        ...