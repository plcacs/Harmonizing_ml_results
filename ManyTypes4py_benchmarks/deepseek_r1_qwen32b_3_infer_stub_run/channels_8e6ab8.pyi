"""Channel.

A channel is used to send values to streams.

The stream will iterate over incoming events in the channel.

"""

import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    WeakSet,
)
from weakref import WeakSet
from mode import Seconds
from mode.utils.futures import maybe_async
from mode.utils.queues import ThrowableQueue
from faust.types import (
    AppT,
    ChannelT,
    CodecArg,
    EventT,
    FutureMessage,
    HeadersArg,
    K,
    Message,
    MessageSentCallback,
    ModelArg,
    OpenHeadersArg,
    PendingMessage,
    RecordMetadata,
    SchemaT,
    StreamT,
    TP,
    V,
)
from faust.types.core import HeadersArg, OpenHeadersArg
from faust.types.tuples import _PendingMessage_to_Message

__all__ = ['Channel']

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    """Create new channel."""

    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        is_iterator: bool = False,
        queue: Optional[ThrowableQueue] = None,
        maxsize: Optional[int] = None,
        root: Optional['Channel'] = None,
        active_partitions: Optional[Set[TP]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        ...

    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> ChannelT:
        ...

    def clone_using_queue(self, queue: ThrowableQueue) -> Channel:
        ...

    def stream(self, **kwargs: Any) -> StreamT:
        ...

    def get_topic_name(self) -> str:
        ...

    async def send(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
    ) -> FutureMessage:
        ...

    def send_soon(
        self,
        *,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False,
    ) -> None:
        ...

    def as_future_message(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        eager_partitioning: bool = False,
    ) -> FutureMessage:
        ...

    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg:
        ...

    async def _send_now(
        self,
        key: Optional[K] = None,
        value: Optional[V] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
    ) -> FutureMessage:
        ...

    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> FutureMessage:
        ...

    def _future_message_to_event(self, fut: FutureMessage) -> EventT:
        ...

    async def _finalize_message(
        self, fut: FutureMessage, result: RecordMetadata
    ) -> FutureMessage:
        ...

    @stampede
    async def maybe_declare(self) -> None:
        ...

    async def declare(self) -> None:
        ...

    def prepare_key(
        self,
        key: Optional[K],
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[HeadersArg] = None,
    ) -> Tuple[Optional[K], HeadersArg]:
        ...

    def prepare_value(
        self,
        value: Optional[V],
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[HeadersArg] = None,
    ) -> Tuple[Optional[V], HeadersArg]:
        ...

    async def decode(self, message: Message, *, propagate: bool = False) -> EventT:
        ...

    async def deliver(self, message: Message) -> None:
        ...

    def _compile_deliver(self) -> Callable[[], Awaitable[None]]:
        ...

    def _create_event(
        self,
        key: Optional[K],
        value: Optional[V],
        headers: Optional[HeadersArg],
        message: Message,
    ) -> EventT:
        ...

    async def put(self, value: T) -> None:
        ...

    async def get(self, *, timeout: Optional[Seconds] = None) -> T:
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

    def derive(self, **kwargs: Any) -> ChannelT:
        ...

    def __aiter__(self) -> ChannelT:
        ...

    async def __anext__(self) -> T:
        ...

    async def throw(self, exc: Exception) -> None:
        ...

    def _throw(self, exc: Exception) -> None:
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

class SerializedChannel(Channel[T]):
    """Channel with serialization support."""

    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        ...

    def prepare_key(
        self,
        key: Optional[K],
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[HeadersArg] = None,
    ) -> Tuple[Optional[Any], HeadersArg]:
        ...

    def prepare_value(
        self,
        value: Optional[V],
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[HeadersArg] = None,
    ) -> Tuple[Optional[Any], HeadersArg]:
        ...