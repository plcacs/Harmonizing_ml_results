from typing import Any, Awaitable, Callable, Mapping, MutableSet, Optional, Set, Tuple, TypeVar
from weakref import WeakSet
import asyncio
from mode import Seconds, get_logger, want_seconds
from mode.utils.futures import maybe_async, stampede
from mode.utils.queues import ThrowableQueue
from .types import AppT, ChannelT, CodecArg, EventT, FutureMessage, K, Message, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, StreamT, TP, V
from .types.core import HeadersArg, OpenHeadersArg, prepare_headers
from .types.tuples import _PendingMessage_to_Message

__all__ = ['Channel']

logger = get_logger(__name__)

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    def __init__(self, app: AppT, *, schema: Optional[SchemaT] = None, key_type: Optional[TypeVar] = None, value_type: Optional[TypeVar] = None, is_iterator: bool = False, queue: Optional[ThrowableQueue] = None, maxsize: Optional[int] = None, root: Optional['Channel'] = None, active_partitions: Optional[Set[TP]] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        ...

    def _get_default_schema(self, key_type: Optional[TypeVar] = None, value_type: Optional[TypeVar] = None) -> SchemaT:
        ...

    @property
    def queue(self) -> ThrowableQueue:
        ...

    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> 'Channel':
        ...

    def clone_using_queue(self, queue: ThrowableQueue) -> 'Channel':
        ...

    def _clone(self, **kwargs: Any) -> 'Channel':
        ...

    def _clone_args(self) -> Mapping[str, Any]:
        ...

    def stream(self, **kwargs: Any) -> StreamT:
        ...

    def get_topic_name(self) -> str:
        ...

    async def send(self, *, key: Any = None, value: Any = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, callback: Optional[MessageSentCallback] = None, force: bool = False) -> RecordMetadata:
        ...

    def send_soon(self, *, key: Any = None, value: Any = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, callback: Optional[MessageSentCallback] = None, force: bool = False, eager_partitioning: bool = False) -> None:
        ...

    def as_future_message(self, key: Any = None, value: Any = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, callback: Optional[MessageSentCallback] = None, eager_partitioning: bool = False) -> FutureMessage:
        ...

    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg:
        ...

    async def _send_now(self, key: Any = None, value: Any = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[HeadersArg] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, callback: Optional[MessageSentCallback] = None) -> RecordMetadata:
        ...

    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> RecordMetadata:
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

    def prepare_key(self, key: Any, key_serializer: Optional[Callable], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, OpenHeadersArg]:
        ...

    def prepare_value(self, value: Any, value_serializer: Optional[Callable], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, OpenHeadersArg]:
        ...

    async def decode(self, message: Message, propagate: bool = False) -> EventT:
        ...

    async def deliver(self, message: Message) -> None:
        ...

    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]:
        ...

    def _create_event(self, key: Any, value: Any, headers: OpenHeadersArg, message: Message) -> EventT:
        ...

    async def put(self, value: Any) -> None:
        ...

    async def get(self, timeout: Optional[float] = None) -> Any:
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

    def derive(self, **kwargs: Any) -> 'Channel':
        ...

    def __aiter__(self) -> 'Channel':
        ...

    async def __anext__(self) -> Any:
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
    def __init__(self, app: AppT, *, schema: Optional[SchemaT] = None, key_type: Optional[TypeVar] = None, value_type: Optional[TypeVar] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, allow_empty: Optional[bool] = None, **kwargs: Any) -> None:
        ...

    def _contribute_to_schema(self, schema: SchemaT, *, key_type: Optional[TypeVar] = None, value_type: Optional[TypeVar] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, allow_empty: Optional[bool] = None) -> None:
        ...

    def _get_default_schema(self, key_type: Optional[TypeVar] = None, value_type: Optional[TypeVar] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, allow_empty: Optional[bool] = None) -> SchemaT:
        ...

    @no_type_check
    def _clone_args(self) -> Mapping[str, Any]:
        ...

    def prepare_key(self, key: Any, key_serializer: Optional[Callable], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, OpenHeadersArg]:
        ...

    def prepare_value(self, value: Any, value_serializer: Optional[Callable], schema: Optional[SchemaT] = None, headers: Optional[HeadersArg] = None) -> Tuple[Any, OpenHeadersArg]:
        ...
