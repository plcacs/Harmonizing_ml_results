```pyi
from typing import Any, Awaitable, Callable, Mapping, MutableSet, Optional, Set, Tuple, TypeVar, WeakSet
from mode import Seconds
from .types import AppT, ChannelT, CodecArg, EventT, FutureMessage, K, Message, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, StreamT, TP, V
from .types.core import HeadersArg, OpenHeadersArg

__all__: list[str]

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    app: AppT
    loop: Any
    is_iterator: bool
    _queue: Any
    maxsize: Optional[int]
    deliver: Callable[[Message], Awaitable[None]]
    _root: Optional[Channel]
    active_partitions: Optional[Set[TP]]
    schema: SchemaT
    key_type: Any
    value_type: Any
    _subscribers: WeakSet[Channel]
    
    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        is_iterator: bool = False,
        queue: Any = None,
        maxsize: Optional[int] = None,
        root: Optional[Channel] = None,
        active_partitions: Optional[Set[TP]] = None,
        loop: Any = None,
    ) -> None: ...
    
    def _get_default_schema(
        self,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
    ) -> SchemaT: ...
    
    @property
    def queue(self) -> Any: ...
    
    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> Channel: ...
    def clone_using_queue(self, queue: Any) -> Channel: ...
    def _clone(self, **kwargs: Any) -> Channel: ...
    def _clone_args(self) -> dict[str, Any]: ...
    def stream(self, **kwargs: Any) -> StreamT: ...
    def get_topic_name(self) -> str: ...
    
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
    ) -> RecordMetadata: ...
    
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
    ) -> FutureMessage: ...
    
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
    ) -> FutureMessage: ...
    
    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg: ...
    
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
    ) -> RecordMetadata: ...
    
    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> FutureMessage: ...
    def _future_message_to_event(self, fut: FutureMessage) -> EventT: ...
    async def _finalize_message(self, fut: FutureMessage, result: RecordMetadata) -> FutureMessage: ...
    async def maybe_declare(self) -> None: ...
    async def declare(self) -> None: ...
    
    def prepare_key(
        self,
        key: Optional[K],
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None,
    ) -> Tuple[Optional[K], OpenHeadersArg]: ...
    
    def prepare_value(
        self,
        value: Optional[V],
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None,
    ) -> Tuple[Optional[V], OpenHeadersArg]: ...
    
    async def decode(self, message: Message, *, propagate: bool = False) -> EventT: ...
    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]: ...
    def _create_event(self, key: Any, value: Any, headers: OpenHeadersArg, message: Message) -> EventT: ...
    async def put(self, value: EventT) -> None: ...
    async def get(self, *, timeout: Optional[Seconds] = None) -> EventT: ...
    def empty(self) -> bool: ...
    
    async def on_key_decode_error(self, exc: Exception, message: Message) -> None: ...
    async def on_value_decode_error(self, exc: Exception, message: Message) -> None: ...
    async def on_decode_error(self, exc: Exception, message: Message) -> None: ...
    def on_stop_iteration(self) -> None: ...
    def derive(self, **kwargs: Any) -> Channel: ...
    
    def __aiter__(self) -> Channel: ...
    async def __anext__(self) -> EventT: ...
    async def throw(self, exc: Exception) -> None: ...
    def _throw(self, exc: Exception) -> None: ...
    def __repr__(self) -> str: ...
    def _object_id_as_hex(self) -> str: ...
    def __str__(self) -> str: ...
    
    @property
    def subscriber_count(self) -> int: ...
    
    @property
    def label(self) -> str: ...

class SerializedChannel(Channel[T]):
    key_serializer: Any
    value_serializer: Any
    allow_empty: Any
    
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
    ) -> None: ...
    
    def _contribute_to_schema(
        self,
        schema: SchemaT,
        *,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> None: ...
    
    def _get_default_schema(
        self,
        key_type: Optional[ModelArg] = None,
        value_type: Optional[ModelArg] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> SchemaT: ...
    
    def _clone_args(self) -> dict[str, Any]: ...
    
    def prepare_key(
        self,
        key: Optional[K],
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None,
    ) -> Tuple[Optional[bytes], OpenHeadersArg]: ...
    
    def prepare_value(
        self,
        value: Optional[V],
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = None,
        headers: Optional[OpenHeadersArg] = None,
    ) -> Tuple[Optional[bytes], OpenHeadersArg]: ...
```