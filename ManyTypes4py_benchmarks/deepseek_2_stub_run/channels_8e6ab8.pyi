```python
import asyncio
from typing import Any, Awaitable, Callable, Mapping, MutableSet, Optional, Set, Tuple, TypeVar, Union
from weakref import WeakSet
from mode import Seconds
from .types import AppT, ChannelT, CodecArg, EventT, FutureMessage, K, Message, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, StreamT, TP, V
from .types.core import HeadersArg, OpenHeadersArg

__all__: Tuple[str, ...] = ('Channel',)

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    app: Any
    loop: Any
    is_iterator: bool
    maxsize: Optional[int]
    schema: SchemaT
    key_type: ModelArg
    value_type: ModelArg
    active_partitions: Optional[Set[TP]]
    _queue: Any
    _root: 'Channel'
    _subscribers: WeakSet

    def __init__(
        self,
        app: Any,
        *,
        schema: Optional[SchemaT] = ...,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        is_iterator: bool = ...,
        queue: Any = ...,
        maxsize: Optional[int] = ...,
        root: Optional['Channel'] = ...,
        active_partitions: Optional[Set[TP]] = ...,
        loop: Any = ...
    ) -> None: ...

    def _get_default_schema(self, key_type: Optional[ModelArg] = ..., value_type: Optional[ModelArg] = ...) -> SchemaT: ...

    @property
    def queue(self) -> Any: ...

    def clone(self, *, is_iterator: Optional[bool] = ..., **kwargs: Any) -> 'Channel[T]': ...

    def clone_using_queue(self, queue: Any) -> 'Channel[T]': ...

    def _clone(self, **kwargs: Any) -> 'Channel[T]': ...

    def _clone_args(self) -> Mapping[str, Any]: ...

    def stream(self, **kwargs: Any) -> StreamT: ...

    def get_topic_name(self) -> str: ...

    async def send(
        self,
        *,
        key: Any = ...,
        value: Any = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
        force: bool = ...
    ) -> RecordMetadata: ...

    def send_soon(
        self,
        *,
        key: Any = ...,
        value: Any = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
        force: bool = ...,
        eager_partitioning: bool = ...
    ) -> Any: ...

    def as_future_message(
        self,
        key: Any = ...,
        value: Any = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
        eager_partitioning: bool = ...
    ) -> FutureMessage: ...

    def prepare_headers(self, headers: Optional[HeadersArg]) -> OpenHeadersArg: ...

    async def _send_now(
        self,
        key: Any = ...,
        value: Any = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...
    ) -> RecordMetadata: ...

    async def publish_message(self, fut: FutureMessage, wait: bool = ...) -> FutureMessage: ...

    def _future_message_to_event(self, fut: FutureMessage) -> EventT: ...

    async def _finalize_message(self, fut: FutureMessage, result: RecordMetadata) -> FutureMessage: ...

    async def maybe_declare(self) -> None: ...

    async def declare(self) -> None: ...

    def prepare_key(
        self,
        key: Any,
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = ...,
        headers: Optional[OpenHeadersArg] = ...
    ) -> Tuple[Any, OpenHeadersArg]: ...

    def prepare_value(
        self,
        value: Any,
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = ...,
        headers: Optional[OpenHeadersArg] = ...
    ) -> Tuple[Any, OpenHeadersArg]: ...

    async def decode(self, message: Message, *, propagate: bool = ...) -> EventT: ...

    async def deliver(self, message: Message) -> None: ...

    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]: ...

    def _create_event(self, key: Any, value: Any, headers: Optional[HeadersArg], message: Message) -> EventT: ...

    async def put(self, value: EventT) -> None: ...

    async def get(self, *, timeout: Optional[Seconds] = ...) -> EventT: ...

    def empty(self) -> bool: ...

    async def on_key_decode_error(self, exc: Exception, message: Message) -> None: ...

    async def on_value_decode_error(self, exc: Exception, message: Message) -> None: ...

    async def on_decode_error(self, exc: Exception, message: Message) -> None: ...

    def on_stop_iteration(self) -> None: ...

    def derive(self, **kwargs: Any) -> 'Channel[T]': ...

    def __aiter__(self) -> 'Channel[T]': ...

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
    key_serializer: Optional[CodecArg]
    value_serializer: Optional[CodecArg]
    allow_empty: Optional[bool]

    def __init__(
        self,
        app: Any,
        *,
        schema: Optional[SchemaT] = ...,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        allow_empty: Optional[bool] = ...,
        **kwargs: Any
    ) -> None: ...

    def _contribute_to_schema(
        self,
        schema: SchemaT,
        *,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        allow_empty: Optional[bool] = ...
    ) -> None: ...

    def _get_default_schema(
        self,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        allow_empty: Optional[bool] = ...
    ) -> SchemaT: ...

    def _clone_args(self) -> Mapping[str, Any]: ...

    def prepare_key(
        self,
        key: Any,
        key_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = ...,
        headers: Optional[OpenHeadersArg] = ...
    ) -> Tuple[Any, OpenHeadersArg]: ...

    def prepare_value(
        self,
        value: Any,
        value_serializer: Optional[CodecArg],
        schema: Optional[SchemaT] = ...,
        headers: Optional[OpenHeadersArg] = ...
    ) -> Tuple[Any, OpenHeadersArg]: ...
```