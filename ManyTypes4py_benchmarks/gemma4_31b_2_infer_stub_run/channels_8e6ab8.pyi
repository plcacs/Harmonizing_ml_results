import asyncio
from typing import Any, Awaitable, Callable, Optional, Set, Tuple, TypeVar, Union, overload
from .types import AppT, ChannelT, CodecArg, EventT, FutureMessage, K, Message, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, StreamT, TP, V

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)

class Channel(ChannelT[T]):
    app: AppT
    loop: Optional[asyncio.AbstractEventLoop]
    is_iterator: bool
    maxsize: Optional[int]
    deliver: Callable[[Message], Awaitable[None]]
    _root: Channel[T]
    active_partitions: Optional[Set[TP]]
    _subscribers: Set[Channel[T]]
    schema: SchemaT
    key_type: Optional[Any]
    value_type: Optional[Any]

    def __init__(
        self,
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        is_iterator: bool = False,
        queue: Optional[Any] = None,
        maxsize: Optional[int] = None,
        root: Optional[Channel[T]] = None,
        active_partitions: Optional[Set[TP]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None: ...

    def _get_default_schema(self, key_type: Optional[Any] = None, value_type: Optional[Any] = None) -> SchemaT: ...

    @property
    def queue(self) -> Any: ...

    def clone(self, *, is_iterator: Optional[bool] = None, **kwargs: Any) -> Channel[T]: ...

    def clone_using_queue(self, queue: Any) -> Channel[T]: ...

    def _clone(self, **kwargs: Any) -> Channel[T]: ...

    def _clone_args(self) -> dict[str, Any]: ...

    def stream(self, **kwargs: Any) -> StreamT: ...

    def get_topic_name(self) -> str: ...

    async def send(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[Union[dict[str, bytes], list[Tuple[str, bytes]]]] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
    ) -> RecordMetadata: ...

    def send_soon(
        self,
        *,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[Union[dict[str, bytes], list[Tuple[str, bytes]]]] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        force: bool = False,
        eager_partitioning: bool = False,
    ) -> None: ...

    def as_future_message(
        self,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[Union[dict[str, bytes], list[Tuple[str, bytes]]]] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
        eager_partitioning: bool = False,
    ) -> FutureMessage: ...

    def prepare_headers(self, headers: Optional[Union[dict[str, bytes], list[Tuple[str, bytes]]]]) -> dict[str, bytes]: ...

    async def _send_now(
        self,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[Union[dict[str, bytes], list[Tuple[str, bytes]]]] = None,
        schema: Optional[SchemaT] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        callback: Optional[MessageSentCallback] = None,
    ) -> RecordMetadata: ...

    async def publish_message(self, fut: FutureMessage, wait: bool = True) -> FutureMessage: ...

    def _future_message_to_event(self, fut: FutureMessage) -> EventT: ...

    async def _finalize_message(self, fut: FutureMessage, result: RecordMetadata) -> FutureMessage: ...

    @stampede
    async def maybe_declare(self) -> None: ...

    async def declare(self) -> None: ...

    def prepare_key(self, key: Any, key_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[dict[str, bytes]] = None) -> Tuple[Any, Optional[dict[str, bytes]]]: ...

    def prepare_value(self, value: Any, value_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[dict[str, bytes]] = None) -> Tuple[Any, Optional[dict[str, bytes]]]: ...

    async def decode(self, message: Message, *, propagate: bool = False) -> EventT: ...

    async def deliver(self, message: Message) -> None: ...

    def _compile_deliver(self) -> Callable[[Message], Awaitable[None]]: ...

    def _create_event(self, key: Any, value: Any, headers: dict[str, bytes], message: Message) -> EventT: ...

    async def put(self, value: EventT) -> None: ...

    async def get(self, *, timeout: Optional[Union[int, float]] = None) -> EventT: ...

    def empty(self) -> bool: ...

    async def on_key_decode_error(self, exc: Exception, message: Message) -> None: ...

    async def on_value_decode_error(self, exc: Exception, message: Message) -> None: ...

    async def on_decode_error(self, exc: Exception, message: Message) -> None: ...

    def on_stop_iteration(self) -> None: ...

    def derive(self, **kwargs: Any) -> Channel[T]: ...

    def __aiter__(self) -> Channel[T]: ...

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
        app: AppT,
        *,
        schema: Optional[SchemaT] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
        **kwargs: Any,
    ) -> None: ...

    def _contribute_to_schema(
        self,
        schema: SchemaT,
        *,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> None: ...

    def _get_default_schema(
        self,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        allow_empty: Optional[bool] = None,
    ) -> SchemaT: ...

    def _clone_args(self) -> dict[str, Any]: ...

    def prepare_key(self, key: Any, key_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[dict[str, bytes]] = None) -> Tuple[Any, Optional[dict[str, bytes]]]: ...

    def prepare_value(self, value: Any, value_serializer: Optional[CodecArg], schema: Optional[SchemaT] = None, headers: Optional[dict[str, bytes]] = None) -> Tuple[Any, Optional[dict[str, bytes]]]: ...