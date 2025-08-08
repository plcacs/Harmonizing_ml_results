from typing import Any, Awaitable, Callable, Mapping, Optional, Pattern, Sequence, Set, Union, cast
from .types import AppT, CodecArg, EventT, FutureMessage, HeadersArg, K, MessageSentCallback, ModelArg, PendingMessage, RecordMetadata, SchemaT, TP, V
from .types.topics import ChannelT, TopicT
from .types.transports import ProducerT

class Topic(SerializedChannel, TopicT):
    _partitions: Optional[int] = None
    _pattern: Optional[Pattern] = None

    def __init__(self, app: AppT, *, topics: Optional[Sequence[str]] = None, pattern: Optional[str] = None, schema: Optional[SchemaT] = None, key_type: Optional[Union[ModelArg, str, bytes]] = None, value_type: Optional[Union[ModelArg, str, bytes]] = None, is_iterator: bool = False, partitions: Optional[int] = None, retention: Optional[Union[float, datetime.timedelta]] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Mapping[str, Any]] = None, queue: Optional[ThrowableQueue] = None, key_serializer: Optional[Callable[[Any], bytes]] = None, value_serializer: Optional[Callable[[Any], bytes]] = None, maxsize: Optional[int] = None, root: Optional[Any] = None, active_partitions: Optional[Set[TP]] = None, allow_empty: Optional[bool] = None, has_prefix: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        ...

    async def send(self, *, key: Optional[Any] = None, value: Optional[Any] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[Mapping[str, Any]] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable[[Any], bytes]] = None, value_serializer: Optional[Callable[[Any], bytes]] = None, callback: Optional[MessageSentCallback] = None, force: bool = False) -> Awaitable[None]:
        ...

    def send_soon(self, *, key: Optional[Any] = None, value: Optional[Any] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[Mapping[str, Any]] = None, schema: Optional[SchemaT] = None, key_serializer: Optional[Callable[[Any], bytes]] = None, value_serializer: Optional[Callable[[Any], bytes]] = None, callback: Optional[MessageSentCallback] = None, force: bool = False, eager_partitioning: bool = False) -> FutureMessage:
        ...

    async def put(self, event: EventT) -> None:
        ...

    def _clone_args(self) -> Mapping[str, Any]:
        ...

    @property
    def pattern(self) -> Optional[Pattern]:
        ...

    @pattern.setter
    def pattern(self, pattern: Optional[str]) -> None:
        ...

    @property
    def partitions(self) -> Optional[int]:
        ...

    @partitions.setter
    def partitions(self, partitions: Optional[int]) -> None:
        ...

    def derive(self, **kwargs: Any) -> TopicT:
        ...

    def derive_topic(self, *, topics: Optional[Sequence[str]] = None, schema: Optional[SchemaT] = None, key_type: Optional[Union[ModelArg, str, bytes]] = None, value_type: Optional[Union[ModelArg, str, bytes]] = None, key_serializer: Optional[Callable[[Any], bytes]] = None, value_serializer: Optional[Callable[[Any], bytes]] = None, partitions: Optional[int] = None, retention: Optional[Union[float, datetime.timedelta]] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, internal: Optional[bool] = None, config: Optional[Mapping[str, Any]] = None, prefix: str = '', suffix: str = '', **kwargs: Any) -> TopicT:
        ...

    def get_topic_name(self) -> str:
        ...

    async def _get_producer(self) -> ProducerT:
        ...

    async def publish_message(self, fut: FutureMessage, wait: bool = False) -> Awaitable[None]:
        ...

    def _topic_name_or_default(self, obj: Union[str, TopicT]) -> str:
        ...

    def _on_published(self, fut: asyncio.Future, message: PendingMessage, producer: ProducerT, state: Any) -> None:
        ...

    @stampede
    async def maybe_declare(self) -> None:
        ...

    async def declare(self) -> None:
        ...

    def __aiter__(self) -> TopicT:
        ...
