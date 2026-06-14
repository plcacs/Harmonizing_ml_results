"""Message transport using :pypi:`confluent_kafka`."""
import asyncio
import typing
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Type,
)

from mode import Service, get_logger
from mode.threads import QueueServiceThread
from mode.utils.times import Seconds
from yarl import URL

from faust.transport import base
from faust.transport.consumer import ConsumerThread, RecordMap, ThreadDelegateConsumer
from faust.types import AppT, ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT

if typing.TYPE_CHECKING:
    from confluent_kafka import Consumer as _Consumer
    from confluent_kafka import Message as _Message
    from confluent_kafka import Producer as _Producer
else:
    class _Consumer: ...
    class _Producer: ...
    class _Message: ...

__all__: List[str]
logger: Any

def server_list(urls: Iterable[URL], default_port: int) -> str: ...

class Consumer(ThreadDelegateConsumer):
    logger: Any
    def _new_consumer_thread(self) -> ConfluentConsumerThread: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    def _to_message(self, tp: TP, record: _Message) -> ConsumerMessage: ...
    def _new_topicpartition(self, topic: str, partition: int) -> TP: ...

class ConfluentConsumerThread(ConsumerThread):
    _consumer: Optional[Any]
    _assigned: bool
    async def on_start(self) -> None: ...
    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> Any: ...
    def _create_worker_consumer(self, transport: Transport, loop: asyncio.AbstractEventLoop) -> Any: ...
    def _create_client_consumer(self, transport: Transport, loop: asyncio.AbstractEventLoop) -> Any: ...
    def close(self) -> None: ...
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    def _on_assign(self, consumer: Any, assigned: Any) -> None: ...
    def _on_revoke(self, consumer: Any, revoked: Any) -> None: ...
    async def seek_to_committed(self) -> Mapping[TP, int]: ...
    async def _seek_to_committed(self) -> Mapping[TP, int]: ...
    async def _committed_offsets(self, partitions: Iterable[TP]) -> Mapping[TP, int]: ...
    async def commit(self, tps: Mapping[TP, int]) -> bool: ...
    async def position(self, tp: TP) -> Optional[int]: ...
    async def seek_to_beginning(self, *partitions: TP) -> None: ...
    async def seek_wait(self, partitions: Mapping[TP, int]) -> None: ...
    async def _seek_wait(self, consumer: Any, partitions: Mapping[TP, int]) -> None: ...
    def seek(self, partition: TP, offset: int) -> None: ...
    def assignment(self) -> Set[TP]: ...
    def highwater(self, tp: TP) -> int: ...
    def topic_partitions(self, topic: str) -> Optional[int]: ...
    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def _earliest_offsets(self, partitions: Iterable[TP]) -> Mapping[TP, int]: ...
    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]: ...
    async def _highwaters(self, partitions: Iterable[TP]) -> Mapping[TP, int]: ...
    def _ensure_consumer(self) -> Any: ...
    async def getmany(self, active_partitions: Optional[Set[TP]], timeout: float) -> RecordMap: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes], partition: int = ...) -> Optional[int]: ...

class ProducerProduceFuture(asyncio.Future[RecordMetadata]):
    def set_from_on_delivery(self, err: Optional[Any], msg: _Message) -> None: ...
    def message_to_metadata(self, message: _Message) -> RecordMetadata: ...

class ProducerThread(QueueServiceThread):
    _producer: Optional[Any]
    _flush_soon: Optional[asyncio.Future[Any]]
    producer: Producer
    transport: Transport
    app: AppT
    def __init__(self, producer: Producer, **kwargs: Any) -> None: ...
    async def on_start(self) -> None: ...
    async def flush(self) -> None: ...
    async def on_thread_stop(self) -> None: ...
    def produce(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        on_delivery: Callable[..., Any],
        **kwargs: Any,
    ) -> None: ...
    async def _background_flush(self) -> None: ...

class Producer(base.Producer):
    logger: Any
    _quick_produce: Optional[Callable[..., None]]
    _producer_thread: ProducerThread
    def __post_init__(self) -> None: ...
    async def _on_irrecoverable_error(self, exc: BaseException) -> None: ...
    async def on_restart(self) -> None: ...
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = ...,
        timeout: Seconds = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        ensure_created: bool = ...,
    ) -> None: ...
    async def on_start(self) -> None: ...
    async def on_stop(self) -> None: ...
    async def send(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = ...,
    ) -> Awaitable[RecordMetadata]: ...
    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = ...,
    ) -> RecordMetadata: ...
    async def flush(self) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes]) -> Optional[int]: ...

class Transport(base.Transport):
    Consumer: ClassVar[Type[Consumer]]  # type: ignore[assignment]
    Producer: ClassVar[Type[Producer]]  # type: ignore[assignment]
    default_port: int
    driver_version: str
    def _topic_config(
        self,
        retention: Optional[int] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
    ) -> Mapping[str, Any]: ...