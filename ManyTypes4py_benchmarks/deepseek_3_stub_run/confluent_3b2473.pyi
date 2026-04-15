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
    Tuple,
    Type,
    Union,
    cast,
)

from mode import Service
from mode.threads import QueueServiceThread
from mode.utils.futures import notify
from mode.utils.times import Seconds
from yarl import URL
from faust.exceptions import ConsumerNotStarted, ProducerSendError
from faust.transport import base
from faust.transport.consumer import ConsumerThread, RecordMap, ThreadDelegateConsumer
from faust.types import AppT, ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT
import confluent_kafka
from confluent_kafka import TopicPartition as _TopicPartition
from confluent_kafka import KafkaException

if typing.TYPE_CHECKING:
    from confluent_kafka import Consumer as _Consumer
    from confluent_kafka import Producer as _Producer
    from confluent_kafka import Message as _Message
else:
    class _Consumer: ...
    class _Producer: ...
    class _Message: ...

__all__: List[str] = ["Consumer", "Producer", "Transport"]

logger: Any = ...

def server_list(urls: Iterable[URL], default_port: int) -> str: ...

class Consumer(ThreadDelegateConsumer):
    logger: ClassVar[Any] = ...
    
    def _new_consumer_thread(self) -> "ConfluentConsumerThread": ...
    
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: Seconds = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...
    
    def _to_message(self, tp: TP, record: _Message) -> ConsumerMessage: ...
    
    def _new_topicpartition(self, topic: str, partition: int) -> TP: ...

class ConfluentConsumerThread(ConsumerThread):
    _consumer: Optional[_Consumer] = ...
    _assigned: bool = ...
    
    async def on_start(self) -> None: ...
    
    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    
    def _create_worker_consumer(self, transport: "Transport", loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    
    def _create_client_consumer(self, transport: "Transport", loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    
    def close(self) -> None: ...
    
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    
    def _on_assign(self, consumer: _Consumer, assigned: List[_TopicPartition]) -> None: ...
    
    def _on_revoke(self, consumer: _Consumer, revoked: List[_TopicPartition]) -> None: ...
    
    async def seek_to_committed(self) -> Dict[TP, int]: ...
    
    async def _seek_to_committed(self) -> Dict[TP, int]: ...
    
    async def _committed_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    
    async def commit(self, tps: Dict[TP, int]) -> bool: ...
    
    async def position(self, tp: TP) -> int: ...
    
    async def seek_to_beginning(self, *partitions: _TopicPartition) -> None: ...
    
    async def seek_wait(self, partitions: Dict[_TopicPartition, int]) -> None: ...
    
    async def _seek_wait(self, consumer: _Consumer, partitions: Dict[_TopicPartition, int]) -> None: ...
    
    def seek(self, partition: _TopicPartition, offset: int) -> None: ...
    
    def assignment(self) -> Set[TP]: ...
    
    def highwater(self, tp: TP) -> int: ...
    
    def topic_partitions(self, topic: str) -> Optional[int]: ...
    
    async def earliest_offsets(self, *partitions: TP) -> Dict[TP, int]: ...
    
    async def _earliest_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    
    async def highwaters(self, *partitions: TP) -> Dict[TP, int]: ...
    
    async def _highwaters(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    
    def _ensure_consumer(self) -> _Consumer: ...
    
    async def getmany(
        self,
        active_partitions: Optional[Set[TP]],
        timeout: float,
    ) -> Dict[TP, List[_Message]]: ...
    
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: Seconds = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None: ...
    
    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = None) -> TP: ...

class ProducerProduceFuture(asyncio.Future[RecordMetadata]):
    def set_from_on_delivery(self, err: Optional[KafkaException], msg: _Message) -> None: ...
    
    def message_to_metadata(self, message: _Message) -> RecordMetadata: ...

class ProducerThread(QueueServiceThread):
    _producer: Optional[_Producer] = ...
    _flush_soon: Optional[asyncio.Future[None]] = ...
    
    def __init__(self, producer: "Producer", **kwargs: Any) -> None: ...
    
    async def on_start(self) -> None: ...
    
    async def flush(self) -> None: ...
    
    async def on_thread_stop(self) -> None: ...
    
    def produce(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        on_delivery: Callable[[Optional[KafkaException], _Message], None],
    ) -> None: ...
    
    @Service.task
    async def _background_flush(self) -> None: ...

class Producer(base.Producer):
    logger: ClassVar[Any] = ...
    _quick_produce: Optional[Callable[..., None]] = ...
    
    def __post_init__(self) -> None: ...
    
    async def _on_irrecoverable_error(self, exc: BaseException) -> None: ...
    
    async def on_restart(self) -> None: ...
    
    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: Seconds = 20.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
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
        transactional_id: Optional[str] = None,
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
        transactional_id: Optional[str] = None,
    ) -> RecordMetadata: ...
    
    async def flush(self) -> None: ...
    
    def key_partition(self, topic: str, key: bytes) -> TP: ...

class Transport(base.Transport):
    Consumer: ClassVar[Type[Consumer]] = ...
    Producer: ClassVar[Type[Producer]] = ...
    default_port: ClassVar[int] = ...
    driver_version: ClassVar[str] = ...
    
    def _topic_config(
        self,
        retention: Optional[int] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> Dict[str, Any]: ...