"""Message transport using :pypi:`confluent_kafka`."""

import asyncio
import typing
from collections import defaultdict
from typing import Any, Awaitable, Callable, ClassVar, Iterable, List, Mapping, MutableMapping, Optional, Set, Type, cast
from mode import Service, get_logger
from mode.threads import QueueServiceThread
from mode.utils.futures import notify
from mode.utils.times import Seconds, want_seconds
from yarl import URL
from faust.exceptions import ConsumerNotStarted, ProducerSendError
from faust.transport import base
from faust.transport.consumer import ConsumerThread, RecordMap, ThreadDelegateConsumer, ensure_TP, ensure_TPset
from faust.types import AppT, ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT
import confluent_kafka
from confluent_kafka import TopicPartition as _TopicPartition
from confluent_kafka import KafkaException

__all__: List[str] = ['Consumer', 'Producer', 'Transport']
logger: ClassVar[get_logger] = get_logger(__name__)

def server_list(urls: Iterable[URL], default_port: int) -> str:
    ...

class Consumer(ThreadDelegateConsumer):
    logger: ClassVar[get_logger] = logger

    def _new_consumer_thread(self) -> ConfluentConsumerThread:
        ...

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[MutableMapping[str, Any]] = None, timeout: Seconds = 30.0, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    def _to_message(self, tp: TP, record: confluent_kafka.Message) -> ConsumerMessage:
        ...

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        ...

class ConfluentConsumerThread(ConsumerThread):
    _consumer: Optional[confluent_kafka.Consumer] = None
    _assigned: bool = False

    async def on_start(self) -> None:
        ...

    def _create_consumer(self, loop: asyncio.events.AbstractEventLoop) -> confluent_kafka.Consumer:
        ...

    def _create_worker_consumer(self, transport: 'Transport', loop: asyncio.events.AbstractEventLoop) -> confluent_kafka.Consumer:
        ...

    def _create_client_consumer(self, transport: 'Transport', loop: asyncio.events.AbstractEventLoop) -> confluent_kafka.Consumer:
        ...

    def close(self) -> None:
        ...

    async def subscribe(self, topics: Iterable[str]) -> None:
        ...

    def _on_assign(self, consumer: confluent_kafka.Consumer, assigned: Iterable[_TopicPartition]) -> None:
        ...

    def _on_revoke(self, consumer: confluent_kafka.Consumer, revoked: Iterable[_TopicPartition]) -> None:
        ...

    async def seek_to_committed(self) -> dict[TP, int]:
        ...

    async def _seek_to_committed(self) -> dict[TP, int]:
        ...

    async def _committed_offsets(self, partitions: Iterable[TP]) -> dict[TP, int]:
        ...

    async def commit(self, tps: Mapping[TP, int]) -> bool:
        ...

    async def position(self, tp: TP) -> int:
        ...

    async def seek_to_beginning(self, *partitions: _TopicPartition) -> None:
        ...

    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        ...

    async def _seek_wait(self, consumer: confluent_kafka.Consumer, partitions: Mapping[TP, int]) -> None:
        ...

    def seek(self, partition: _TopicPartition, offset: int) -> None:
        ...

    def assignment(self) -> Set[TP]:
        ...

    def highwater(self, tp: TP) -> int:
        ...

    def topic_partitions(self, topic: str) -> None:
        ...

    async def earliest_offsets(self, *partitions: TP) -> dict[TP, int]:
        ...

    async def _earliest_offsets(self, partitions: Iterable[TP]) -> dict[TP, int]:
        ...

    async def highwaters(self, *partitions: TP) -> dict[TP, int]:
        ...

    async def _highwaters(self, partitions: Iterable[TP]) -> dict[TP, int]:
        ...

    def _ensure_consumer(self) -> confluent_kafka.Consumer:
        ...

    async def getmany(self, active_partitions: Iterable[TP], timeout: Seconds) -> RecordMap:
        ...

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[MutableMapping[str, Any]] = None, timeout: Seconds = 30.0, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = None) -> None:
        ...

class ProducerProduceFuture(asyncio.Future):
    def set_from_on_delivery(self, err: Optional[KafkaException], msg: confluent_kafka.Message) -> None:
        ...

    def message_to_metadata(self, message: confluent_kafka.Message) -> RecordMetadata:
        ...

class ProducerThread(QueueServiceThread):
    _producer: Optional[confluent_kafka.Producer] = None
    _flush_soon: Optional[asyncio.Future] = None

    def __init__(self, producer: 'Producer', **kwargs: Any) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def flush(self) -> None:
        ...

    async def on_thread_stop(self) -> None:
        ...

    def produce(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], on_delivery: Callable[[Optional[KafkaException], confluent_kafka.Message], None]) -> None:
        ...

    @Service.task
    async def _background_flush(self) -> None:
        ...

class Producer(base.Producer):
    logger: ClassVar[get_logger] = logger
    _quick_produce: Optional[Callable[..., None]] = None

    def __post_init__(self) -> None:
        ...

    async def _on_irrecoverable_error(self, exc: Exception) -> None:
        ...

    async def on_restart(self) -> None:
        ...

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[MutableMapping[str, Any]] = None, timeout: Seconds = 20.0, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def on_stop(self) -> None:
        ...

    async def send(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: HeadersArg) -> Awaitable[RecordMetadata]:
        ...

    async def send_and_wait(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: HeadersArg) -> RecordMetadata:
        ...

    async def flush(self) -> None:
        ...

    def key_partition(self, topic: str, key: Optional[bytes]) -> None:
        ...

class Transport(base.Transport):
    Consumer = Consumer
    Producer = Producer
    default_port: int = 9092
    driver_version: str = f'confluent_kafka={confluent_kafka.__version__}'

    def _topic_config(self, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None) -> dict[str, Any]:
        ...