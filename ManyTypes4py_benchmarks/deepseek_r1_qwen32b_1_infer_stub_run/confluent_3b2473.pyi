"""Message transport using :pypi:`confluent_kafka`."""

import asyncio
import typing
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Type,
)
from mode import Service
from mode.threads import QueueServiceThread
from mode.utils.times import Seconds
from yarl import URL
from faust.exceptions import ConsumerNotStarted, ProducerSendError
from faust.types import (
    AppT,
    ConsumerMessage,
    HeadersArg,
    RecordMetadata,
    TP,
)
from faust.types.transports import ConsumerT, ProducerT
from confluent_kafka import (
    KafkaException,
    _Consumer,
    _Producer,
    _Message,
    _TopicPartition,
)

__all__ = ['Consumer', 'Producer', 'Transport']

logger = get_logger(__name__)


def server_list(urls, default_port) -> str:
    ...


class Consumer(ThreadDelegateConsumer):
    """Kafka consumer using :pypi:`confluent_kafka`."""
    logger: ClassVar[logging.Logger] = logger

    def _new_consumer_thread(self) -> ConfluentConsumerThread:
        ...

    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[MutableMapping[str, Any]] = None,
        timeout: Seconds = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None:
        ...

    def _to_message(self, tp: TP, record: _Message) -> ConsumerMessage:
        ...

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        ...


class ConfluentConsumerThread(ConsumerThread):
    """Thread managing underlying :pypi:`confluent_kafka` consumer."""
    _consumer: Optional[_Consumer] = None
    _assigned: bool = False

    async def on_start(self) -> None:
        ...

    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> _Consumer:
        ...

    def _create_worker_consumer(
        self,
        transport: 'Transport',
        loop: asyncio.AbstractEventLoop,
    ) -> _Consumer:
        ...

    def _create_client_consumer(
        self,
        transport: 'Transport',
        loop: asyncio.AbstractEventLoop,
    ) -> _Consumer:
        ...

    async def close(self) -> None:
        ...

    async def subscribe(self, topics: Iterable[str]) -> None:
        ...

    def _on_assign(
        self,
        consumer: _Consumer,
        assigned: List[_TopicPartition],
    ) -> None:
        ...

    def _on_revoke(
        self,
        consumer: _Consumer,
        revoked: List[_TopicPartition],
    ) -> None:
        ...

    async def seek_to_committed(self) -> Mapping[TP, int]:
        ...

    async def _committed_offsets(self, partitions: Iterable[TP]) -> Mapping[TP, int]:
        ...

    async def commit(self, tps: Mapping[TP, int]) -> bool:
        ...

    async def position(self, tp: TP) -> int:
        ...

    async def seek_to_beginning(self, *partitions: TP) -> None:
        ...

    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        ...

    def seek(self, partition: TP, offset: int) -> None:
        ...

    def assignment(self) -> Set[TP]:
        ...

    def highwater(self, tp: TP) -> int:
        ...

    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]:
        ...

    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]:
        ...

    def _ensure_consumer(self) -> _Consumer:
        ...

    async def getmany(
        self,
        active_partitions: Set[TP],
        timeout: Seconds,
    ) -> RecordMap:
        ...

    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[MutableMapping[str, Any]] = None,
        timeout: Seconds = 30.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None:
        ...

    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int]) -> TP:
        ...


class ProducerProduceFuture(asyncio.Future[RecordMetadata]):
    def set_from_on_delivery(self, err: Optional[KafkaException], msg: _Message) -> None:
        ...

    def message_to_metadata(self, message: _Message) -> RecordMetadata:
        ...


class ProducerThread(QueueServiceThread):
    """Thread managing underlying :pypi:`confluent_kafka` producer."""
    _producer: Optional[_Producer] = None
    _flush_soon: Optional[asyncio.Future[None]] = None

    def __init__(self, producer: 'Producer', **kwargs: Any) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def flush(self) -> None:
        ...

    async def on_thread_stop(self) -> None:
        ...

    def produce(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        on_delivery: Callable[[Optional[KafkaException], _Message], None],
    ) -> None:
        ...

    @Service.task
    async def _background_flush(self) -> None:
        ...


class Producer(base.Producer):
    """Kafka producer using :pypi:`confluent_kafka`."""
    logger: ClassVar[logging.Logger] = logger
    _quick_produce: Optional[Callable[..., None]] = None

    def __post_init__(self) -> None:
        ...

    async def _on_irrecoverable_error(self, exc: BaseException) -> None:
        ...

    async def on_restart(self) -> None:
        ...

    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[MutableMapping[str, Any]] = None,
        timeout: Seconds = 20.0,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False,
    ) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def on_stop(self) -> None:
        ...

    async def send(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: HeadersArg,
        *,
        transactional_id: Optional[str] = None,
    ) -> Awaitable[RecordMetadata]:
        ...

    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: HeadersArg,
        *,
        transactional_id: Optional[str] = None,
    ) -> RecordMetadata:
        ...

    async def flush(self) -> None:
        ...

    def key_partition(self, topic: str, key: Optional[bytes]) -> TP:
        ...


class Transport(base.Transport):
    """Kafka transport using :pypi:`confluent_kafka`."""
    Consumer = Consumer
    Producer = Producer
    default_port: int = 9092
    driver_version: str = f'confluent_kafka={confluent_kafka.__version__}'

    def _topic_config(
        self,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> MutableMapping[str, Any]:
        ...