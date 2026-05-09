"""Message transport using :pypi:`confluent_kafka`."""
from confluent_kafka import (
    Consumer as _Consumer,
    KafkaException,
    Message as _Message,
    Producer as _Producer,
    TopicPartition as _TopicPartition,
)
from faust.types import AppT, ConsumerMessage, HeadersArg, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT
from mode import Service
from mode.threads import QueueServiceThread
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
)
from yarl import URL

__all__ = ['Consumer', 'Producer', 'Transport']
logger: get_logger

def server_list(urls: Iterable[URL], default_port: int) -> str: ...

class Consumer(ConsumerT, ThreadDelegateConsumer):
    logger: ClassVar[Logger] = logger

    def _new_consumer_thread(self) -> ConfluentConsumerThread: ...
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
    ) -> None: ...
    def _to_message(self, tp: TP, record: _Message) -> ConsumerMessage: ...
    def _new_topicpartition(self, topic: str, partition: int) -> TP: ...

class ConfluentConsumerThread(ConsumerThread):
    _consumer: Optional[_Consumer] = None
    _assigned: bool = False

    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    def _create_worker_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    def _create_client_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    def _on_assign(self, consumer: _Consumer, assigned: Iterable[_TopicPartition]) -> None: ...
    def _on_revoke(self, consumer: _Consumer, revoked: Iterable[_TopicPartition]) -> None: ...
    async def seek_to_committed(self) -> Dict[TP, int]: ...
    async def _seek_to_committed(self) -> Dict[TP, int]: ...
    async def _committed_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    async def commit(self, tps: Dict[TP, int]) -> bool: ...
    async def position(self, tp: TP) -> int: ...
    async def seek_to_beginning(self, *partitions: _TopicPartition) -> None: ...
    async def seek_wait(self, partitions: Dict[TP, int]) -> None: ...
    async def _seek_wait(self, consumer: _Consumer, partitions: Dict[TP, int]) -> None: ...
    def seek(self, partition: _TopicPartition, offset: int) -> None: ...
    def assignment(self) -> Set[TP]: ...
    def highwater(self, tp: TP) -> int: ...
    async def earliest_offsets(self, *partitions: TP) -> Dict[TP, int]: ...
    async def _earliest_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    async def highwaters(self, *partitions: TP) -> Dict[TP, int]: ...
    async def _highwaters(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    def _ensure_consumer(self) -> _Consumer: ...
    async def getmany(
        self,
        active_partitions: Iterable[TP],
        timeout: Seconds,
    ) -> RecordMap: ...
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
    ) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int]) -> TP: ...

class ProducerProduceFuture(asyncio.Future):
    def set_from_on_delivery(self, err: Optional[KafkaException], msg: Optional[_Message]) -> None: ...
    def message_to_metadata(self, message: _Message) -> RecordMetadata: ...

class ProducerThread(QueueServiceThread):
    _producer: Optional[_Producer] = None
    _flush_soon: Optional[asyncio.Future] = None

    def produce(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        on_delivery: Callable[[Optional[KafkaException], Optional[_Message]], None],
    ) -> None: ...
    async def _background_flush(self) -> None: ...

class Producer(ProducerT, base.Producer):
    logger: ClassVar[Logger] = logger
    _quick_produce: Optional[Callable] = None

    async def _on_irrecoverable_error(self, exc: BaseException) -> None: ...
    async def on_restart(self) -> None: ...
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
        headers: HeadersArg,
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
        headers: HeadersArg,
        *,
        transactional_id: Optional[str] = None,
    ) -> RecordMetadata: ...
    async def flush(self) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes]) -> TP: ...

class Transport(base.Transport):
    Consumer: ClassVar[Type[Consumer]] = Consumer
    Producer: ClassVar[Type[Producer]] = Producer
    default_port: ClassVar[int] = 9092
    driver_version: ClassVar[str] = 'confluent_kafka=...'

    def _topic_config(
        self,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> Dict[str, Any]: ...