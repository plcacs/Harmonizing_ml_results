from __future__ import annotations

import asyncio
from logging import Logger
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Set

from confluent_kafka import Consumer as _Consumer
from confluent_kafka import Message as _Message
from confluent_kafka import Producer as _Producer
from confluent_kafka import TopicPartition as _TopicPartition
from faust.transport import base
from faust.transport.consumer import ConsumerThread, RecordMap, ThreadDelegateConsumer
from faust.types import AppT, ConsumerMessage, HeadersArg, RecordMetadata, TP
from mode.threads import QueueServiceThread
from mode.utils.times import Seconds
from yarl import URL

__all__: List[str] = ...
logger: Logger = ...


def server_list(urls: Iterable[URL], default_port: int) -> str: ...


class Consumer(ThreadDelegateConsumer):
    """Kafka consumer using confluent_kafka."""
    logger: ClassVar[Logger]

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
    """Thread managing underlying confluent_kafka consumer."""
    _consumer: Optional[_Consumer]
    _assigned: bool

    async def on_start(self) -> None: ...
    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    def _create_worker_consumer(self, transport: Transport, loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    def _create_client_consumer(self, transport: Transport, loop: asyncio.AbstractEventLoop) -> _Consumer: ...
    def close(self) -> None: ...
    async def subscribe(self, topics: Iterable[str]) -> None: ...
    def _on_assign(self, consumer: _Consumer, assigned: Iterable[_TopicPartition]) -> None: ...
    def _on_revoke(self, consumer: _Consumer, revoked: Iterable[_TopicPartition]) -> None: ...
    async def seek_to_committed(self) -> Dict[TP, int]: ...
    async def _seek_to_committed(self) -> Dict[TP, int]: ...
    async def _committed_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    async def commit(self, tps: Mapping[TP, int]) -> bool: ...
    async def position(self, tp: TP) -> Any: ...
    async def seek_to_beginning(self, *partitions: _TopicPartition) -> None: ...
    async def seek_wait(self, partitions: Mapping[_TopicPartition, int]) -> None: ...
    async def _seek_wait(self, consumer: _Consumer, partitions: Mapping[_TopicPartition, int]) -> None: ...
    def seek(self, partition: _TopicPartition, offset: int) -> None: ...
    def assignment(self) -> Set[TP]: ...
    def highwater(self, tp: TP) -> int: ...
    def topic_partitions(self, topic: str) -> None: ...
    async def earliest_offsets(self, *partitions: TP) -> Dict[TP, int]: ...
    async def _earliest_offsets(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    async def highwaters(self, *partitions: TP) -> Dict[TP, int]: ...
    async def _highwaters(self, partitions: Iterable[TP]) -> Dict[TP, int]: ...
    def _ensure_consumer(self) -> _Consumer: ...
    async def getmany(self, active_partitions: Set[TP], timeout: Seconds) -> RecordMap: ...
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
    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = ...) -> int: ...


class ProducerProduceFuture(asyncio.Future[RecordMetadata]):
    def set_from_on_delivery(self, err: Any, msg: _Message) -> None: ...
    def message_to_metadata(self, message: _Message) -> RecordMetadata: ...


class ProducerThread(QueueServiceThread):
    """Thread managing underlying confluent_kafka producer."""
    producer: Producer
    transport: Transport
    app: AppT
    _producer: Optional[_Producer]
    _flush_soon: Optional[asyncio.Future[Any]]

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
        on_delivery: Callable[[Any, _Message], None],
    ) -> None: ...
    async def _background_flush(self) -> None: ...


class Producer(base.Producer):
    """Kafka producer using confluent_kafka."""
    logger: ClassVar[Logger]
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
        headers: HeadersArg,
        *,
        transactional_id: Optional[str] = ...,
    ) -> asyncio.Future[RecordMetadata]: ...
    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: HeadersArg,
        *,
        transactional_id: Optional[str] = ...,
    ) -> RecordMetadata: ...
    async def flush(self) -> None: ...
    def key_partition(self, topic: str, key: Optional[bytes]) -> int: ...


class Transport(base.Transport):
    """Kafka transport using confluent_kafka."""
    Consumer: type[Consumer]
    Producer: type[Producer]
    default_port: ClassVar[int]
    driver_version: ClassVar[str]

    def _topic_config(
        self,
        retention: Optional[int] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
    ) -> Dict[str, Any]: ...