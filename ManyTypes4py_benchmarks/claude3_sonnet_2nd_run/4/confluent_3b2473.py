"""Message transport using :pypi:`confluent_kafka`."""
import asyncio
import typing
from collections import defaultdict
from typing import Any, Awaitable, Callable, ClassVar, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Type, Union, cast
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
if typing.TYPE_CHECKING:
    from confluent_kafka import Consumer as _Consumer
    from confluent_kafka import Producer as _Producer
    from confluent_kafka import Message as _Message
else:

    class _Consumer:
        ...

    class _Producer:
        ...

    class _Message:
        ...
__all__ = ['Consumer', 'Producer', 'Transport']
logger = get_logger(__name__)

def server_list(urls: List[URL], default_port: int) -> str:
    default_host = '127.0.0.1'
    return ','.join([f'{u.host or default_host}:{u.port or default_port}' for u in urls])

class Consumer(ThreadDelegateConsumer):
    """Kafka consumer using :pypi:`confluent_kafka`."""
    logger = logger

    def _new_consumer_thread(self) -> ConsumerThread:
        return ConfluentConsumerThread(self, loop=self.loop, beacon=self.beacon)

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]] = None, timeout: float = 30.0, retention: Optional[float] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        """Create topic on broker."""
        return
        await self._thread.create_topic(topic, partitions, replication, config=config, timeout=int(want_seconds(timeout) * 1000.0), retention=int(want_seconds(retention) * 1000.0), compacting=compacting, deleting=deleting, ensure_created=ensure_created)

    def _to_message(self, tp: TP, record: _Message) -> ConsumerMessage:
        timestamp_type, timestamp = record.timestamp()
        timestamp_s = cast(float, None)
        if timestamp is not None:
            timestamp_s = timestamp / 1000.0
        key = record.key()
        key_size = len(key) if key is not None else 0
        value = record.value()
        value_size = len(value) if value is not None else 0
        return ConsumerMessage(record.topic(), record.partition(), record.offset(), timestamp_s, timestamp_type, [], key, value, None, key_size, value_size, tp)

    def _new_topicpartition(self, topic: str, partition: int) -> TP:
        return cast(TP, _TopicPartition(topic, partition))

class ConfluentConsumerThread(ConsumerThread):
    """Thread managing underlying :pypi:`confluent_kafka` consumer."""
    _consumer: Optional[_Consumer] = None
    _assigned: bool = False

    async def on_start(self) -> None:
        self._consumer = self._create_consumer(loop=self.thread_loop)

    def _create_consumer(self, loop: asyncio.AbstractEventLoop) -> _Consumer:
        transport = cast(Transport, self.transport)
        if self.app.client_only:
            return self._create_client_consumer(transport, loop=loop)
        else:
            return self._create_worker_consumer(transport, loop=loop)

    def _create_worker_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> _Consumer:
        conf = self.app.conf
        self._assignor = self.app.assignor
        return confluent_kafka.Consumer({'bootstrap.servers': server_list(transport.url, transport.default_port), 'group.id': conf.id, 'client.id': conf.broker_client_id, 'default.topic.config': {'auto.offset.reset': 'earliest'}, 'enable.auto.commit': False, 'fetch.max.bytes': conf.consumer_max_fetch_size, 'request.timeout.ms': int(conf.broker_request_timeout * 1000.0), 'check.crcs': conf.broker_check_crcs, 'session.timeout.ms': int(conf.broker_session_timeout * 1000.0), 'heartbeat.interval.ms': int(conf.broker_heartbeat_interval * 1000.0)})

    def _create_client_consumer(self, transport: 'Transport', loop: asyncio.AbstractEventLoop) -> _Consumer:
        conf = self.app.conf
        return confluent_kafka.Consumer({'bootstrap.servers': server_list(transport.url, transport.default_port), 'client.id': conf.broker_client_id, 'enable.auto.commit': True, 'default.topic.config': {'auto.offset.reset': 'earliest'}})

    def close(self) -> None:
        ...

    async def subscribe(self, topics: Iterable[str]) -> None:
        await self.call_thread(self._ensure_consumer().subscribe, topics=list(topics), on_assign=self._on_assign, on_revoke=self._on_revoke)
        while not self._assigned:
            self.log.info('Still waiting for assignment...')
            self._ensure_consumer().poll(timeout=1)

    def _on_assign(self, consumer: _Consumer, assigned: List[_TopicPartition]) -> None:
        self._assigned = True
        self.thread_loop.run_until_complete(self.on_partitions_assigned({TP(tp.topic, tp.partition) for tp in assigned}))

    def _on_revoke(self, consumer: _Consumer, revoked: List[_TopicPartition]) -> None:
        self.thread_loop.run_until_complete(self.on_partitions_revoked({TP(tp.topic, tp.partition) for tp in revoked}))

    async def seek_to_committed(self) -> Mapping[TP, int]:
        return await self.call_thread(self._seek_to_committed)

    async def _seek_to_committed(self) -> Mapping[TP, int]:
        consumer = self._ensure_consumer()
        assignment = consumer.assignment()
        committed = consumer.committed(assignment)
        for tp in committed:
            consumer.seek(tp)
        return {ensure_TP(tp): tp.offset for tp in committed}

    async def _committed_offsets(self, partitions: Set[TP]) -> Dict[TP, int]:
        consumer = self._ensure_consumer()
        committed = consumer.committed([_TopicPartition(tp[0], tp[1]) for tp in partitions])
        return {TP(tp.topic, tp.partition): tp.offset for tp in committed}

    async def commit(self, tps: Mapping[TP, int]) -> bool:
        self.call_thread(self._ensure_consumer().commit, offsets=[_TopicPartition(tp.topic, tp.partition, offset=offset) for tp, offset in tps.items()], asynchronous=False)
        return True

    async def position(self, tp: TP) -> Optional[int]:
        return await self.call_thread(self._ensure_consumer().position, tp)

    async def seek_to_beginning(self, *partitions: TP) -> None:
        await self.call_thread(self._ensure_consumer().seek_to_beginning, *partitions)

    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        consumer = self._ensure_consumer()
        await self.call_thread(self._seek_wait, consumer, partitions)

    async def _seek_wait(self, consumer: _Consumer, partitions: Mapping[TP, int]) -> None:
        for tp, offset in partitions.items():
            self.log.dev('SEEK %r -> %r', tp, offset)
            consumer.seek(tp, offset)
        await asyncio.gather(*[consumer.position(tp) for tp in partitions])

    def seek(self, partition: TP, offset: int) -> None:
        self._ensure_consumer().seek(partition, offset)

    def assignment(self) -> Set[TP]:
        return ensure_TPset(self._ensure_consumer().assignment())

    def highwater(self, tp: TP) -> int:
        _, hw = self._ensure_consumer().get_watermark_offsets(_TopicPartition(tp.topic, tp.partition), cached=True)
        return hw

    def topic_partitions(self, topic: str) -> Optional[int]:
        return None

    async def earliest_offsets(self, *partitions: TP) -> MutableMapping[TP, int]:
        if not partitions:
            return {}
        return await self.call_thread(self._earliest_offsets, partitions)

    async def _earliest_offsets(self, partitions: List[TP]) -> Dict[TP, int]:
        consumer = self._ensure_consumer()
        return {tp: consumer.get_watermark_offsets(_TopicPartition(tp[0], tp[1]))[0] for tp in partitions}

    async def highwaters(self, *partitions: TP) -> MutableMapping[TP, int]:
        if not partitions:
            return {}
        return await self.call_thread(self._highwaters, partitions)

    async def _highwaters(self, partitions: List[TP]) -> Dict[TP, int]:
        consumer = self._ensure_consumer()
        return {tp: consumer.get_watermark_offsets(_TopicPartition(tp[0], tp[1]))[1] for tp in partitions}

    def _ensure_consumer(self) -> _Consumer:
        if self._consumer is None:
            raise ConsumerNotStarted('Consumer thread not yet started')
        return self._consumer

    async def getmany(self, active_partitions: Set[TP], timeout: float) -> RecordMap:
        _consumer = self._ensure_consumer()
        messages = await self.call_thread(_consumer.consume, num_messages=10000, timeout=timeout)
        records: Dict[TP, List[_Message]] = defaultdict(list)
        for message in messages:
            tp = TP(message.topic(), message.partition())
            records[tp].append(message)
        return records

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]] = None, timeout: float = 30.0, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        return

    def key_partition(self, topic: str, key: bytes, partition: Optional[int] = None) -> int:
        raise NotImplementedError('TODO')

class ProducerProduceFuture(asyncio.Future):

    def set_from_on_delivery(self, err: Optional[Exception], msg: _Message) -> None:
        if err:
            self.set_exception(err)
        else:
            metadata = self.message_to_metadata(msg)
            self.set_result(metadata)

    def message_to_metadata(self, message: _Message) -> RecordMetadata:
        topic, partition = tp = TP(message.topic(), message.partition())
        return RecordMetadata(topic, partition, tp, message.offset())

class ProducerThread(QueueServiceThread):
    """Thread managing underlying :pypi:`confluent_kafka` producer."""
    _producer: Optional[_Producer] = None
    _flush_soon: Optional[asyncio.Future] = None

    def __init__(self, producer: 'Producer', **kwargs: Any) -> None:
        self.producer = producer
        self.transport = cast(Transport, self.producer.transport)
        self.app = self.transport.app
        super().__init__(**kwargs)

    async def on_start(self) -> None:
        self._producer = confluent_kafka.Producer({'bootstrap.servers': server_list(self.transport.url, self.transport.default_port), 'client.id': self.app.conf.broker_client_id, 'max.in.flight.requests.per.connection': 1})

    async def flush(self) -> None:
        if self._producer is not None:
            self._producer.flush()

    async def on_thread_stop(self) -> None:
        if self._producer is not None:
            self._producer.flush()

    def produce(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], on_delivery: Callable[[Optional[Exception], _Message], None], timestamp: Optional[int] = None) -> None:
        if self._producer is None:
            raise RuntimeError('Producer not started')
        if partition is not None:
            self._producer.produce(topic, key, value, partition, on_delivery=on_delivery)
        else:
            self._producer.produce(topic, key, value, on_delivery=on_delivery)
        notify(self._flush_soon)

    @Service.task
    async def _background_flush(self) -> None:
        producer = cast(_Producer, self._producer)
        _size = producer.__len__
        _flush = producer.flush
        _poll = producer.poll
        _sleep = self.sleep
        _create_future = self.loop.create_future
        while not self.should_stop:
            if not _size():
                flush_soon = self._flush_soon
                if flush_soon is None:
                    flush_soon = self._flush_soon = _create_future()
                stopped = False
                try:
                    stopped = await self.wait_for_stopped(flush_soon, timeout=1.0)
                finally:
                    self._flush_soon = None
                if not stopped:
                    _flush(timeout=100)
                    _poll(timeout=1)
                    await _sleep(0)

class Producer(base.Producer):
    """Kafka producer using :pypi:`confluent_kafka`."""
    logger = logger
    _quick_produce: Optional[Callable] = None

    def __post_init__(self) -> None:
        self._producer_thread = ProducerThread(self, loop=self.loop, beacon=self.beacon)
        self._quick_produce = self._producer_thread.produce

    async def _on_irrecoverable_error(self, exc: BaseException) -> None:
        consumer = self.transport.app.consumer
        if consumer is not None:
            await consumer.crash(exc)
        await self.crash(exc)

    async def on_restart(self) -> None:
        """Call when producer is restarting."""
        self.on_init()

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]] = None, timeout: float = 20.0, retention: Optional[float] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        """Create topic on broker."""
        return
        _retention = int(want_seconds(retention) * 1000.0) if retention else None
        await cast(Transport, self.transport)._create_topic(self, self._producer.client, topic, partitions, replication, config=config, timeout=int(want_seconds(timeout) * 1000.0), retention=_retention, compacting=compacting, deleting=deleting, ensure_created=ensure_created)

    async def on_start(self) -> None:
        """Call when producer is starting."""
        await self._producer_thread.start()
        await self.sleep(0.5)

    async def on_stop(self) -> None:
        """Call when producer is stopping."""
        await self._producer_thread.stop()

    async def send(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str] = None) -> Awaitable[RecordMetadata]:
        """Send message for future delivery."""
        fut = ProducerProduceFuture(loop=self.loop)
        self._quick_produce(topic, value, key, partition, timestamp=int(timestamp * 1000) if timestamp else timestamp, on_delivery=fut.set_from_on_delivery)
        return cast(Awaitable[RecordMetadata], fut)
        try:
            return cast(Awaitable[RecordMetadata], await self._producer.send(topic, value, key=key, partition=partition))
        except KafkaException as exc:
            raise ProducerSendError(f'Error while sending: {exc!r}') from exc

    async def send_and_wait(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str] = None) -> RecordMetadata:
        """Send message and wait for it to be delivered to broker(s)."""
        fut = await self.send(topic, key, value, partition, timestamp, headers)
        return await fut

    async def flush(self) -> None:
        """Flush producer buffer.

        This will wait until the producer has written
        all buffered up messages to any connected brokers.
        """
        await self._producer_thread.flush()

    def key_partition(self, topic: str, key: bytes) -> Tuple[str, int]:
        """Return topic and partition destination for key."""
        raise NotImplementedError()

class Transport(base.Transport):
    """Kafka transport using :pypi:`confluent_kafka`."""
    Consumer: Type[Consumer] = Consumer
    Producer: Type[Producer] = Producer
    default_port: int = 9092
    driver_version: str = f'confluent_kafka={confluent_kafka.__version__}'

    def _topic_config(self, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        cleanup_flags: Set[str] = set()
        if compacting:
            cleanup_flags |= {'compact'}
        if deleting:
            cleanup_flags |= {'delete'}
        if cleanup_flags:
            config['cleanup.policy'] = ','.join(sorted(cleanup_flags))
        if retention:
            config['retention.ms'] = retention
        return config
