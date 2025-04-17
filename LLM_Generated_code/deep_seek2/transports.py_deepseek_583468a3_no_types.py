import abc
import asyncio
import ssl
import typing
from typing import AbstractSet, Any, AsyncIterator, Awaitable, Callable, ClassVar, Iterable, Iterator, List, Mapping, MutableSet, Optional, Sequence, Set, Tuple, Type, Union, no_type_check
from mode import Seconds, ServiceT
from yarl import URL
from .core import HeadersArg
from .topics import TopicT
from .tuples import FutureMessage, Message, RecordMetadata, TP
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
else:

    class _AppT:
        ...
__all__ = ['ConsumerCallback', 'TPorTopicSet', 'PartitionsRevokedCallback', 'PartitionsAssignedCallback', 'PartitionerT', 'ConsumerT', 'ProducerT', 'ConductorT', 'TransactionManagerT', 'TransportT']
ConsumerCallback = Callable[[Message], Awaitable[None]]
TPorTopic = Union[str, TP]
TPorTopicSet = AbstractSet[TPorTopic]
PartitionsRevokedCallback = Callable[[Set[TP]], Awaitable[None]]
PartitionsAssignedCallback = Callable[[Set[TP]], Awaitable[None]]
PartitionerT = Callable[[Optional[bytes], Sequence[int], Sequence[int]], int]

class ProducerBufferT(ServiceT):
    max_messages: int
    pending: asyncio.Queue[FutureMessage]

    @abc.abstractmethod
    def put(self, fut):
        ...

    @abc.abstractmethod
    async def flush(self) -> None:
        ...

    @abc.abstractmethod
    async def flush_atmost(self, n: int) -> int:
        ...

    @abc.abstractmethod
    async def wait_until_ebb(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def size(self):
        ...

class ProducerT(ServiceT):
    transport: 'TransportT'
    buffer: ProducerBufferT
    client_id: str
    linger_ms: int
    max_batch_size: int
    acks: int
    max_request_size: int
    compression_type: Optional[str]
    ssl_context: Optional[ssl.SSLContext]
    partitioner: Optional[PartitionerT]
    request_timeout: float

    @abc.abstractmethod
    def __init__(self, transport, loop=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    async def send(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str]=None) -> Awaitable[RecordMetadata]:
        ...

    @abc.abstractmethod
    def send_soon(self, fut):
        ...

    @abc.abstractmethod
    async def send_and_wait(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str]=None) -> RecordMetadata:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]]=None, timeout: Seconds=1000.0, retention: Optional[Seconds]=None, compacting: Optional[bool]=None, deleting: Optional[bool]=None, ensure_created: bool=False) -> None:
        ...

    @abc.abstractmethod
    def key_partition(self, topic, key):
        ...

    @abc.abstractmethod
    async def flush(self) -> None:
        ...

    @abc.abstractmethod
    async def begin_transaction(self, transactional_id: str) -> None:
        ...

    @abc.abstractmethod
    async def commit_transaction(self, transactional_id: str) -> None:
        ...

    @abc.abstractmethod
    async def abort_transaction(self, transactional_id: str) -> None:
        ...

    @abc.abstractmethod
    async def stop_transaction(self, transactional_id: str) -> None:
        ...

    @abc.abstractmethod
    async def maybe_begin_transaction(self, transactional_id: str) -> None:
        ...

    @abc.abstractmethod
    async def commit_transactions(self, tid_to_offset_map: Mapping[str, Mapping[TP, int]], group_id: str, start_new_transaction: bool=True) -> None:
        ...

    @abc.abstractmethod
    def supports_headers(self):
        ...

class TransactionManagerT(ProducerT):
    consumer: 'ConsumerT'
    producer: 'ProducerT'

    @abc.abstractmethod
    def __init__(self, transport, loop=None, *, consumer: 'ConsumerT', producer: 'ProducerT', **kwargs: Any):
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, offsets: Mapping[TP, int], start_new_transaction: bool=True) -> bool:
        ...

    async def begin_transaction(self, transactional_id: str) -> None:
        raise NotImplementedError()

    async def commit_transaction(self, transactional_id: str) -> None:
        raise NotImplementedError()

    async def abort_transaction(self, transactional_id: str) -> None:
        raise NotImplementedError()

    async def stop_transaction(self, transactional_id: str) -> None:
        raise NotImplementedError()

    async def maybe_begin_transaction(self, transactional_id: str) -> None:
        raise NotImplementedError()

    async def commit_transactions(self, tid_to_offset_map: Mapping[str, Mapping[TP, int]], group_id: str, start_new_transaction: bool=True) -> None:
        raise NotImplementedError()

class SchedulingStrategyT:

    @abc.abstractmethod
    def __init__(self):
        ...

    @abc.abstractmethod
    def iterate(self, records):
        ...

class ConsumerT(ServiceT):
    transport: 'TransportT'
    transactions: TransactionManagerT
    commit_interval: float
    randomly_assigned_topics: Set[str]
    in_transaction: bool
    scheduler: SchedulingStrategyT

    @abc.abstractmethod
    def __init__(self, transport, callback, on_partitions_revoked, on_partitions_assigned, *, commit_interval: Optional[float]=None, loop: asyncio.AbstractEventLoop=None, **kwargs: Any):
        self._on_partitions_revoked: PartitionsRevokedCallback
        self._on_partitions_assigned: PartitionsAssignedCallback

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]]=None, timeout: Seconds=1000.0, retention: Optional[Seconds]=None, compacting: Optional[bool]=None, deleting: Optional[bool]=None, ensure_created: bool=False) -> None:
        ...

    @abc.abstractmethod
    async def subscribe(self, topics: Iterable[str]) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    async def getmany(self, timeout: float) -> AsyncIterator[Tuple[TP, Message]]:
        ...

    @abc.abstractmethod
    def track_message(self, message):
        ...

    @abc.abstractmethod
    async def perform_seek(self) -> None:
        ...

    @abc.abstractmethod
    def ack(self, message):
        ...

    @abc.abstractmethod
    async def wait_empty(self) -> None:
        ...

    @abc.abstractmethod
    def assignment(self):
        ...

    @abc.abstractmethod
    def highwater(self, tp):
        ...

    @abc.abstractmethod
    def stop_flow(self):
        ...

    @abc.abstractmethod
    def resume_flow(self):
        ...

    @abc.abstractmethod
    def pause_partitions(self, tps):
        ...

    @abc.abstractmethod
    def resume_partitions(self, tps):
        ...

    @abc.abstractmethod
    async def position(self, tp: TP) -> Optional[int]:
        ...

    @abc.abstractmethod
    async def seek(self, partition: TP, offset: int) -> None:
        ...

    @abc.abstractmethod
    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, topics: Optional[TPorTopicSet]=None, start_new_transaction: bool=True) -> bool:
        ...

    @abc.abstractmethod
    async def on_task_error(self, exc: BaseException) -> None:
        ...

    @abc.abstractmethod
    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]:
        ...

    @abc.abstractmethod
    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]:
        ...

    @abc.abstractmethod
    def topic_partitions(self, topic):
        ...

    @abc.abstractmethod
    def key_partition(self, topic, key, partition=None):
        ...

    @abc.abstractmethod
    def close(self):
        ...

    @abc.abstractmethod
    def verify_recovery_event_path(self, now, tp):
        ...

    @property
    @abc.abstractmethod
    def unacked(self):
        ...

    @abc.abstractmethod
    def on_buffer_full(self, tp):
        ...

    @abc.abstractmethod
    def on_buffer_drop(self, tp):
        ...

class ConductorT(ServiceT, MutableSet[TopicT]):
    app: _AppT

    @abc.abstractmethod
    def __init__(self, app, **kwargs: Any):
        self.on_message: ConsumerCallback

    @abc.abstractmethod
    async def on_client_only_start(self) -> None:
        ...

    @abc.abstractmethod
    def acks_enabled_for(self, topic):
        ...

    @abc.abstractmethod
    async def commit(self, topics: TPorTopicSet) -> bool:
        ...

    @abc.abstractmethod
    async def wait_for_subscriptions(self) -> None:
        ...

    @abc.abstractmethod
    async def maybe_wait_for_subscriptions(self) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        ...

    @property
    @abc.abstractmethod
    def acking_topics(self):
        ...

class TransportT(abc.ABC):
    Consumer: ClassVar[Type[ConsumerT]]
    Producer: ClassVar[Type[ProducerT]]
    TransactionManager: ClassVar[Type[TransactionManagerT]]
    Conductor: ClassVar[Type[ConductorT]]
    Fetcher: ClassVar[Type[ServiceT]]
    app: _AppT
    url: List[URL]
    driver_version: str
    loop: asyncio.AbstractEventLoop

    @abc.abstractmethod
    def __init__(self, url, app, loop=None):
        ...

    @abc.abstractmethod
    def create_consumer(self, callback, **kwargs: Any):
        ...

    @abc.abstractmethod
    def create_producer(self, **kwargs: Any):
        ...

    @abc.abstractmethod
    def create_transaction_manager(self, consumer, producer, **kwargs: Any):
        ...

    @abc.abstractmethod
    def create_conductor(self, **kwargs: Any):
        ...