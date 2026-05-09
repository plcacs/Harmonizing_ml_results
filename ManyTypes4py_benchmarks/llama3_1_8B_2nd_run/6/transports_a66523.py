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
ConsumerCallback = Callable[[Message], Awaitable]
TPorTopic = Union[str, TP]
TPorTopicSet = AbstractSet[TPorTopic]
PartitionsRevokedCallback = Callable[[Set[TP]], Awaitable[None]]
PartitionsAssignedCallback = Callable[[Set[TP]], Awaitable[None]]
PartitionerT = Callable[[Optional[bytes], Sequence[int], Sequence[int]], int]

class ProducerBufferT(ServiceT):
    """Producer Buffer Service."""
    @abc.abstractmethod
    def put(self, fut: FutureMessage) -> None:
        ...

    @abc.abstractmethod
    async def flush(self) -> None:
        ...

    @abc.abstractmethod
    async def flush_atmost(self, n: int) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_ebb(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

class ProducerT(ServiceT):
    """Producer Service."""
    @abc.abstractmethod
    def __init__(self, transport: TransportT, loop: Optional[asyncio.BaseEventLoop] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def send(self, topic: str, key: str, value: Any, partition: int, timestamp: Optional[int] = None, headers: HeadersArg = None, *, transactional_id: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def send_soon(self, fut: FutureMessage) -> None:
        ...

    @abc.abstractmethod
    async def send_and_wait(self, topic: str, key: str, value: Any, partition: int, timestamp: Optional[int] = None, headers: HeadersArg = None, *, transactional_id: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Mapping[str, Any] = None, timeout: float = 1000.0, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: str, partition: Optional[int] = None) -> int:
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
    async def commit_transactions(self, tid_to_offset_map: Mapping[str, int], group_id: str, start_new_transaction: bool = True) -> None:
        ...

    @abc.abstractmethod
    def supports_headers(self) -> bool:
        ...

class TransactionManagerT(ProducerT):
    """Transaction Manager Service."""
    @abc.abstractmethod
    def __init__(self, transport: TransportT, loop: Optional[asyncio.BaseEventLoop] = None, *, consumer: ConsumerT, producer: ProducerT, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[TP]) -> Awaitable[None]:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> Awaitable[None]:
        ...

    @abc.abstractmethod
    async def commit(self, offsets: Mapping[TP, int], start_new_transaction: bool = True) -> None:
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

    async def commit_transactions(self, tid_to_offset_map: Mapping[str, int], group_id: str, start_new_transaction: bool = True) -> None:
        raise NotImplementedError()

class SchedulingStrategyT:
    """Scheduling Strategy."""
    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @abc.abstractmethod
    def iterate(self, records: Sequence[RecordMetadata]) -> Iterator[RecordMetadata]:
        ...

class ConsumerT(ServiceT):
    """Consumer Service."""
    @abc.abstractmethod
    def __init__(self, transport: TransportT, callback: ConsumerCallback, on_partitions_revoked: PartitionsRevokedCallback, on_partitions_assigned: PartitionsAssignedCallback, *, commit_interval: Optional[Seconds] = None, loop: Optional[asyncio.BaseEventLoop] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Mapping[str, Any] = None, timeout: float = 1000.0, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    async def subscribe(self, topics: Iterable[str]) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    async def getmany(self, timeout: float) -> Awaitable[Sequence[Message]]:
        ...

    @abc.abstractmethod
    def track_message(self, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def perform_seek(self) -> None:
        ...

    @abc.abstractmethod
    def ack(self, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def wait_empty(self) -> None:
        ...

    @abc.abstractmethod
    def assignment(self) -> Mapping[str, int]:
        ...

    @abc.abstractmethod
    def highwater(self, tp: str) -> int:
        ...

    @abc.abstractmethod
    def stop_flow(self) -> None:
        ...

    @abc.abstractmethod
    def resume_flow(self) -> None:
        ...

    @abc.abstractmethod
    def pause_partitions(self, tps: Iterable[str]) -> None:
        ...

    @abc.abstractmethod
    def resume_partitions(self, tps: Iterable[str]) -> None:
        ...

    @abc.abstractmethod
    async def position(self, tp: str) -> int:
        ...

    @abc.abstractmethod
    async def seek(self, partition: int, offset: int) -> None:
        ...

    @abc.abstractmethod
    async def seek_wait(self, partitions: Iterable[int]) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, topics: Optional[Iterable[str]] = None, start_new_transaction: bool = True) -> None:
        ...

    @abc.abstractmethod
    async def on_task_error(self, exc: Exception) -> None:
        ...

    @abc.abstractmethod
    async def earliest_offsets(self, *partitions: int) -> Mapping[int, int]:
        ...

    @abc.abstractmethod
    async def highwaters(self, *partitions: int) -> Mapping[int, int]:
        ...

    @abc.abstractmethod
    def topic_partitions(self, topic: str) -> Iterable[int]:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: str, partition: Optional[int] = None) -> int:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    def verify_recovery_event_path(self, now: float, tp: str) -> None:
        ...

    @property
    @abc.abstractmethod
    def unacked(self) -> Mapping[str, int]:
        ...

    @abc.abstractmethod
    def on_buffer_full(self, tp: str) -> None:
        ...

    @abc.abstractmethod
    def on_buffer_drop(self, tp: str) -> None:
        ...

class ConductorT(ServiceT, MutableSet[TopicT]):
    """Conductor Service."""
    @abc.abstractmethod
    def __init__(self, app: _AppT, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def on_client_only_start(self) -> None:
        ...

    @abc.abstractmethod
    def acks_enabled_for(self, topic: str) -> bool:
        ...

    @abc.abstractmethod
    async def commit(self, topics: Iterable[str]) -> None:
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
    def acking_topics(self) -> Mapping[str, bool]:
        ...

class TransportT(abc.ABC):
    """Transport."""
    @abc.abstractmethod
    def __init__(self, url: URL, app: _AppT, loop: Optional[asyncio.BaseEventLoop] = None) -> None:
        ...

    @abc.abstractmethod
    def create_consumer(self, callback: ConsumerCallback, **kwargs: Any) -> ConsumerT:
        ...

    @abc.abstractmethod
    def create_producer(self, **kwargs: Any) -> ProducerT:
        ...

    @abc.abstractmethod
    def create_transaction_manager(self, consumer: ConsumerT, producer: ProducerT, **kwargs: Any) -> TransactionManagerT:
        ...

    @abc.abstractmethod
    def create_conductor(self, **kwargs: Any) -> ConductorT:
        ...
