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

    @abc.abstractmethod
    def __init__(self, transport: 'TransportT', loop: Optional[asyncio.AbstractEventLoop] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def send(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str] = None) -> FutureMessage:
        ...

    @abc.abstractmethod
    def send_soon(self, fut: FutureMessage) -> None:
        ...

    @abc.abstractmethod
    async def send_and_wait(self, topic: str, key: Optional[bytes], value: Optional[bytes], partition: Optional[int], timestamp: Optional[float], headers: Optional[HeadersArg], *, transactional_id: Optional[str] = None) -> RecordMetadata:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]] = None, timeout: float = 1000.0, retention: Optional[float] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: Optional[bytes]) -> int:
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
    async def commit_transactions(self, tid_to_offset_map: Mapping[str, Mapping[TP, int]], group_id: str, start_new_transaction: bool = True) -> None:
        ...

    @abc.abstractmethod
    def supports_headers(self) -> bool:
        ...

class TransactionManagerT(ProducerT):

    @abc.abstractmethod
    def __init__(self, transport: 'TransportT', loop: Optional[asyncio.AbstractEventLoop] = None, *, consumer: 'ConsumerT', producer: ProducerT, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
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

    async def commit_transactions(self, tid_to_offset_map: Mapping[str, Mapping[TP, int]], group_id: str, start_new_transaction: bool = True) -> None:
        raise NotImplementedError()

class SchedulingStrategyT:

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @abc.abstractmethod
    def iterate(self, records: Iterable) -> Iterator:
        ...

class ConsumerT(ServiceT):

    @abc.abstractmethod
    def __init__(self, transport: 'TransportT', callback: ConsumerCallback, on_partitions_revoked: PartitionsRevokedCallback, on_partitions_assigned: PartitionsAssignedCallback, *, commit_interval: Optional[float] = None, loop: Optional[asyncio.AbstractEventLoop] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Optional[Mapping[str, Any]] = None, timeout: float = 1000.0, retention: Optional[float] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    async def subscribe(self, topics: Iterable[str]) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    async def getmany(self, timeout: float) -> Mapping[TP, List[Message]]:
        ...

    @abc.abstractmethod
    def track_message(self, message: Message) -> None:
        ...

    @abc.abstractmethod
    async def perform_seek(self) -> None:
        ...

    @abc.abstractmethod
    def ack(self, message: Message) -> bool:
        ...

    @abc.abstractmethod
    async def wait_empty(self) -> None:
        ...

    @abc.abstractmethod
    def assignment(self) -> Set[TP]:
        ...

    @abc.abstractmethod
    def highwater(self, tp: TP) -> int:
        ...

    @abc.abstractmethod
    def stop_flow(self) -> None:
        ...

    @abc.abstractmethod
    def resume_flow(self) -> None:
        ...

    @abc.abstractmethod
    def pause_partitions(self, tps: Iterable[TP]) -> None:
        ...

    @abc.abstractmethod
    def resume_partitions(self, tps: Iterable[TP]) -> None:
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
    async def commit(self, topics: Optional[TPorTopicSet] = None, start_new_transaction: bool = True) -> bool:
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
    def topic_partitions(self, topic: str) -> Optional[int]:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = None) -> int:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    def verify_recovery_event_path(self, now: float, tp: TP) -> bool:
        ...

    @property
    @abc.abstractmethod
    def unacked(self) -> Set[Message]:
        ...

    @abc.abstractmethod
    def on_buffer_full(self, tp: TP) -> None:
        ...

    @abc.abstractmethod
    def on_buffer_drop(self, tp: TP) -> None:
        ...

class ConductorT(ServiceT, MutableSet[TopicT]):

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
    def acking_topics(self) -> Set[str]:
        ...

class TransportT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, url: Union[URL, str], app: _AppT, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
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
