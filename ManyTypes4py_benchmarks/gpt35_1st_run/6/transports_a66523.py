from typing import AbstractSet, Any, Awaitable, Callable, Optional, Sequence, Set, Union

class ProducerBufferT(ServiceT):

    @abc.abstractmethod
    def put(self, fut: Any) -> None:
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
    def size(self) -> Any:
        ...

class ProducerT(ServiceT):

    @abc.abstractmethod
    def __init__(self, transport: Any, loop: Optional[Any] = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def send(self, topic: str, key: Any, value: Any, partition: int, timestamp: Any, headers: Any, *, transactional_id: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def send_soon(self, fut: Any) -> None:
        ...

    @abc.abstractmethod
    async def send_and_wait(self, topic: str, key: Any, value: Any, partition: int, timestamp: Any, headers: Any, *, transactional_id: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Any = None, timeout: float = 1000.0, retention: Any = None, compacting: Any = None, deleting: Any = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: Any) -> int:
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
    async def commit_transactions(self, tid_to_offset_map: Any, group_id: Any, start_new_transaction: bool = True) -> None:
        ...

    @abc.abstractmethod
    def supports_headers(self) -> Any:
        ...

class TransactionManagerT(ProducerT):

    @abc.abstractmethod
    def __init__(self, transport: Any, loop: Optional[Any] = None, *, consumer: Any, producer: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked: Set[Any]) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[Any], revoked: Set[Any], newly_assigned: Set[Any]) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, offsets: Any, start_new_transaction: bool = True) -> None:
        ...

class ConsumerT(ServiceT):

    @abc.abstractmethod
    def __init__(self, transport: Any, callback: Callable[[Message], Awaitable], on_partitions_revoked: Callable[[Set[TP]], Awaitable[None]], on_partitions_assigned: Callable[[Set[TP]], Awaitable[None], *, commit_interval: Any = None, loop: Optional[Any] = None, **kwargs: Any) -> None:

    @abc.abstractmethod
    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Any = None, timeout: float = 1000.0, retention: Any = None, compacting: Any = None, deleting: Any = None, ensure_created: bool = False) -> None:
        ...

    @abc.abstractmethod
    async def subscribe(self, topics: Any) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    async def getmany(self, timeout: Any) -> None:
        ...

    @abc.abstractmethod
    def track_message(self, message: Any) -> None:
        ...

    @abc.abstractmethod
    async def perform_seek(self) -> None:
        ...

    @abc.abstractmethod
    def ack(self, message: Any) -> None:
        ...

    @abc.abstractmethod
    async def wait_empty(self) -> None:
        ...

    @abc.abstractmethod
    def assignment(self) -> Any:
        ...

    @abc.abstractmethod
    def highwater(self, tp: Any) -> Any:
        ...

    @abc.abstractmethod
    def stop_flow(self) -> None:
        ...

    @abc.abstractmethod
    def resume_flow(self) -> None:
        ...

    @abc.abstractmethod
    def pause_partitions(self, tps: Any) -> None:
        ...

    @abc.abstractmethod
    def resume_partitions(self, tps: Any) -> None:
        ...

    @abc.abstractmethod
    async def position(self, tp: Any) -> None:
        ...

    @abc.abstractmethod
    async def seek(self, partition: Any, offset: Any) -> None:
        ...

    @abc.abstractmethod
    async def seek_wait(self, partitions: Any) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, topics: Any = None, start_new_transaction: bool = True) -> None:
        ...

    @abc.abstractmethod
    async def on_task_error(self, exc: Any) -> None:
        ...

    @abc.abstractmethod
    async def earliest_offsets(self, *partitions: Any) -> None:
        ...

    @abc.abstractmethod
    async def highwaters(self, *partitions: Any) -> None:
        ...

    @abc.abstractmethod
    def topic_partitions(self, topic: str) -> Any:
        ...

    @abc.abstractmethod
    def key_partition(self, topic: str, key: Any, partition: Optional[int] = None) -> int:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    def verify_recovery_event_path(self, now: Any, tp: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def unacked(self) -> Any:
        ...

    @abc.abstractmethod
    def on_buffer_full(self, tp: Any) -> None:
        ...

    @abc.abstractmethod
    def on_buffer_drop(self, tp: Any) -> None:
        ...

class ConductorT(ServiceT, MutableSet[TopicT]):

    @abc.abstractmethod
    def __init__(self, app: Any, **kwargs: Any) -> None:

    @abc.abstractmethod
    async def on_client_only_start(self) -> None:
        ...

    @abc.abstractmethod
    def acks_enabled_for(self, topic: Any) -> None:
        ...

    @abc.abstractmethod
    async def commit(self, topics: Any) -> None:
        ...

    @abc.abstractmethod
    async def wait_for_subscriptions(self) -> None:
        ...

    @abc.abstractmethod
    async def maybe_wait_for_subscriptions(self) -> None:
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def acking_topics(self) -> Any:
        ...

class TransportT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, url: Any, app: Any, loop: Optional[Any] = None) -> None:
        ...

    @abc.abstractmethod
    def create_consumer(self, callback: Callable[[Message], Awaitable], **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def create_producer(self, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def create_transaction_manager(self, consumer: Any, producer: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def create_conductor(self, **kwargs: Any) -> None:
        ...
