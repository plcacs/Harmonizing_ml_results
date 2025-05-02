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

    @abc.abstractmethod
    def put(self, fut):
        ...

    @abc.abstractmethod
    async def flush(self):
        ...

    @abc.abstractmethod
    async def flush_atmost(self, n):
        ...

    @abc.abstractmethod
    async def wait_until_ebb(self):
        ...

    @property
    @abc.abstractmethod
    def size(self):
        ...

class ProducerT(ServiceT):

    @abc.abstractmethod
    def __init__(self, transport, loop=None, **kwargs):
        ...

    @abc.abstractmethod
    async def send(self, topic, key, value, partition, timestamp, headers, *, transactional_id=None):
        ...

    @abc.abstractmethod
    def send_soon(self, fut):
        ...

    @abc.abstractmethod
    async def send_and_wait(self, topic, key, value, partition, timestamp, headers, *, transactional_id=None):
        ...

    @abc.abstractmethod
    async def create_topic(self, topic, partitions, replication, *, config=None, timeout=1000.0, retention=None, compacting=None, deleting=None, ensure_created=False):
        ...

    @abc.abstractmethod
    def key_partition(self, topic, key):
        ...

    @abc.abstractmethod
    async def flush(self):
        ...

    @abc.abstractmethod
    async def begin_transaction(self, transactional_id):
        ...

    @abc.abstractmethod
    async def commit_transaction(self, transactional_id):
        ...

    @abc.abstractmethod
    async def abort_transaction(self, transactional_id):
        ...

    @abc.abstractmethod
    async def stop_transaction(self, transactional_id):
        ...

    @abc.abstractmethod
    async def maybe_begin_transaction(self, transactional_id):
        ...

    @abc.abstractmethod
    async def commit_transactions(self, tid_to_offset_map, group_id, start_new_transaction=True):
        ...

    @abc.abstractmethod
    def supports_headers(self):
        ...

class TransactionManagerT(ProducerT):

    @abc.abstractmethod
    def __init__(self, transport, loop=None, *, consumer, producer, **kwargs):
        ...

    @abc.abstractmethod
    async def on_partitions_revoked(self, revoked):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    async def commit(self, offsets, start_new_transaction=True):
        ...

    async def begin_transaction(self, transactional_id):
        raise NotImplementedError()

    async def commit_transaction(self, transactional_id):
        raise NotImplementedError()

    async def abort_transaction(self, transactional_id):
        raise NotImplementedError()

    async def stop_transaction(self, transactional_id):
        raise NotImplementedError()

    async def maybe_begin_transaction(self, transactional_id):
        raise NotImplementedError()

    async def commit_transactions(self, tid_to_offset_map, group_id, start_new_transaction=True):
        raise NotImplementedError()

class SchedulingStrategyT:

    @abc.abstractmethod
    def __init__(self):
        ...

    @abc.abstractmethod
    def iterate(self, records):
        ...

class ConsumerT(ServiceT):

    @abc.abstractmethod
    def __init__(self, transport, callback, on_partitions_revoked, on_partitions_assigned, *, commit_interval=None, loop=None, **kwargs):

    @abc.abstractmethod
    async def create_topic(self, topic, partitions, replication, *, config=None, timeout=1000.0, retention=None, compacting=None, deleting=None, ensure_created=False):
        ...

    @abc.abstractmethod
    async def subscribe(self, topics):
        ...

    @abc.abstractmethod
    @no_type_check
    async def getmany(self, timeout):
        ...

    @abc.abstractmethod
    def track_message(self, message):
        ...

    @abc.abstractmethod
    async def perform_seek(self):
        ...

    @abc.abstractmethod
    def ack(self, message):
        ...

    @abc.abstractmethod
    async def wait_empty(self):
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
    async def position(self, tp):
        ...

    @abc.abstractmethod
    async def seek(self, partition, offset):
        ...

    @abc.abstractmethod
    async def seek_wait(self, partitions):
        ...

    @abc.abstractmethod
    async def commit(self, topics=None, start_new_transaction=True):
        ...

    @abc.abstractmethod
    async def on_task_error(self, exc):
        ...

    @abc.abstractmethod
    async def earliest_offsets(self, *partitions):
        ...

    @abc.abstractmethod
    async def highwaters(self, *partitions):
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

    @abc.abstractmethod
    def __init__(self, app, **kwargs):

    @abc.abstractmethod
    async def on_client_only_start(self):
        ...

    @abc.abstractmethod
    def acks_enabled_for(self, topic):
        ...

    @abc.abstractmethod
    async def commit(self, topics):
        ...

    @abc.abstractmethod
    async def wait_for_subscriptions(self):
        ...

    @abc.abstractmethod
    async def maybe_wait_for_subscriptions(self):
        ...

    @abc.abstractmethod
    async def on_partitions_assigned(self, assigned):
        ...

    @property
    @abc.abstractmethod
    def acking_topics(self):
        ...

class TransportT(abc.ABC):

    @abc.abstractmethod
    def __init__(self, url, app, loop=None):
        ...

    @abc.abstractmethod
    def create_consumer(self, callback, **kwargs):
        ...

    @abc.abstractmethod
    def create_producer(self, **kwargs):
        ...

    @abc.abstractmethod
    def create_transaction_manager(self, consumer, producer, **kwargs):
        ...

    @abc.abstractmethod
    def create_conductor(self, **kwargs):
        ...