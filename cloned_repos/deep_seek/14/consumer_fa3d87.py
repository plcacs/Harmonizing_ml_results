"""Consumer - fetching messages and managing consumer state."""
import abc
import asyncio
import gc
import typing
from collections import defaultdict
from time import monotonic
from typing import (
    Any, AsyncIterator, Awaitable, ClassVar, Dict, Iterable, Iterator, List,
    Mapping, MutableMapping, MutableSet, NamedTuple, Optional, Set, Tuple,
    Type, Union, cast
)
from weakref import WeakSet
from mode import Service, ServiceT, flight_recorder, get_logger
from mode.threads import MethodQueue, QueueServiceThread
from mode.utils.futures import notify
from mode.utils.locks import Event
from mode.utils.text import pluralize
from mode.utils.times import Seconds
from faust.exceptions import ProducerSendError
from faust.types import AppT, ConsumerMessage, Message, RecordMetadata, TP
from faust.types.core import HeadersArg
from faust.types.transports import (
    ConsumerCallback, ConsumerT, PartitionsAssignedCallback,
    PartitionsRevokedCallback, ProducerT, TPorTopicSet, TransactionManagerT,
    TransportT
)
from faust.types.tuples import FutureMessage
from faust.utils import terminal
from faust.utils.functional import consecutive_numbers
from faust.utils.tracing import traced_from_parent_span

if typing.TYPE_CHECKING:
    from faust.app import App as _App
else:
    class _App:
        ...

__all__ = ['Consumer', 'Fetcher']

CONSUMER_FETCHING = 'FETCHING'
CONSUMER_PARTITIONS_REVOKED = 'PARTITIONS_REVOKED'
CONSUMER_PARTITIONS_ASSIGNED = 'PARTITIONS_ASSIGNED'
CONSUMER_COMMITTING = 'COMMITTING'
CONSUMER_SEEKING = 'SEEKING'
CONSUMER_WAIT_EMPTY = 'WAIT_EMPTY'

logger = get_logger(__name__)
RecordMap = Mapping[TP, List[Any]]

class TopicPartitionGroup(NamedTuple):
    """Tuple of ``(topic, partition, group)``."""
    topic: str
    partition: int
    group: str

def ensure_TP(tp: Any) -> TP:
    """Convert aiokafka ``TopicPartition`` to Faust ``TP``."""
    return tp if isinstance(tp, TP) else TP(tp.topic, tp.partition)

def ensure_TPset(tps: Iterable[Any]) -> Set[TP]:
    """Convert set of aiokafka ``TopicPartition`` to Faust ``TP``."""
    return {ensure_TP(tp) for tp in tps}

class Fetcher(Service):
    """Service fetching messages from Kafka."""
    logger: ClassVar[Any] = logger
    _drainer: Optional[asyncio.Future] = None

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        self.app: AppT = app
        super().__init__(**kwargs)

    async def on_stop(self) -> None:
        """Call when the fetcher is stopping."""
        if self._drainer is not None and (not self._drainer.done()):
            self._drainer.cancel()
            while True:
                try:
                    await asyncio.wait_for(self._drainer, timeout=1.0)
                except StopIteration:
                    break
                except asyncio.CancelledError:
                    break
                except asyncio.TimeoutError:
                    self.log.warning('Fetcher is ignoring cancel or slow :(')
                else:
                    break

    @Service.task
    async def _fetcher(self) -> None:
        try:
            consumer = cast(Consumer, self.app.consumer)
            self._drainer = asyncio.ensure_future(
                consumer._drain_messages(self), loop=self.loop)
            await self._drainer
        except asyncio.CancelledError:
            pass
        finally:
            self.set_shutdown()

class TransactionManager(Service, TransactionManagerT):
    """Manage producer transactions."""
    transactional_id_format: str = '{group_id}-{tpg.group}-{tpg.partition}'

    def __init__(
        self,
        transport: TransportT,
        *,
        consumer: ConsumerT,
        producer: ProducerT,
        **kwargs: Any
    ) -> None:
        self.transport: TransportT = transport
        self.app: AppT = self.transport.app
        self.consumer: ConsumerT = consumer
        self.producer: ProducerT = producer
        super().__init__(**kwargs)

    async def flush(self) -> None:
        """Wait for producer to transmit all pending messages."""
        await self.producer.flush()

    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call when the cluster is rebalancing and partitions are revoked."""
        await traced_from_parent_span()(self.flush)()

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP]
    ) -> None:
        """Call when the cluster is rebalancing."""
        T = traced_from_parent_span()
        revoked_tids = sorted(self._tps_to_transactional_ids(revoked))
        if revoked_tids:
            self.log.info(
                'Stopping %r transactional %s for %r revoked %s...',
                len(revoked_tids),
                pluralize(len(revoked_tids), 'producer'),
                len(revoked),
                pluralize(len(revoked), 'partition'))
            await T(self._stop_transactions, tids=revoked_tids)(revoked_tids)
        assigned_tids = sorted(self._tps_to_transactional_ids(assigned))
        if assigned_tids:
            self.log.info(
                'Starting %r transactional %s for %r assigned %s...',
                len(assigned_tids),
                pluralize(len(assigned_tids), 'producer'),
                len(assigned),
                pluralize(len(assigned), 'partition'))
            await T(self._start_transactions, tids=assigned_tids)(assigned_tids)

    async def _stop_transactions(self, tids: List[str]) -> None:
        T = traced_from_parent_span()
        producer = self.producer
        for transactional_id in tids:
            await T(producer.stop_transaction)(transactional_id)

    async def _start_transactions(self, tids: List[str]) -> None:
        T = traced_from_parent_span()
        producer = self.producer
        for transactional_id in tids:
            await T(producer.maybe_begin_transaction)(transactional_id)

    def _tps_to_transactional_ids(self, tps: Set[TP]) -> Set[str]:
        return {
            self.transactional_id_format.format(
                tpg=tpg,
                group_id=self.app.conf.id
            ) for tpg in self._tps_to_active_tpgs(tps)
        }

    def _tps_to_active_tpgs(self, tps: Set[TP]) -> Set[TopicPartitionGroup]:
        assignor = self.app.assignor
        return {
            TopicPartitionGroup(tp.topic, tp.partition, assignor.group_for_topic(tp.topic))
            for tp in tps if not assignor.is_standby(tp)
        }

    async def send(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = None
    ) -> Awaitable[RecordMetadata]:
        """Schedule message to be sent by producer."""
        group = transactional_id = None
        p = self.consumer.key_partition(topic, key, partition)
        if p is not None:
            group = self.app.assignor.group_for_topic(topic)
            transactional_id = f'{self.app.conf.id}-{group}-{p}'
        return await self.producer.send(
            topic, key, value, p, timestamp, headers,
            transactional_id=transactional_id)

    def send_soon(self, fut: FutureMessage) -> None:
        raise NotImplementedError()

    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: Optional[bytes],
        partition: Optional[int],
        timestamp: Optional[float],
        headers: Optional[HeadersArg],
        *,
        transactional_id: Optional[str] = None
    ) -> RecordMetadata:
        """Send message and wait for it to be transmitted."""
        fut = await self.send(
            topic, key, value, partition, timestamp, headers,
            transactional_id=transactional_id)
        return await fut

    async def commit(
        self,
        offsets: Mapping[TP, int],
        start_new_transaction: bool = True
    ) -> bool:
        """Commit offsets for partitions."""
        producer = self.producer
        group_id = self.app.conf.id
        by_transactional_id: Dict[str, Dict[TP, int]] = defaultdict(dict)
        for tp, offset in offsets.items():
            group = self.app.assignor.group_for_topic(tp.topic)
            transactional_id = f'{group_id}-{group}-{tp.partition}'
            by_transactional_id[transactional_id][tp] = offset
        if by_transactional_id:
            await producer.commit_transactions(
                by_transactional_id, group_id,
                start_new_transaction=start_new_transaction)
        return True

    def key_partition(self, topic: str, key: Optional[bytes]) -> Optional[int]:
        raise NotImplementedError()

    async def create_topic(
        self,
        topic: str,
        partitions: int,
        replication: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        timeout: float = 30.0,
        retention: Optional[float] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        ensure_created: bool = False
    ) -> None:
        """Create/declare topic on server."""
        return await self.producer.create_topic(
            topic, partitions, replication, config=config, timeout=timeout,
            retention=retention, compacting=compacting, deleting=deleting,
            ensure_created=ensure_created)

    def supports_headers(self) -> bool:
        """Return :const:`True` if the Kafka server supports headers."""
        return self.producer.supports_headers()

class Consumer(Service, ConsumerT):
    """Base Consumer."""
    logger: ClassVar[Any] = logger
    consumer_stopped_errors: ClassVar[Tuple[Type[Exception], ...]] = ()
    _waiting_for_ack: Optional[asyncio.Future] = None
    _commit_fut: Optional[asyncio.Future] = None
    _n_acked: int = 0
    flow_active: bool = True

    def __init__(
        self,
        transport: TransportT,
        callback: ConsumerCallback,
        on_partitions_revoked: PartitionsRevokedCallback,
        on_partitions_assigned: PartitionsAssignedCallback,
        *,
        commit_interval: Optional[float] = None,
        commit_livelock_soft_timeout: Optional[float] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs: Any
    ) -> None:
        assert callback is not None
        self.transport: TransportT = transport
        self.app: AppT = self.transport.app
        self.in_transaction: bool = self.app.in_transaction
        self.callback: ConsumerCallback = callback
        self._on_message_in = self.app.sensors.on_message_in
        self._on_partitions_revoked: PartitionsRevokedCallback = on_partitions_revoked
        self._on_partitions_assigned: PartitionsAssignedCallback = on_partitions_assigned
        self._commit_every: Optional[int] = self.app.conf.broker_commit_every
        self.scheduler = self.app.conf.ConsumerScheduler()
        self.commit_interval: float = (
            commit_interval or self.app.conf.broker_commit_interval)
        self.commit_livelock_soft_timeout: float = (
            commit_livelock_soft_timeout or
            self.app.conf.broker_commit_livelock_soft_timeout)
        self._gap: Dict[TP, List[int]] = defaultdict(list)
        self._acked: Dict[TP, List[int]] = defaultdict(list)
        self._acked_index: Dict[TP, Set[int]]] = defaultdict(set)
        self._read_offset: Dict[TP, Optional[int]] = defaultdict(lambda: None)
        self._committed_offset: Dict[TP, Optional[int]] = defaultdict(lambda: None)
        self._unacked_messages: MutableSet[Message] = WeakSet()
        self._buffered_partitions: Set[TP] = set()
        self._waiting_for_ack: Optional[asyncio.Future] = None
        self._time_start: float = monotonic()
        self._end_offset_monitor_interval: float = self.commit_interval * 2
        self.randomly_assigned_topics: Set[str] = set()
        self.can_resume_flow: Event = Event()
        self._reset_state()
        super().__init__(
            loop=loop or self.transport.loop, **kwargs)
        self.transactions: TransactionManagerT = (
            self.transport.create_transaction_manager(
                consumer=self,
                producer=self.app.producer,
                beacon=self.beacon,
                loop=self.loop))

    def on_init_dependencies(self) -> List[ServiceT]:
        """Return list of services this consumer depends on."""
        if self.in_transaction:
            return [self.transactions]
        return []

    def _reset_state(self) -> None:
        self._active_partitions: Optional[Set[TP]] = None
        self._paused_partitions: Set[TP] = set()
        self._buffered_partitions: Set[TP] = set()
        self.can_resume_flow.clear()
        self.flow_active = True
        self._time_start = monotonic()

    async def on_restart(self) -> None:
        """Call when the consumer is restarted."""
        self._reset_state()
        self.on_init()

    def _get_active_partitions(self) -> Set[TP]:
        tps = self._active_partitions
        if tps is None:
            return self._set_active_tps(self.assignment())
        assert all((isinstance(x, TP) for x in tps))
        return tps

    def _set_active_tps(self, tps: Iterable[TP]) -> Set[TP]:
        xtps = self._active_partitions = ensure_TPset(tps)
        xtps.difference_update(self._paused_partitions)
        return xtps

    def on_buffer_full(self, tp: TP) -> None:
        active_partitions = self._get_active_partitions()
        active_partitions.discard(tp)
        self._buffered_partitions.add(tp)

    def on_buffer_drop(self, tp: TP) -> None:
        buffered_partitions = self._buffered_partitions
        if tp in buffered_partitions:
            active_partitions = self._get_active_partitions()
            active_partitions.add(tp)
            buffered_partitions.discard(tp)

    @abc.abstractmethod
    async def _commit(self, offsets: Mapping[TP, int]) -> bool:
        ...

    async def perform_seek(self) -> None:
        """Seek all partitions to their current committed position."""
        read_offset = self._read_offset
        _committed_offsets = await self.seek_to_committed()
        read_offset.update({
            tp: offset if offset is not None and offset >= 0 else None
            for tp, offset in _committed_offsets.items()
        })
        committed_offsets = {
            ensure_TP(tp): offset if offset else None
            for tp, offset in _committed_offsets.items()
            if offset is not None
        }
        self._committed_offset.update(committed_offsets)

    @abc.abstractmethod
    async def seek_to_committed(self) -> Dict[TP, Optional[int]]:
        """Seek all partitions to their committed offsets."""
        ...

    async def seek(self, partition: TP, offset: Optional[int]) -> None:
        """Seek partition to specific offset."""
        self.log.dev('SEEK %r -> %r', partition, offset)
        await self._seek(partition, offset)
        self._read_offset[ensure_TP(partition)] = offset if offset else None

    @abc.abstractmethod
    async def _seek(self, partition: TP, offset: Optional[int]) -> None:
        ...

    def stop_flow(self) -> None:
        """Block consumer from processing any more messages."""
        self.flow_active = False
        self.can_resume_flow.clear()

    def resume_flow(self) -> None:
        """Allow consumer to process messages."""
        self.flow_active = True
        self.can_resume_flow.set()

    def pause_partitions(self, tps: Iterable[TP]) -> None:
        """Pause fetching from partitions."""
        tpset = ensure_TPset(tps)
        self._get_active_partitions().difference_update(tpset)
        self._paused_partitions.update(tpset)

    def resume_partitions(self, tps: Iterable[TP]) -> None:
        """Resume fetching from partitions."""
        tpset = ensure_TPset(tps)
        self._get_active_partitions().update(tps)
        self._paused_partitions.difference_update(tpset)

    @abc.abstractmethod
    def _new_topicpartition(self, topic: str, partition: int) -> Any:
        ...

    def _is_changelog_tp(self, tp: TP) -> bool:
        return tp.topic in self.app.tables.changelog_topics

    @Service.transitions_to(CONSUMER_PARTITIONS_REVOKED)
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call during rebalancing when partitions are being revoked."""
        span = self.app._start_span_from_rebalancing('on_partitions_revoked')
        T = traced_from_parent_span(span)
        with span:
            if self