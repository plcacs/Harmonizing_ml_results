"""Consumer - fetching messages and managing consumer state.

The Consumer is responsible for:

   - Holds reference to the transport that created it

   - ... and the app via ``self.transport.app``.

   - Has a callback that usually points back to ``Conductor.on_message``.

   - Receives messages and calls the callback for every message received.

   - Keeps track of the message and its acked/unacked status.

   - The Conductor forwards the message to all Streams that subscribes
     to the topic the message was sent to.

       + Messages are reference counted, and the Conductor increases
         the reference count to the number of subscribed streams.

       + ``Stream.__aiter__`` is set up in a way such that when what is
         iterating over the stream is finished with the message, a
         finally: block will decrease the reference count by one.

       + When the reference count for a message hits zero, the stream will
         call ``Consumer.ack(message)``, which will mark that topic +
         partition + offset combination as "committable"

       + If all the streams share the same key_type/value_type,
         the conductor will only deserialize the payload once.

   - Commits the offset at an interval

      + The Consumer has a background thread that periodically commits the
        offset.

      - If the consumer marked an offset as committable this thread
        will advance the committed offset.

      + To find the offset that it can safely advance to the commit thread
        will traverse the _acked mapping of TP to list of acked offsets, by
        finding a range of consecutive acked offsets (see note in
        _new_offset).

"""
import abc
import asyncio
import gc
import typing

from collections import defaultdict
from time import monotonic
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
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
    ConsumerCallback,
    ConsumerT,
    PartitionsAssignedCallback,
    PartitionsRevokedCallback,
    ProducerT,
    TPorTopicSet,
    TransactionManagerT,
    TransportT,
)
from faust.types.tuples import FutureMessage
from faust.utils import terminal
from faust.utils.functional import consecutive_numbers
from faust.utils.tracing import traced_from_parent_span

if typing.TYPE_CHECKING:  # pragma: no cover
    from faust.app import App as _App
else:
    class _App: ...  # noqa: E701

__all__ = ['Consumer', 'Fetcher']

# These flags are used for Service.diag, tracking what the consumer
# service is currently doing.
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
    group: int


def ensure_TP(tp: Any) -> TP:
    """Convert aiokafka ``TopicPartition`` to Faust ``TP``."""
    return tp if isinstance(tp, TP) else TP(tp.topic, tp.partition)


def ensure_TPset(tps: Iterable[Any]) -> Set[TP]:
    """Convert set of aiokafka ``TopicPartition`` to Faust ``TP``."""
    return {ensure_TP(tp) for tp in tps}


class Fetcher(Service):
    """Service fetching messages from Kafka."""

    app: AppT

    logger = logger
    _drainer: Optional[asyncio.Future] = None

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        self.app = app
        super().__init__(**kwargs)

    async def on_stop(self) -> None:
        """Call when the fetcher is stopping."""
        if self._drainer is not None and not self._drainer.done():
            self._drainer.cancel()
            while True:
                try:
                    await asyncio.wait_for(self._drainer, timeout=1.0)
                except StopIteration:
                    # Task is cancelled right before coro stops.
                    break
                except asyncio.CancelledError:
                    break
                except asyncio.TimeoutError:
                    self.log.warning('Fetcher is ignoring cancel or slow :(')
                else:  # pragma: no cover
                    # coverage does not record this line as being executed
                    # but I've verified that it is [ask]
                    break

    @Service.task
    async def _fetcher(self) -> None:
        try:
            consumer = cast(Consumer, self.app.consumer)
            self._drainer = asyncio.ensure_future(
                consumer._drain_messages(self),
                loop=self.loop,
            )
            await self._drainer
        except asyncio.CancelledError:
            pass
        finally:
            self.set_shutdown()


class TransactionManager(Service, TransactionManagerT):
    """Manage producer transactions."""

    app: AppT

    transactional_id_format = '{group_id}-{tpg.group}-{tpg.partition}'

    def __init__(self, transport: TransportT,
                 *,
                 consumer: ConsumerT,
                 producer: ProducerT,
                 **kwargs: Any) -> None:
        self.transport = transport
        self.app = self.transport.app
        self.consumer = consumer
        self.producer = producer
        super().__init__(**kwargs)

    async def flush(self) -> None:
        """Wait for producer to transmit all pending messages."""
        await self.producer.flush()

    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        """Call when the cluster is rebalancing and partitions are revoked."""
        await traced_from_parent_span()(self.flush)()

    async def on_rebalance(self,
                           assigned: Set[TP],
                           revoked: Set[TP],
                           newly_assigned: Set[TP]) -> None:
        """Call when the cluster is rebalancing."""
        T = traced_from_parent_span()
        # Stop producers for revoked partitions.
        revoked_tids = sorted(self._tps_to_transactional_ids(revoked))
        if revoked_tids:
            self.log.info(
                'Stopping %r transactional %s for %r revoked %s...',
                len(revoked_tids),
                pluralize(len(revoked_tids), 'producer'),
                len(revoked),
                pluralize(len(revoked), 'partition'))
            await T(self._stop_transactions, tids=revoked_tids)(revoked_tids)

        # Start produers for assigned partitions
        assigned_tids = sorted(self._tps_to_transactional_ids(assigned))
        if assigned_tids:
            self.log.info(
                'Starting %r transactional %s for %r assigned %s...',
                len(assigned_tids),
                pluralize(len(assigned_tids), 'producer'),
                len(assigned),
                pluralize(len(assigned), 'partition'))
            await T(self._start_transactions,
                    tids=assigned_tids)(assigned_tids)

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
                group_id=self.app.conf.id,
            )
            for tpg in self._tps_to_active_tpgs(tps)
        }

    def _tps_to_active_tpgs(self, tps: Set[TP]) -> Set[TopicPartitionGroup]:
        assignor = self.app.assignor
        return {
            TopicPartitionGroup(
                tp.topic,
                tp.partition,
                assignor.group_for_topic(tp.topic),
            )
            for tp in tps
            if not assignor.is_standby(tp)
        }

    async def send(self, topic: str, key: Optional[bytes],
                   value: Optional[bytes],
                   partition: Optional[int],
                   timestamp: Optional[float],
                   headers: Optional[HeadersArg],
                   *,
                   transactional_id: Optional[str] = None) -> Awaitable[RecordMetadata]:
        """Schedule message to be sent by producer."""
        group = transactional_id = None
        p = self.consumer.key_partition(topic, key, partition)
        if p is not None:
            group = self.app.assignor.group_for_topic(topic)
            transactional_id = f'{self.app.conf.id}-{group}-{p}'
        return await self.producer.send(
            topic, key, value, p, timestamp, headers,
            transactional_id=transactional_id,
        )

    def send_soon(self, fut: FutureMessage) -> None:
        raise NotImplementedError()

    async def send_and_wait(self, topic: str, key: Optional[bytes],
                            value: Optional[bytes],
                            partition: Optional[int],
                            timestamp: Optional[float],
                            headers: Optional[HeadersArg],
                            *,
                            transactional_id: Optional[str] = None) -> RecordMetadata:
        """Send message and wait for it to be transmitted."""
        fut = await self.send(topic, key, value, partition, timestamp, headers)
        return await fut

    async def commit(self, offsets: Mapping[TP, int],
                     start_new_transaction: bool = True) -> bool:
        """Commit offsets for partitions."""
        producer = self.producer
        group_id = self.app.conf.id
        by_transactional_id: MutableMapping[str, MutableMapping[TP, int]]
        by_transactional_id = defaultdict(dict)

        for tp, offset in offsets.items():
            group = self.app.assignor.group_for_topic(tp.topic)
            transactional_id = f'{group_id}-{group}-{tp.partition}'
            by_transactional_id[transactional_id][tp] = offset

        if by_transactional_id:
            await producer.commit_transactions(
                by_transactional_id, group_id,
                start_new_transaction=start_new_transaction,
            )
        return True

    def key_partition(self, topic: str, key: bytes) -> TP:
        raise NotImplementedError()

    async def create_topic(self,
                           topic: str,
                           partitions: int,
                           replication: int,
                           *,
                           config: Optional[Mapping[str, Any]] = None,
                           timeout: Seconds = 30.0,
                           retention: Optional[Seconds] = None,
                           compacting: Optional[bool] = None,
                           deleting: Optional[bool] = None,
                           ensure_created: bool = False) -> None:
        """Create/declare topic on server."""
        return await self.producer.create_topic(
            topic, partitions, replication,
            config=config,
            timeout=timeout,
            retention=retention,
            compacting=compacting,
            deleting=deleting,
            ensure_created=ensure_created,
        )

    def supports_headers(self) -> bool:
        """Return :const:`True` if the Kafka server supports headers."""
        return self.producer.supports_headers()


class Consumer(Service, ConsumerT):
    """Base Consumer."""

    app: AppT

    logger = logger

    #: Tuple of exception types that may be raised when the
    #: underlying consumer driver is stopped.
    consumer_stopped_errors: ClassVar[Tuple[Type[BaseException], ...]] = ()

    # Mapping of TP to list of gap in offsets.
    _gap: MutableMapping[TP, List[int]]

    # Mapping of TP to list of acked offsets.
    _acked: MutableMapping[TP, List[int]]

    #: Fast lookup to see if tp+offset was acked.
    _acked_index: MutableMapping[TP, Set[int]]

    #: Keeps track of the currently read offset in each TP
    _read_offset: MutableMapping[TP, Optional[int]]

    #: Keeps track of the currently committed offset in each TP.
    _committed_offset: MutableMapping[TP, Optional[int]]

    #: The consumer.wait_empty() method will set this to be notified
    #: when something acks a message.
    _waiting_for_ack: Optional[asyncio.Future] = None

    #: Used by .commit to ensure only one thread is comitting at a time.
    #: Other thread starting to commit while a commit is already active,
    #: will wait for the original request to finish, and do nothing.
    _commit_fut: Optional[asyncio.Future] = None

    #: Set of unacked messages: that is messages that we started processing
    #: and that we MUST attempt to complete processing of, before
    #: shutting down or resuming a rebalance.
    _unacked_messages: MutableSet[Message]

    #: Time of when the consumer was started.
    _time_start: float

    # How often to poll and track log end offsets.
    _end_offset_monitor_interval: float

    _commit_every: Optional[int]
    _n_acked: int = 0

    _active_partitions: Optional[Set[TP]]
    _paused_partitions: Set[TP]
    _buffered_partitions: Set[TP]

    flow_active: bool = True
    can_resume_flow: Event

    def __init__(self,
                 transport: TransportT,
                 callback: ConsumerCallback,
                 on_partitions_revoked: PartitionsRevokedCallback,
                 on_partitions_assigned: PartitionsAssignedCallback,
                 *,
                 commit_interval: Optional[float] = None,
                 commit_livelock_soft_timeout: Optional[float] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 **kwargs: Any) -> None:
        assert callback is not None
        self.transport = transport
        self.app = self.transport.app
        self.in_transaction = self.app.in_transaction
        self.callback = callback
        self._on_message_in = self.app.sensors.on_message_in
        self._on_partitions_revoked = on_partitions_revoked
        self._on_partitions_assigned = on_partitions_assigned
        self._commit_every = self.app.conf.broker_commit_every
        self.scheduler = self.app.conf.ConsumerScheduler()  # type: ignore
        self.commit_interval = (
            commit_interval or self.app.conf.broker_commit_interval)
        self.commit_livelock_soft_timeout = (
            commit_livelock_soft_timeout or
            self.app.conf.broker_commit_livelock_soft_timeout)
        self._gap = defaultdict(list)
        self._acked = defaultdict(list)
        self._acked_index = defaultdict(set)
        self._read_offset = defaultdict(lambda: None)
        self._committed_offset = defaultdict(lambda: None)
        self._unacked_messages = WeakSet()
        self._buffered_partitions = set()
        self._waiting_for_ack = None
        self._time_start = monotonic()
        self._end_offset_monitor_interval = self.commit_interval * 2
        self.randomly_assigned_topics = set()
        self.can_resume_flow = Event()
        self._reset_state()
        super().__init__(loop=loop or self.transport.loop, **kwargs)
        self.transactions = self.transport.create_transaction_manager(
            consumer=self,
            producer=self.app.producer,
            beacon=self.beacon,
            loop=self.loop,
        )

    def on_init_dependencies(self) -> Iterable[ServiceT]:
        """Return list of services this consumer depends on."""
        # We start the TransactionManager only if
        # processing_guarantee='exactly_once'
        if self.in_transaction:
            return [self.transactions]
        return []

    def _reset_state(self) -> None:
        self._active_partitions = None
        self._paused_partitions = set()
        self._buffered_partitions = set()
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
        assert all(isinstance(x, TP) for x in tps)
        return tps

    def _set_active_tps(self, tps: Iterable[TP]) -> Set[TP]:
        xtps = self._active_partitions = ensure_TPset(tps)  # copy
        xtps.difference_update(self._paused_partitions)
        return xtps

    def on_buffer_full(self, tp: TP) -> None:
        active_partitions = self._get_active_partitions()
        active_partitions.discard(tp)
        self._buffered_partitions.add(tp)

    def on_buffer_drop(self, tp: TP)