#!/usr/bin/env python3
"""
Consumer - fetching messages and managing consumer state.

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
            await T(self._start_transactions,
                    tids=assigned_tids)(assigned_tids)

    async def _stop_transactions(self, tids: Iterable[str]) -> None:
        T = traced_from_parent_span()
        producer = self.producer
        for transactional_id in tids:
            await T(producer.stop_transaction)(transactional_id)

    async def _start_transactions(self, tids: Iterable[str]) -> None:
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
                   timestamp: float,
                   headers: HeadersArg,
                   *,
                   transactional_id: Optional[str] = None) -> Awaitable[RecordMetadata]:
        """Schedule message to be sent by producer."""
        group: Optional[int] = None
        p: Optional[int] = self.consumer.key_partition(topic, key, partition)
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
                            timestamp: float,
                            headers: HeadersArg,
                            *,
                            transactional_id: Optional[str] = None) -> RecordMetadata:
        """Send message and wait for it to be transmitted."""
        fut = await self.send(topic, key, value, partition, timestamp, headers,
                              transactional_id=transactional_id)
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

    def key_partition(self, topic: str, key: Optional[bytes], partition: Optional[int] = None) -> Optional[int]:
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
                           deleting: Any = None,
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
    _commit_fut: Optional[asyncio.Future] = None

    #: Set of unacked messages: that is messages that we started processing
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

    def _set_active_tps(self, tps: Set[TP]) -> Set[TP]:
        xtps: Set[TP] = self._active_partitions = ensure_TPset(tps)
        xtps.difference_update(self._paused_partitions)
        return xtps

    def on_buffer_full(self, tp: TP) -> None:
        active_partitions = self._get_active_partitions()
        active_partitions.discard(tp)
        self._buffered_partitions.add(tp)

    def on_buffer_drop(self, tp: TP) -> None:
        if tp in self._buffered_partitions:
            active_partitions = self._get_active_partitions()
            active_partitions.add(tp)
            self._buffered_partitions.discard(tp)

    @abc.abstractmethod
    async def _commit(self,
                      offsets: Mapping[TP, int]) -> bool:  # pragma: no cover
        ...

    async def perform_seek(self) -> None:
        """Seek all partitions to their current committed position."""
        read_offset = self._read_offset
        _committed_offsets: Mapping[TP, int] = await self.seek_to_committed()
        read_offset.update({
            tp: offset if offset is not None and offset >= 0 else None
            for tp, offset in _committed_offsets.items()
        })
        committed_offsets: Mapping[TP, Optional[int]] = {
            ensure_TP(tp): offset if offset else None
            for tp, offset in _committed_offsets.items()
            if offset is not None
        }
        self._committed_offset.update(committed_offsets)

    @abc.abstractmethod
    async def seek_to_committed(self) -> Mapping[TP, int]:
        """Seek all partitions to their committed offsets."""
        ...

    async def seek(self, partition: TP, offset: int) -> None:
        """Seek partition to specific offset."""
        self.log.dev('SEEK %r -> %r', partition, offset)
        await self._seek(partition, offset)
        self._read_offset[ensure_TP(partition)] = offset if offset else None

    @abc.abstractmethod
    async def _seek(self, partition: TP, offset: int) -> None:
        ...

    @abc.abstractmethod
    def _new_topicpartition(self, topic: str, partition: int) -> TP:  # pragma: no cover
        ...

    def _is_changelog_tp(self, tp: TP) -> bool:
        return tp.topic in self.app.tables.changelog_topics

    @Service.transitions_to(CONSUMER_PARTITIONS_REVOKED)
    async def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        span = self.app._start_span_from_rebalancing('on_partitions_revoked')
        T = traced_from_parent_span(span)
        with span:
            if self._active_partitions is not None:
                self._active_partitions.difference_update(revoked)
            self._paused_partitions.difference_update(revoked)
            await T(self._on_partitions_revoked, partitions=revoked)(revoked)

    @Service.transitions_to(CONSUMER_PARTITIONS_ASSIGNED)
    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        span = self.app._start_span_from_rebalancing('on_partitions_assigned')
        T = traced_from_parent_span(span)
        with span:
            self._paused_partitions.intersection_update(assigned)
            self._set_active_tps(assigned)
            await T(self._on_partitions_assigned, partitions=assigned)(assigned)
        self.app.on_rebalance_return()

    @abc.abstractmethod
    async def _getmany(self,
                       active_partitions: Optional[Set[TP]],
                       timeout: float) -> RecordMap:
        ...

    async def getmany(self, timeout: float) -> AsyncIterator[Tuple[TP, Message]]:
        """Fetch batch of messages from server."""
        records, active_partitions = await self._wait_next_records(timeout)
        if records is None or self.should_stop:
            return

        records_it: Iterator[Tuple[TP, Any]] = self.scheduler.iterate(records)
        to_message = self._to_message
        if self.flow_active:
            for tp, record in records_it:
                if not self.flow_active:
                    break
                if active_partitions is None or tp in active_partitions:
                    highwater_mark = self.highwater(tp)
                    self.app.monitor.track_tp_end_offset(tp, highwater_mark)
                    yield tp, to_message(tp, record)

    async def _wait_next_records(
            self, timeout: float) -> Tuple[Optional[RecordMap], Optional[Set[TP]]]:
        if not self.flow_active:
            await self.wait(self.can_resume_flow)
        is_client_only: bool = self.app.client_only
        active_partitions: Optional[Set[TP]]
        if is_client_only:
            active_partitions = None
        else:
            active_partitions = self._get_active_partitions()

        records: RecordMap = {}
        if is_client_only or active_partitions:
            records = await self._getmany(
                active_partitions=active_partitions,
                timeout=timeout,
            )
        else:
            await self.sleep(1)
        return records, active_partitions

    @abc.abstractmethod
    def _to_message(self, tp: TP, record: Any) -> ConsumerMessage:
        ...

    def track_message(self, message: Message) -> None:
        """Track message and mark it as pending ack."""
        self._unacked_messages.add(message)
        self._on_message_in(message.tp, message.offset, message)

    def ack(self, message: Message) -> bool:
        """Mark message as being acknowledged by stream."""
        if not message.acked:
            message.acked = True
            tp: TP = message.tp
            offset: int = message.offset
            if self.app.topics.acks_enabled_for(message.topic):
                committed = self._committed_offset[tp]
                try:
                    if committed is None or offset > committed:
                        acked_index = self._acked_index[tp]
                        if offset not in acked_index:
                            self._unacked_messages.discard(message)
                            acked_index.add(offset)
                            self._acked[tp].append(offset)
                            self._n_acked += 1
                            return True
                finally:
                    notify(self._waiting_for_ack)
        return False

    async def _wait_for_ack(self, timeout: float) -> None:
        self._waiting_for_ack = asyncio.Future(loop=self.loop)
        try:
            await asyncio.wait_for(
                self._waiting_for_ack, loop=self.loop, timeout=1)
        except (asyncio.TimeoutError,
                asyncio.CancelledError):
            pass
        finally:
            self._waiting_for_ack = None

    @Service.transitions_to(CONSUMER_WAIT_EMPTY)
    async def wait_empty(self) -> None:
        """Wait for all messages that started processing to be acked."""
        wait_count: int = 0
        T = traced_from_parent_span()
        while not self.should_stop and self._unacked_messages:
            wait_count += 1
            if not wait_count % 10:
                remaining = [(m.refcount, m) for m in self._unacked_messages]
                self.log.warning('wait_empty: Waiting for tasks %r', remaining)
                self.log.info(
                    'Agent tracebacks:\n%s',
                    self.app.agents.human_tracebacks(),
                )
            self.log.dev('STILL WAITING FOR ALL STREAMS TO FINISH')
            self.log.dev('WAITING FOR %r EVENTS', len(self._unacked_messages))
            gc.collect()
            await T(self.commit)()
            if not self._unacked_messages:
                break
            await T(self._wait_for_ack)(timeout=1)
            self._clean_unacked_messages()

        self.log.dev('COMMITTING AGAIN AFTER STREAMS DONE')
        await T(self.commit_and_end_transactions)()

    def _clean_unacked_messages(self) -> None:
        self._unacked_messages -= {message for message in self._unacked_messages if message.acked}

    async def commit_and_end_transactions(self) -> None:
        """Commit all safe offsets and end transaction."""
        await self.commit(start_new_transaction=False)

    async def on_stop(self) -> None:
        """Call when consumer is stopping."""
        if self.app.conf.stream_wait_empty:
            await self.wait_empty()
        else:
            await self.commit_and_end_transactions()

    @Service.task
    async def _commit_handler(self) -> None:
        interval: float = self.commit_interval
        await self.sleep(interval)
        async for sleep_time in self.itertimer(interval, name='commit'):
            await self.commit()

    @Service.task
    async def _commit_livelock_detector(self) -> None:  # pragma: no cover
        interval: float = self.commit_interval * 2.5
        await self.sleep(interval)
        async for sleep_time in self.itertimer(interval, name='livelock'):
            if not self.app.rebalancing:
                await self.verify_all_partitions_active()

    async def verify_all_partitions_active(self) -> None:
        now: float = monotonic()
        for tp in self.assignment():
            await self.sleep(0)
            if not self.should_stop:
                self.verify_event_path(now, tp)

    def verify_event_path(self, now: float, tp: TP) -> None:
        ...

    def verify_recovery_event_path(self, now: float, tp: TP) -> None:
        ...

    async def commit(self, topics: Optional[TPorTopicSet] = None,
                     start_new_transaction: bool = True) -> bool:
        """Maybe commit the offset for all or specific topics."""
        if self.app.client_only:
            return False
        if await self.maybe_wait_for_commit_to_finish():
            return False

        self._commit_fut = asyncio.Future(loop=self.loop)
        try:
            return await self.force_commit(
                topics,
                start_new_transaction=start_new_transaction,
            )
        finally:
            fut, self._commit_fut = self._commit_fut, None
            notify(fut)

    async def maybe_wait_for_commit_to_finish(self) -> bool:
        """Wait for any existing commit operation to finish."""
        if self._commit_fut is not None:
            try:
                await self._commit_fut
            except asyncio.CancelledError:
                pass
            else:
                return True
        return False

    @Service.transitions_to(CONSUMER_COMMITTING)
    async def force_commit(self,
                           topics: Optional[TPorTopicSet] = None,
                           start_new_transaction: bool = True) -> bool:
        """Force offset commit."""
        sensor_state = self.app.sensors.on_commit_initiated(self)
        commit_tps: List[TP] = list(self._filter_tps_with_pending_acks(topics))
        did_commit: bool = await self._commit_tps(
            commit_tps, start_new_transaction=start_new_transaction)
        self.app.sensors.on_commit_completed(self, sensor_state)
        return did_commit

    async def _commit_tps(self,
                          tps: Iterable[TP],
                          start_new_transaction: bool) -> bool:
        commit_offsets: Dict[TP, int] = self._filter_committable_offsets(tps)
        if commit_offsets:
            try:
                await self._handle_attached(commit_offsets)
            except ProducerSendError as exc:
                await self.crash(exc)
            else:
                return await self._commit_offsets(
                    commit_offsets,
                    start_new_transaction=start_new_transaction)
        return False

    def _filter_committable_offsets(self, tps: Iterable[TP]) -> Dict[TP, int]:
        commit_offsets: Dict[TP, int] = {}
        for tp in tps:
            offset: Optional[int] = self._new_offset(tp)
            if offset is not None and self._should_commit(tp, offset):
                commit_offsets[tp] = offset
        return commit_offsets

    async def _handle_attached(self, commit_offsets: Dict[TP, int]) -> None:
        for tp, offset in commit_offsets.items():
            app = cast(_App, self.app)
            attachments = app._attachments
            producer = app.producer
            pending = await attachments.publish_for_tp_offset(tp, offset)
            if pending:
                await cast(Service, producer).wait_many(pending)

    async def _commit_offsets(self, offsets: Mapping[TP, int],
                              start_new_transaction: bool = True) -> bool:
        table = terminal.logtable(
            [(str(tp), str(offset))
             for tp, offset in offsets.items()],
            title='Commit Offsets',
            headers=['TP', 'Offset'],
        )
        self.log.dev('COMMITTING OFFSETS:\n%s', table)
        assignment = self.assignment()
        committable_offsets: Dict[TP, int] = {}
        revoked: Dict[TP, int] = {}
        for tp, offset in offsets.items():
            if tp in assignment:
                committable_offsets[tp] = offset
            else:
                revoked[tp] = offset
        if revoked:
            self.log.info(
                'Discarded commit for revoked partitions that '
                'will be eventually processed again: %r',
                revoked,
            )
        if not committable_offsets:
            return False
        with flight_recorder(self.log, timeout=300.0) as on_timeout:
            did_commit: bool = False
            on_timeout.info('+consumer.commit()')
            if self.in_transaction:
                did_commit = await self.transactions.commit(
                    committable_offsets,
                    start_new_transaction=start_new_transaction,
                )
            else:
                did_commit = await self._commit(committable_offsets)
            on_timeout.info('-consumer.commit()')
            if did_commit:
                on_timeout.info('+tables.on_commit')
                self.app.tables.on_commit(committable_offsets)
                on_timeout.info('-tables.on_commit')
        self._committed_offset.update(committable_offsets)
        self.app.monitor.on_tp_commit(committable_offsets)
        return did_commit

    def _filter_tps_with_pending_acks(
            self, topics: Optional[TPorTopicSet] = None) -> Iterator[TP]:
        return (tp for tp in self._acked
                if topics is None or tp in topics or tp.topic in topics)

    def _should_commit(self, tp: TP, offset: int) -> bool:
        committed = self._committed_offset[tp]
        return committed is None or bool(offset) and offset > committed

    def _new_offset(self, tp: TP) -> Optional[int]:
        acked: List[int] = self._acked[tp]
        if acked:
            max_offset: int = max(acked)
            gap_for_tp: List[int] = self._gap[tp]
            if gap_for_tp:
                gap_index = next((i for i, x in enumerate(gap_for_tp)
                                  if x > max_offset), len(gap_for_tp))
                gaps = gap_for_tp[:gap_index]
                acked.extend(gaps)
                gap_for_tp[:gap_index] = []
            acked.sort()
            batch = next(consecutive_numbers(acked))
            acked[:len(batch) - 1] = []
            self._acked_index[tp].difference_update(batch)
            return batch[-1]
        return None

    async def on_task_error(self, exc: BaseException) -> None:
        """Call when processing a message failed."""
        await self.commit()

    def _add_gap(self, tp: TP, offset_from: int, offset_to: int) -> None:
        committed = self._committed_offset[tp]
        gap_for_tp = self._gap[tp]
        for offset in range(offset_from, offset_to):
            if committed is None or offset > committed:
                gap_for_tp.append(offset)

    async def _drain_messages(self, fetcher: ServiceT) -> None:  # pragma: no cover
        callback = self.callback
        getmany = self.getmany
        consumer_should_stop = cast(Service, self)._stopped.is_set
        fetcher_should_stop = cast(Service, fetcher)._stopped.is_set

        get_read_offset = self._read_offset.__getitem__
        set_read_offset = self._read_offset.__setitem__
        flag_consumer_fetching = CONSUMER_FETCHING
        set_flag = self.diag.set_flag
        unset_flag = self.diag.unset_flag
        commit_every = self._commit_every
        acks_enabled_for = self.app.topics.acks_enabled_for

        yield_every: int = 100
        num_since_yield: int = 0
        sleep = asyncio.sleep

        try:
            while not (consumer_should_stop() or fetcher_should_stop()):
                set_flag(flag_consumer_fetching)
                ait: AsyncIterator[Tuple[TP, Message]] = cast(AsyncIterator, getmany(timeout=1.0))
                await self.sleep(0)
                if not self.should_stop:
                    async for tp, message in ait:
                        num_since_yield += 1
                        if num_since_yield > yield_every:
                            await sleep(0)
                            num_since_yield = 0

                        offset: int = message.offset
                        r_offset = get_read_offset(tp)
                        if r_offset is None or offset > r_offset:
                            gap = offset - (r_offset or 0)
                            if gap > 1 and r_offset:
                                acks_enabled = acks_enabled_for(message.topic)
                                if acks_enabled:
                                    self._add_gap(tp, r_offset + 1, offset)
                            if commit_every is not None:
                                if self._n_acked >= commit_every:
                                    self._n_acked = 0
                                    await self.commit()
                            await callback(message)
                            set_read_offset(tp, offset)
                        else:
                            self.log.dev('DROPPED MESSAGE ROFF %r: k=%r v=%r',
                                         offset, message.key, message.value)
                    unset_flag(flag_consumer_fetching)

        except self.consumer_stopped_errors:
            if self.transport.app.should_stop:
                self.log.info('Broker stopped consumer, shutting down...')
                return
            raise
        except asyncio.CancelledError:
            if self.transport.app.should_stop:
                self.log.info('Consumer shutting down for user cancel.')
                return
            raise
        except Exception as exc:
            self.log.exception('Drain messages raised: %r', exc)
            raise
        finally:
            unset_flag(flag_consumer_fetching)

    def close(self) -> None:
        """Close consumer for graceful shutdown."""
        ...

    @property
    def unacked(self) -> Set[Message]:
        """Return the set of currently unacknowledged messages."""
        return cast(Set[Message], self._unacked_messages)


class ConsumerThread(QueueServiceThread):
    """Consumer running in a dedicated thread."""
    app: AppT
    consumer: 'ThreadDelegateConsumer'
    transport: TransportT

    def __init__(self, consumer: 'ThreadDelegateConsumer',
                 **kwargs: Any) -> None:
        self.consumer = consumer
        self.transport = self.consumer.transport
        self.app = self.transport.app
        super().__init__(**kwargs)

    @abc.abstractmethod
    async def subscribe(self, topics: Iterable[str]) -> None:
        """Reset subscription (requires rebalance)."""
        ...

    @abc.abstractmethod
    async def seek_to_committed(self) -> Mapping[TP, int]:
        """Seek all partitions to their committed offsets."""
        ...

    @abc.abstractmethod
    async def commit(self, tps: Mapping[TP, int]) -> bool:
        """Commit offsets in topic partitions."""
        ...

    @abc.abstractmethod
    async def position(self, tp: TP) -> Optional[int]:
        """Return the current offset for partition."""
        ...

    @abc.abstractmethod
    async def seek_to_beginning(self, *partitions: TP) -> None:
        """Seek to the earliest offsets available for partitions."""
        ...

    @abc.abstractmethod
    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        """Seek partitions to specific offsets and wait."""
        ...

    @abc.abstractmethod
    def seek(self, partition: TP, offset: int) -> None:
        """Seek partition to specific offset."""
        ...

    @abc.abstractmethod
    def assignment(self) -> Set[TP]:
        """Return the current assignment."""
        ...

    @abc.abstractmethod
    def highwater(self, tp: TP) -> int:
        """Return the last available offset in partition."""
        ...

    @abc.abstractmethod
    def topic_partitions(self, topic: str) -> Optional[int]:
        """Return number of configured partitions for topic by name."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @abc.abstractmethod
    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]:
        """Return the earliest available offset for list of partitions."""
        ...

    @abc.abstractmethod
    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]:
        """Return the last available offset for list of partitions."""
        ...

    @abc.abstractmethod
    async def getmany(self,
                      active_partitions: Optional[Set[TP]],
                      timeout: float) -> RecordMap:
        """Fetch batch of messages from server."""
        ...

    @abc.abstractmethod
    async def create_topic(self,
                           topic: str,
                           partitions: int,
                           replication: int,
                           *,
                           config: Optional[Mapping[str, Any]] = None,
                           timeout: float = 30.0,
                           retention: Optional[Seconds] = None,
                           compacting: Optional[bool] = None,
                           deleting: Any = None,
                           ensure_created: bool = False) -> None:
        """Create/declare topic on server."""
        ...

    async def on_partitions_revoked(
            self, revoked: Set[TP]) -> None:
        """Call on rebalance when partitions are being revoked."""
        await self.consumer.threadsafe_partitions_revoked(
            self.thread_loop, revoked)

    async def on_partitions_assigned(
            self, assigned: Set[TP]) -> None:
        """Call on rebalance when partitions are being assigned."""
        await self.consumer.threadsafe_partitions_assigned(
            self.thread_loop, assigned)

    @abc.abstractmethod
    def key_partition(self,
                      topic: str,
                      key: Optional[bytes],
                      partition: Optional[int] = None) -> Optional[int]:
        """Hash key to determine partition number."""
        ...

    def verify_recovery_event_path(self, now: float, tp: TP) -> None:
        ...


class ThreadDelegateConsumer(Consumer):

    _thread: ConsumerThread

    _method_queue: MethodQueue

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._method_queue = MethodQueue(loop=self.loop, beacon=self.beacon)
        self.add_dependency(self._method_queue)
        self._thread = self._new_consumer_thread()
        self.add_dependency(self._thread)

    @abc.abstractmethod
    def _new_consumer_thread(self) -> ConsumerThread:
        ...

    async def threadsafe_partitions_revoked(
            self,
            receiver_loop: asyncio.AbstractEventLoop,
            revoked: Set[TP]) -> None:
        """Call rebalancing callback in a thread-safe manner."""
        promise = await self._method_queue.call(
            receiver_loop.create_future(),
            self.on_partitions_revoked,
            revoked,
        )
        await promise

    async def threadsafe_partitions_assigned(
            self,
            receiver_loop: asyncio.AbstractEventLoop,
            assigned: Set[TP]) -> None:
        """Call rebalancing callback in a thread-safe manner."""
        promise = await self._method_queue.call(
            receiver_loop.create_future(),
            self.on_partitions_assigned,
            assigned,
        )
        await promise

    async def _getmany(self,
                       active_partitions: Optional[Set[TP]],
                       timeout: float) -> RecordMap:
        return await self._thread.getmany(active_partitions, timeout)

    async def subscribe(self, topics: Iterable[str]) -> None:
        """Reset subscription (requires rebalance)."""
        await self._thread.subscribe(topics=topics)

    async def seek_to_committed(self) -> Mapping[TP, int]:
        """Seek all partitions to the committed offset."""
        return await self._thread.seek_to_committed()

    async def position(self, tp: TP) -> Optional[int]:
        """Return the current position for partition."""
        return await self._thread.position(tp)

    async def seek_wait(self, partitions: Mapping[TP, int]) -> None:
        """Seek partitions to specific offsets and wait."""
        return await self._thread.seek_wait(partitions)

    async def _seek(self, partition: TP, offset: int) -> None:
        self._thread.seek(partition, offset)

    def assignment(self) -> Set[TP]:
        """Return the current assignment."""
        return self._thread.assignment()

    def highwater(self, tp: TP) -> int:
        """Return the last available offset for specific partition."""
        return self._thread.highwater(tp)

    def topic_partitions(self, topic: str) -> Optional[int]:
        """Return the number of partitions configured for topic by name."""
        return self._thread.topic_partitions(topic)

    async def earliest_offsets(self, *partitions: TP) -> Mapping[TP, int]:
        """Return the earliest offsets for a list of partitions."""
        return await self._thread.earliest_offsets(*partitions)

    async def highwaters(self, *partitions: TP) -> Mapping[TP, int]:
        """Return the last offset for a list of partitions."""
        return await self._thread.highwaters(*partitions)

    async def _commit(self, offsets: Mapping[TP, int]) -> bool:
        return await self._thread.commit(offsets)

    def close(self) -> None:
        """Close consumer for graceful shutdown."""
        self._thread.close()

    def key_partition(self,
                      topic: str,
                      key: Optional[bytes],
                      partition: Optional[int] = None) -> Optional[int]:
        """Hash key to determine partition number."""
        return self._thread.key_partition(topic, key, partition=partition)

    def verify_recovery_event_path(self, now: float, tp: TP) -> None:
        return self._thread.verify_recovery_event_path(now, tp)
