from typing import Any, AsyncIterator, Awaitable, ClassVar, Dict, Iterable, Iterator, List, Mapping, MutableMapping, MutableSet, NamedTuple, Optional, Set, Tuple, Type, cast

class TopicPartitionGroup(NamedTuple):
    topic: str
    partition: int
    group: str

def ensure_TP(tp: Any) -> Any:
    return tp if isinstance(tp, TP) else TP(tp.topic, tp.partition)

def ensure_TPset(tps: Iterable[Any]) -> Set[Any]:
    return {ensure_TP(tp) for tp in tps}

class Fetcher(Service):
    logger: Any = logger
    _drainer: Optional[Any] = None

    def __init__(self, app: Any, **kwargs: Any) -> None:
        self.app = app
        super().__init__(**kwargs)

    async def on_stop(self) -> None:
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
            self._drainer = asyncio.ensure_future(consumer._drain_messages(self), loop=self.loop)
            await self._drainer
        except asyncio.CancelledError:
            pass
        finally:
            self.set_shutdown()

class TransactionManager(Service, TransactionManagerT):
    transactional_id_format: str = '{group_id}-{tpg.group}-{tpg.partition}'

    def __init__(self, transport: Any, *, consumer: Any, producer: Any, **kwargs: Any) -> None:
        self.transport = transport
        self.app = self.transport.app
        self.consumer = consumer
        self.producer = producer
        super().__init__(**kwargs)

    async def flush(self) -> None:
        await self.producer.flush()

    async def on_partitions_revoked(self, revoked: Any) -> None:
        await traced_from_parent_span()(self.flush)()

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> None:
        T = traced_from_parent_span()
        revoked_tids = sorted(self._tps_to_transactional_ids(revoked))
        if revoked_tids:
            self.log.info('Stopping %r transactional %s for %r revoked %s...', len(revoked_tids), pluralize(len(revoked_tids), 'producer'), len(revoked), pluralize(len(revoked), 'partition'))
            await T(self._stop_transactions, tids=revoked_tids)(revoked_tids)
        assigned_tids = sorted(self._tps_to_transactional_ids(assigned))
        if assigned_tids:
            self.log.info('Starting %r transactional %s for %r assigned %s...', len(assigned_tids), pluralize(len(assigned_tids), 'producer'), len(assigned), pluralize(len(assigned), 'partition'))
            await T(self._start_transactions, tids=assigned_tids)(assigned_tids)

    async def _stop_transactions(self, tids: Any) -> None:
        T = traced_from_parent_span()
        producer = self.producer
        for transactional_id in tids:
            await T(producer.stop_transaction)(transactional_id)

    async def _start_transactions(self, tids: Any) -> None:
        T = traced_from_parent_span()
        producer = self.producer
        for transactional_id in tids:
            await T(producer.maybe_begin_transaction)(transactional_id)

    def _tps_to_transactional_ids(self, tps: Any) -> Set[str]:
        return {self.transactional_id_format.format(tpg=tpg, group_id=self.app.conf.id) for tpg in self._tps_to_active_tpgs(tps)}

    def _tps_to_active_tpgs(self, tps: Any) -> Set[TopicPartitionGroup]:
        assignor = self.app.assignor
        return {TopicPartitionGroup(tp.topic, tp.partition, assignor.group_for_topic(tp.topic)) for tp in tps if not assignor.is_standby(tp)}

    async def send(self, topic: str, key: Any, value: Any, partition: int, timestamp: float, headers: Any, *, transactional_id: Optional[str] = None) -> Any:
        group = transactional_id = None
        p = self.consumer.key_partition(topic, key, partition)
        if p is not None:
            group = self.app.assignor.group_for_topic(topic)
            transactional_id = f'{self.app.conf.id}-{group}-{p}'
        return await self.producer.send(topic, key, value, p, timestamp, headers, transactional_id=transactional_id)

    def send_soon(self, fut: Any) -> None:
        raise NotImplementedError()

    async def send_and_wait(self, topic: str, key: Any, value: Any, partition: int, timestamp: float, headers: Any, *, transactional_id: Optional[str] = None) -> Any:
        fut = await self.send(topic, key, value, partition, timestamp, headers)
        return await fut

    async def commit(self, offsets: Any, start_new_transaction: bool = True) -> bool:
        producer = self.producer
        group_id = self.app.conf.id
        by_transactional_id = defaultdict(dict)
        for tp, offset in offsets.items():
            group = self.app.assignor.group_for_topic(tp.topic)
            transactional_id = f'{group_id}-{group}-{tp.partition}'
            by_transactional_id[transactional_id][tp] = offset
        if by_transactional_id:
            await producer.commit_transactions(by_transactional_id, group_id, start_new_transaction=start_new_transaction)
        return True

    def key_partition(self, topic: str, key: Any) -> Any:
        raise NotImplementedError()

    async def create_topic(self, topic: str, partitions: int, replication: int, *, config: Any = None, timeout: float = 30.0, retention: Any = None, compacting: Any = None, deleting: Any = None, ensure_created: bool = False) -> Any:
        return await self.producer.create_topic(topic, partitions, replication, config=config, timeout=timeout, retention=retention, compacting=compacting, deleting=deleting, ensure_created=ensure_created)

    def supports_headers(self) -> bool:
        return self.producer.supports_headers()

class Consumer(Service, ConsumerT):
    logger: Any = logger
    consumer_stopped_errors: Tuple = ()
    _waiting_for_ack: Optional[Any] = None
    _commit_fut: Optional[Any] = None
    _n_acked: int = 0
    flow_active: bool = True

    def __init__(self, transport: Any, callback: Any, on_partitions_revoked: Any, on_partitions_assigned: Any, *, commit_interval: Optional[float] = None, commit_livelock_soft_timeout: Optional[float] = None, loop: Any = None, **kwargs: Any) -> None:
        assert callback is not None
        self.transport = transport
        self.app = self.transport.app
        self.in_transaction = self.app.in_transaction
        self.callback = callback
        self._on_message_in = self.app.sensors.on_message_in
        self._on_partitions_revoked = on_partitions_revoked
        self._on_partitions_assigned = on_partitions_assigned
        self._commit_every = self.app.conf.broker_commit_every
        self.scheduler = self.app.conf.ConsumerScheduler()
        self.commit_interval = commit_interval or self.app.conf.broker_commit_interval
        self.commit_livelock_soft_timeout = commit_livelock_soft_timeout or self.app.conf.broker_commit_livelock_soft_timeout
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
        self.transactions = self.transport.create_transaction_manager(consumer=self, producer=self.app.producer, beacon=self.beacon, loop=self.loop)

    def on_init_dependencies(self) -> List[Any]:
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
        self._reset_state()
        self.on_init()

    def _get_active_partitions(self) -> Set[Any]:
        tps = self._active_partitions
        if tps is None:
            return self._set_active_tps(self.assignment())
        assert all((isinstance(x, TP) for x in tps))
        return tps

    def _set_active_tps(self, tps: Iterable[Any]) -> Set[Any]:
        xtps = self._active_partitions = ensure_TPset(tps)
        xtps.difference_update(self._paused_partitions)
        return xtps

    def on_buffer_full(self, tp: Any) -> None:
        active_partitions = self._get_active_partitions()
        active_partitions.discard(tp)
        self._buffered_partitions.add(tp)

    def on_buffer_drop(self, tp: Any) -> None:
        buffered_partitions = self._buffered_partitions
        if tp in buffered_partitions:
            active_partitions = self._get_active_partitions()
            active_partitions.add(tp)
            buffered_partitions.discard(tp)

    @abc.abstractmethod
    async def _commit(self, offsets: Any) -> None:
        ...

    async def perform_seek(self) -> None:
        read_offset = self._read_offset
        _committed_offsets = await self.seek_to_committed()
        read_offset.update({tp: offset if offset is not None and offset >= 0 else None for tp, offset in _committed_offsets.items()})
        committed_offsets = {ensure_TP(tp): offset if offset else None for tp, offset in _committed_offsets.items() if offset is not None}
        self._committed_offset.update(committed_offsets)

    @abc.abstractmethod
    async def seek_to_committed(self) -> Mapping[Any, Any]:
        ...

    async def seek(self, partition: Any, offset: Any) -> None:
        self._seek(partition, offset)
        self._read_offset[ensure_TP(partition)] = offset if offset else None

    @abc.abstractmethod
    async def _seek(self, partition: Any, offset: Any) -> None:
        ...

    def stop_flow(self) -> None:
        self.flow_active = False
        self.can_resume_flow.clear()

    def resume_flow(self) -> None:
        self.flow_active = True
        self.can_resume_flow.set()

    def pause_partitions(self, tps: Iterable[Any]) -> None:
        tpset = ensure_TPset(tps)
        self._get_active_partitions().difference_update(tpset)
        self._paused_partitions.update(tpset)

    def resume_partitions(self, tps: Iterable[Any]) -> None:
        tpset = ensure_TPset(tps)
        self._get_active_partitions().update(tps)
        self._paused_partitions.difference_update(tpset)

    @abc.abstractmethod
    def _new_topicpartition(self, topic: str, partition: int) -> Any:
        ...

    def _is_changelog_tp(self, tp: Any) -> bool:
        return tp.topic in self.app.tables.changelog_topics

    @Service.transitions_to(CONSUMER_PARTITIONS_REVOKED)
    async def on_partitions_revoked(self, revoked: Any) -> None:
        span = self.app._start_span_from_rebalancing('on_partitions_revoked')
        T = traced_from_parent_span(span)
        with span:
            if self._active_partitions is not None:
                self._active_partitions.difference_update(revoked)
            self._paused_partitions.difference_update(revoked)
            await T(self._on_partitions_revoked, partitions=revoked)(revoked)

    @Service.transitions_to(CONSUMER_PARTITIONS_ASSIGNED)
    async def on_partitions_assigned(self, assigned: Any) -> None:
        span = self.app._start_span_from_rebalancing('on_partitions_assigned')
        T = traced_from_parent_span(span)
        with span:
            self._paused_partitions.intersection_update(assigned)
            self._set_active_tps(assigned)
            await T(self._on_partitions_assigned, partitions=assigned)(assigned)
        self.app.on_rebalance_return()

    @abc.abstractmethod
    async def _getmany(self, active_partitions: Any, timeout: float) -> Any:
        ...

    async def getmany(self, timeout: float) -> AsyncIterator[Tuple[Any, Any]]:
        records, active_partitions = await self._wait_next_records(timeout)
        if records is None or self.should_stop:
            return
        records_it = self.scheduler.iterate(records)
        to_message = self._to_message
        if self.flow_active:
            for tp, record in records_it:
                if not self.flow_active:
                    break
                if active_partitions is None or tp in active_partitions:
                    highwater_mark = self.highwater(tp)
                    self.app.monitor.track_tp_end_offset(tp, highwater_mark)
                    yield (tp, to_message(tp, record))

    async def _wait_next_records(self, timeout: float) -> Tuple[Any, Any]:
        if not self.flow_active:
            await self.wait(self.can_resume_flow)
        is_client_only = self.app.client_only
        if is_client_only:
            active_partitions = None
        else:
            active_partitions = self._get_active_partitions()
        records = {}
        if is_client_only or active_partitions:
            records = await self._getmany(active_partitions=active_partitions, timeout=timeout)
        else:
            await self.sleep(1)
        return (records, active_partitions)

    @abc.abstractmethod
    def _to_message(self, tp: Any, record: Any) -> Any:
        ...

    def track_message(self, message: Any) -> None:
        self._unacked_messages.add(message)
        self._on_message_in(message.tp, message.offset, message)

    def ack(self, message: Any) -> bool:
        if not message.acked:
            message.acked = True
            tp = message.tp
            offset = message.offset
            if self.app.topics.acks_enabled_for(message.topic):
                committed = self._committed_offset[tp]
                try:
                    if committed is None or offset > committed:
                        acked_index = self._acked_index[tp]
                        if offset not in acked_index:
                            self._unacked_messages.discard(message)
                            acked_index.add(offset)
                            acked_for_tp = self._acked[tp]
                            acked_for_tp.append(offset)
                            self._n_acked += 1
                            return True
                finally:
                    notify(self._waiting_for_ack)
        return False

    async def _wait_for_ack(self, timeout: float) -> None:
        self._waiting_for_ack = asyncio.Future(loop=self.loop)
        try:
            await asyncio.wait_for(self._waiting_for_ack, loop=self.loop, timeout=1)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self._waiting_for_ack = None

    @Service.transitions_to(CONSUMER_WAIT_EMPTY)
    async def wait_empty(self) -> None:
        wait_count = 0
        T = traced_from_parent_span()
        while not self.should_stop and self._unacked_messages:
            wait_count += 1
            if not wait_count % 10:
                remaining = [(m.refcount, m) for m in self._unacked_messages]
                self.log.warning('wait_empty: Waiting for tasks %r', remaining)
                self.log.info('Agent tracebacks:\n%s', self.app.agents.human_tracebacks())
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
        await self.commit(start_new_transaction=False)

    async def on_stop(self) -> None:
        if self.app.conf.stream_wait_empty:
            await self.wait_empty()
        else:
            await self.commit_and_end_transactions()

    @Service.task
    async def _commit_handler(self) -> None:
        interval = self.commit_interval
        await self.sleep(interval)
        async for sleep_time in self.itertimer(interval, name='commit'):
            await self.commit()

    @Service.task
    async def _commit_livelock_detector(self) -> None:
        interval = self.commit_interval * 2.5
        await self.sleep(interval)
        async for sleep_time in self.itertimer(interval, name='livelock'):
            if not self.app.rebalancing:
                await self.verify_all_partitions_active()

    async def verify_all_partitions_active(self) -> None:
        now = monotonic()
        for tp in self.assignment():
            await self.sleep(0)
            if not self.should_stop:
                self.verify_event_path(now, tp)

    def verify_event_path(self, now: float, tp: Any) -> None:
        ...

    def verify_recovery_event_path(self, now: float, tp: Any) -> None:
        ...

    async def commit(self, topics: Optional[Set[Any]] = None, start_new_transaction: bool = True) -> bool:
        if self.app.client_only:
            return False
        if await self.maybe_wait_for_commit_to_finish():
            return False
        self._commit_fut = asyncio.Future(loop=self.loop)
        try:
            return await self.force_commit(topics, start_new_transaction=start_new_transaction)
        finally:
            fut, self._commit_fut = (self._commit_fut, None)
            notify(fut)

    async def maybe_wait_for_commit_to_finish(self) -> bool:
        if self._commit_fut is not None:
            try:
                await self._commit_fut
            except asyncio.CancelledError:
                pass
            else:
                return True
        return False

    @Service.transitions_to(CONSUMER_COMMITTING)
    async