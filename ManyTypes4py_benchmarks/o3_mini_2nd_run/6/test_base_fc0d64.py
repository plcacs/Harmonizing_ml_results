from typing import Any, Dict, Callable, Iterable, List, Set, Tuple, Optional, Union
import asyncio
import operator
from copy import copy

import pytest
from faust import joins
from faust import Event, Record, Stream, Topic
from faust.exceptions import PartitionsMismatch
from faust.stores.base import Store
from faust.tables.base import Collection
from faust.types import TP
from faust.windows import Window
from mode import label, shortlabel
from mode.utils.mocks import AsyncMock, Mock, call, patch

TP1: TP = TP('foo', 0)


class User(Record):
    pass


class MyTable(Collection):
    app: Any
    name: str
    default: Any
    schema: Any
    key_type: Any
    value_type: Any
    extra_topic_configs: Any
    partitions: Any
    recovery_buffer_size: Any
    standby_buffer_size: Any
    use_partitioner: bool
    _recover_callbacks: List[Callable[[], Any]]
    _data: Any
    _store: Store
    _changelog_topic: Topic
    changelog_topic: Topic
    window: Optional[Window]
    _partition_timestamps: Dict[TP, List[float]]
    _partition_timestamp_keys: Dict[Tuple[Any, float], Any]
    _on_changelog_event: Optional[Callable[[Event], Any]]
    _on_window_close: Optional[Callable[[Any, Any], Any]]
    _changelog_compacting: bool
    _changelog_deleting: bool
    _partition_latest_timestamp: Dict[Any, float]
    sleep: Callable[..., Any]
    last_closed_window: Any
    _should_expire_keys: Callable[[], bool]
    data: Any

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.datas: Dict[Any, Any] = {}

    def _has_key(self, key: Any) -> bool:
        return key in self.datas

    def _get_key(self, key: Any) -> Any:
        return self.datas.get(key)

    def _set_key(self, key: Any, value: Any) -> None:
        self.datas[key] = value

    def _del_key(self, key: Any) -> None:
        self.datas.pop(key, None)

    def hopping(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def tumbling(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def using_window(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def as_ansitable(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def _send_changelog(
        self,
        event: Event,
        k: Any,
        v: Any,
        key_serializer: str = 'json',
        value_serializer: str = 'json'
    ) -> None:
        if event is None:
            raise RuntimeError()
        self.changelog_topic.send_soon(
            key=k,
            value=v,
            partition=event.message.partition,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            callback=self._on_changelog_sent,
            eager_partitioning=True,
        )

    def _on_changelog_sent(self, fut: asyncio.Future) -> None:
        result = fut.result()
        if self.app.in_transaction:
            self.app.tables.persist_offset_on_commit(self.data, result.topic_partition, result.offset)
        else:
            self._data.set_persisted_offset(result.topic_partition, result.offset)

    async def _del_old_keys(self) -> None:
        # Dummy implementation for type annotation purposes.
        await asyncio.sleep(0)

    async def on_window_close(self, key: Any, value: Any) -> None:
        if self._on_window_close is None:
            return
        ret = self._on_window_close(key, value)
        if asyncio.iscoroutine(ret):
            await ret

    def _verify_source_topic_partitions(self, topic: str) -> None:
        source_n: Optional[int] = self.app.consumer.topic_partitions(topic)
        change_topic: str = self.changelog_topic.get_topic_name()  # type: ignore
        change_n: Optional[int] = self.app.consumer.topic_partitions(change_topic)
        if source_n is not None and change_n is not None and source_n != change_n:
            raise PartitionsMismatch()

    async def call_recover_callbacks(self) -> None:
        for callback in self._recover_callbacks:
            await callback()

    async def on_recovery_completed(self, set1: Set[Any], set2: Set[Any]) -> None:
        await self.call_recover_callbacks()

    async def on_start(self) -> None:
        await self.changelog_topic.maybe_declare()

    def info(self) -> Dict[str, Any]:
        return {
            'app': self.app,
            'name': self.name,
            'store': self._store,
            'default': self.default,
            'schema': self.schema,
            'key_type': self.key_type,
            'value_type': self.value_type,
            'changelog_topic': self._changelog_topic,
            'window': self.window,
            'extra_topic_configs': self.extra_topic_configs,
            'on_changelog_event': getattr(self, '_on_changelog_event', None),
            'recover_callbacks': self._recover_callbacks,
            'partitions': self.partitions,
            'recovery_buffer_size': self.recovery_buffer_size,
            'standby_buffer_size': self.standby_buffer_size,
            'use_partitioner': self.use_partitioner,
        }

    def persisted_offset(self, tp: TP) -> Any:
        return self._data.persisted_offset()

    async def need_active_standby_for(self, tp: TP) -> Any:
        return await self._data.need_active_standby_for(tp)

    def reset_state(self) -> None:
        self._data.reset_state()

    def _join(self, join_strategy: Any) -> Any:
        raise NotImplementedError()

    def join(self, *fields: Any) -> Any:
        return self._join(joins.RightJoin(stream=self, fields=fields))

    def left_join(self, *fields: Any) -> Any:
        return self._join(joins.LeftJoin(stream=self, fields=fields))

    def inner_join(self, *fields: Any) -> Any:
        return self._join(joins.InnerJoin(stream=self, fields=fields))

    def outer_join(self, *fields: Any) -> Any:
        return self._join(joins.OuterJoin(stream=self, fields=fields))

    def clone(self) -> "MyTable":
        t2 = self.__class__(self.app, name=self.name)
        return t2

    def combine(self, joinable: Stream) -> Any:
        raise NotImplementedError()

    def contribute_to_stream(self, stream: Stream) -> None:
        pass

    async def remove_from_stream(self, stream: Stream) -> None:
        pass

    def _new_changelog_topic(
        self,
        retention: Optional[float] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None
    ) -> Topic:
        topic = Topic(name=f'{self.name}_changelog')  # Dummy topic creation.
        if self.window is not None and getattr(self.window, "expires", None) is not None:
            topic.retention = self.window.expires  # type: ignore
        else:
            topic.retention = retention
        if compacting is None:
            topic.compacting = self._changelog_compacting
        else:
            topic.compacting = compacting
        if deleting is None:
            topic.deleting = self._changelog_deleting
        else:
            topic.deleting = deleting
        return topic

    def _maybe_set_key_ttl(self, key: Any, timestamp: float) -> None:
        if self._should_expire_keys():
            pass

    def _maybe_del_key_ttl(self, key: Any, timestamp: float) -> None:
        if (0, timestamp) not in self._partition_timestamp_keys:
            pass
        elif self._should_expire_keys():
            keys_set = self._partition_timestamp_keys.get((0, timestamp))
            if keys_set and key in keys_set:
                keys_set.discard(key)

    def _apply_window_op(self, op: Callable[[Any, Any], Any], key: Any, operand: Any, timestamp: float) -> None:
        for r in self._window_ranges(timestamp):
            current = self._get_key((key, r))
            if current is not None:
                self._set_key((key, r), op(current, operand))

    def _set_windowed(self, key: Any, value: Any, timestamp: float) -> None:
        for r in self._window_ranges(timestamp):
            self._set_key((key, r), value)

    def _del_windowed(self, key: Any, timestamp: float) -> None:
        for r in self._window_ranges(timestamp):
            self._del_key((key, r))

    def _window_ranges(self, timestamp: float) -> Iterable[Any]:
        return self.window.ranges(timestamp)  # type: ignore

    def _relative_now(self, event: Optional[Event]) -> float:
        if event is not None:
            return self._partition_latest_timestamp.get(event.message.partition, 0.0)
        else:
            import time
            return time.time()

    def _relative_event(self, event: Optional[Event]) -> float:
        if event is None:
            raise RuntimeError()
        return event.message.timestamp

    def _relative_field(self, field: Any) -> Callable[[Optional[Event]], Any]:
        def _inner(event: Optional[Event]) -> Any:
            if event is None:
                raise RuntimeError()
            return getattr(event.value, field)
        return _inner

    def _relative_timestamp(self, timestamp: float) -> Callable[[Event], float]:
        def _inner(event: Event) -> float:
            return timestamp
        return _inner

    def _windowed_now(self, key: Any) -> None:
        t = self.window.earliest()  # type: ignore
        self._get_key((key, t))

    def _windowed_timestamp(self, key: Any, timestamp: float) -> Optional[Any]:
        current = self._get_key((key, self.window.current()))  # type: ignore
        return current

    def _windowed_contains(self, key: Any, timestamp: float) -> bool:
        return self._windowed_timestamp(key, timestamp) is not None

    def _windowed_delta(self, key: Any, timestamp: float, *, event: Event) -> Any:
        return self._get_key((key, self.window.delta(timestamp, event=event)))  # type: ignore

    async def on_rebalance(self, partitions: Set[TP], set1: Set[Any], set2: Set[Any]) -> None:
        await self._data.on_rebalance(self, partitions, set1, set2)

    async def on_changelog_event(self, event: Event) -> None:
        if self._on_changelog_event is None:
            return
        await self._on_changelog_event(event)

    def _to_key(self, key: Any) -> Any:
        if isinstance(key, list):
            return tuple(key)
        return key

    def _to_value(self, value: Any) -> Any:
        return value

    def _human_channel(self) -> str:
        return str(self)

    def _repr_info(self) -> str:
        return self.name

    def partition_for_key(self, key: Any) -> Optional[int]:
        if self.use_partitioner:
            partition: Optional[int] = None  # Dummy partition logic.
            return partition
        return None


class test_Collection:
    @pytest.fixture
    def table(self, *, app: Any) -> MyTable:
        return MyTable(app, name='name')

    def test_key_type_bytes_implies_raw_serializer(self, *, app: Any) -> None:
        table = MyTable(app, name='name', key_type=bytes)
        assert table.key_serializer == 'raw'

    @pytest.mark.asyncio
    async def test_init_on_recover(self, *, app: Any) -> None:
        on_recover: AsyncMock = AsyncMock(name='on_recover')
        t = MyTable(app, name='name', on_recover=on_recover)
        assert on_recover in t._recover_callbacks
        await t.call_recover_callbacks()
        on_recover.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_recovery_completed(self, *, table: MyTable) -> None:
        table.call_recover_callbacks = AsyncMock()
        await table.on_recovery_completed(set(), set())
        table.call_recover_callbacks.assert_called_once_with()

    def test_hash(self, *, table: MyTable) -> None:
        assert hash(table)

    @pytest.mark.asyncio
    async def test_on_start(self, *, table: MyTable) -> None:
        table.changelog_topic = Mock(name='changelog_topic', autospec=Topic, maybe_declare=AsyncMock())
        await table.on_start()
        table.changelog_topic.maybe_declare.assert_called_once_with()

    def test_info(self, *, table: MyTable) -> None:
        assert table.info() == {
            'app': table.app,
            'name': table.name,
            'store': table._store,
            'default': table.default,
            'schema': table.schema,
            'key_type': table.key_type,
            'value_type': table.value_type,
            'changelog_topic': table._changelog_topic,
            'window': table.window,
            'extra_topic_configs': table.extra_topic_configs,
            'on_changelog_event': table._on_changelog_event,
            'recover_callbacks': table._recover_callbacks,
            'partitions': table.partitions,
            'recovery_buffer_size': table.recovery_buffer_size,
            'standby_buffer_size': table.standby_buffer_size,
            'use_partitioner': table.use_partitioner,
        }

    def test_persisted_offset(self, *, table: MyTable) -> None:
        data = table._data = Mock(name='_data')
        assert table.persisted_offset(TP1) == data.persisted_offset()

    @pytest.mark.asyncio
    async def test_need_active_standby_for(self, *, table: MyTable) -> None:
        table._data = Mock(name='_data', autospec=Store, need_active_standby_for=AsyncMock())
        await table.need_active_standby_for(TP1)
        table._data.need_active_standby_for.assert_called_with(TP1)

    def test_reset_state(self, *, table: MyTable) -> None:
        data = table._data = Mock(name='_data', autospec=Store)
        table.reset_state()
        data.reset_state.assert_called_once_with()

    def test_send_changelog(self, *, table: MyTable) -> None:
        table.changelog_topic.send_soon = Mock(name='send_soon')
        event = Mock(name='event')
        table._send_changelog(event, 'k', 'v')
        table.changelog_topic.send_soon.assert_called_once_with(
            key='k',
            value='v',
            partition=event.message.partition,
            key_serializer='json',
            value_serializer='json',
            callback=table._on_changelog_sent,
            eager_partitioning=True,
        )

    def test_send_changelog__custom_serializers(self, *, table: MyTable) -> None:
        event = Mock(name='event')
        table.changelog_topic.send_soon = Mock(name='send_soon')
        table._send_changelog(event, 'k', 'v', key_serializer='raw', value_serializer='raw')
        table.changelog_topic.send_soon.assert_called_once_with(
            key='k',
            value='v',
            partition=event.message.partition,
            key_serializer='raw',
            value_serializer='raw',
            callback=table._on_changelog_sent,
            eager_partitioning=True,
        )

    def test_send_changelog__no_current_event(self, *, table: MyTable) -> None:
        with pytest.raises(RuntimeError):
            table._send_changelog(None, 'k', 'v')

    def test_on_changelog_sent(self, *, table: MyTable) -> None:
        fut = Mock(name='future', autospec=asyncio.Future)
        table._data = Mock(name='data', autospec=Store)
        table._on_changelog_sent(fut)
        table._data.set_persisted_offset.assert_called_once_with(
            fut.result().topic_partition, fut.result().offset
        )

    def test_on_changelog_sent__transactions(self, *, table: MyTable) -> None:
        table.app.in_transaction = True
        table.app.tables = Mock(name='tables')
        fut = Mock(name='fut')
        table._on_changelog_sent(fut)
        table.app.tables.persist_offset_on_commit.assert_called_once_with(
            table.data, fut.result().topic_partition, fut.result().offset
        )

    @pytest.mark.asyncio
    async def test_del_old_keys__empty(self, *, table: MyTable) -> None:
        table.window = Mock(name='window')
        await table._del_old_keys()

    @pytest.mark.asyncio
    async def test_del_old_keys(self, *, table: MyTable) -> None:
        on_window_close: AsyncMock = table._on_window_close = AsyncMock(name='on_window_close')
        table.window = Mock(name='window')
        table._data = {
            ('boo', (1.1, 1.4)): 'BOO',
            ('moo', (1.4, 1.6)): 'MOO',
            ('faa', (1.9, 2.0)): 'FAA',
            ('bar', (4.1, 4.2)): 'BAR',
        }
        table._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        table._partition_timestamp_keys = {
            (TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))],
            (TP1, 5.0): [('bar', (4.1, 4.2))],
        }

        def get_stale(limit: float) -> Callable[[float, float], bool]:
            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale

        table.window.stale.side_effect = get_stale(4.0)
        await table._del_old_keys()
        assert table._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        assert table.data == {('bar', (4.1, 4.2)): 'BAR'}
        on_window_close.assert_has_calls(
            [
                call(('boo', (1.1, 1.4)), 'BOO'),
                call.coro(('boo', (1.1, 1.4)), 'BOO'),
                call(('moo', (1.4, 1.6)), 'MOO'),
                call.coro(('moo', (1.4, 1.6)), 'MOO'),
                call(('faa', (1.9, 2.0)), 'FAA'),
                call.coro(('faa', (1.9, 2.0)), 'FAA'),
            ]
        )
        table.last_closed_window = 8.0
        table.window.stale.side_effect = get_stale(6.0)
        await table._del_old_keys()
        assert not table.data

    @pytest.mark.asyncio
    async def test_del_old_keys_non_async_cb(self, *, table: MyTable) -> None:
        on_window_close: Mock = table._on_window_close = Mock(name='on_window_close')
        table.window = Mock(name='window')
        table._data = {
            ('boo', (1.1, 1.4)): 'BOO',
            ('moo', (1.4, 1.6)): 'MOO',
            ('faa', (1.9, 2.0)): 'FAA',
            ('bar', (4.1, 4.2)): 'BAR',
        }
        table._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        table._partition_timestamp_keys = {
            (TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))],
            (TP1, 5.0): [('bar', (4.1, 4.2))],
        }

        def get_stale(limit: float) -> Callable[[float, float], bool]:
            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale

        table.window.stale.side_effect = get_stale(4.0)
        await table._del_old_keys()
        assert table._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        on_window_close.assert_has_calls(
            [call(('boo', (1.1, 1.4)), 'BOO'),
             call(('moo', (1.4, 1.6)), 'MOO'),
             call(('faa', (1.9, 2.0)), 'FAA')]
        )
        table.last_closed_window = 8.0
        table.window.stale.side_effect = get_stale(6.0)
        await table._del_old_keys()
        assert not table.data

    @pytest.mark.asyncio
    async def test_on_window_close__default(self, *, table: MyTable) -> None:
        table._on_window_close = None
        await table.on_window_close(('boo', (1.1, 1.4)), 'BOO')

    @pytest.mark.parametrize('source_n,change_n,expect_error', [(3, 3, False), (3, None, False), (None, 3, False), (3, 6, True), (6, 3, True)])
    def test__verify_source_topic_partitions(self, source_n: Optional[int], change_n: Optional[int], expect_error: bool, *, app: Any, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        tps: Dict[Any, Optional[int]] = {event.message.topic: source_n, table.changelog_topic.get_topic_name(): change_n}  # type: ignore
        app.consumer.topic_partitions = Mock(side_effect=tps.get)
        if expect_error:
            with pytest.raises(PartitionsMismatch):
                table._verify_source_topic_partitions(event.message.topic)
        else:
            table._verify_source_topic_partitions(event.message.topic)

    @pytest.mark.asyncio
    async def test_clean_data(self, *, table: MyTable) -> None:
        table._should_expire_keys = Mock(name='_should_expire_keys')
        table._should_expire_keys.return_value = False
        await table._clean_data(table)  # type: ignore
        table._should_expire_keys.return_value = True
        table._del_old_keys = AsyncMock(name='_del_old_keys')

        async def on_sleep(secs: float, **kwargs: Any) -> None:
            if table.sleep.call_count > 2:
                table._stopped.set()

        table.sleep = AsyncMock(name='sleep', side_effect=on_sleep)
        await table._clean_data(table)  # type: ignore
        table._del_old_keys.assert_called_once_with()
        table.sleep.assert_called_with(pytest.approx(table.app.conf.table_cleanup_interval, rel=0.1))

    def test_should_expire_keys(self, *, table: MyTable) -> None:
        table.window = None
        assert not table._should_expire_keys()
        table.window = Mock(name='window', autospec=Window)
        table.window.expires = 3600
        assert table._should_expire_keys()

    def test_join(self, *, table: MyTable) -> None:
        table._join = Mock(name='join')
        ret = table.join(User.id, User.name)
        table._join.assert_called_once_with(joins.RightJoin(stream=table, fields=(User.id, User.name)))
        assert ret is table._join()

    def test_left_join(self, *, table: MyTable) -> None:
        table._join = Mock(name='join')
        ret = table.left_join(User.id, User.name)
        table._join.assert_called_once_with(joins.LeftJoin(stream=table, fields=(User.id, User.name)))
        assert ret is table._join()

    def test_inner_join(self, *, table: MyTable) -> None:
        table._join = Mock(name='join')
        ret = table.inner_join(User.id, User.name)
        table._join.assert_called_once_with(joins.InnerJoin(stream=table, fields=(User.id, User.name)))
        assert ret is table._join()

    def test_outer_join(self, *, table: MyTable) -> None:
        table._join = Mock(name='join')
        ret = table.outer_join(User.id, User.name)
        table._join.assert_called_once_with(joins.OuterJoin(stream=table, fields=(User.id, User.name)))
        assert ret is table._join()

    def test__join(self, *, table: MyTable) -> None:
        with pytest.raises(NotImplementedError):
            table._join(Mock(name='join_strategy', autospec=joins.Join))

    def test_clone(self, *, table: MyTable) -> None:
        t2 = table.clone()
        assert t2.info() == table.info()

    def test_combine(self, *, table: MyTable) -> None:
        with pytest.raises(NotImplementedError):
            table.combine(Mock(name='joinable', autospec=Stream))

    def test_contribute_to_stream(self, *, table: MyTable) -> None:
        table.contribute_to_stream(Mock(name='stream', autospec=Stream))

    @pytest.mark.asyncio
    async def test_remove_from_stream(self, *, table: MyTable) -> None:
        await table.remove_from_stream(Mock(name='stream', autospec=Stream))

    def test_new_changelog_topic__window_expires(self, *, table: MyTable) -> None:
        table.window = Mock(name='window', autospec=Window)
        table.window.expires = 3600.3
        assert table._new_changelog_topic(retention=None).retention == 3600.3

    def test_new_changelog_topic__default_compacting(self, *, table: MyTable) -> None:
        table._changelog_compacting = True
        assert table._new_changelog_topic(compacting=None).compacting
        table._changelog_compacting = False
        assert not table._new_changelog_topic(compacting=None).compacting
        assert table._new_changelog_topic(compacting=True).compacting

    def test_new_changelog_topic__default_deleting(self, *, table: MyTable) -> None:
        table._changelog_deleting = True
        assert table._new_changelog_topic(deleting=None).deleting
        table._changelog_deleting = False
        assert not table._new_changelog_topic(deleting=None).deleting
        assert table._new_changelog_topic(deleting=True).deleting

    def test_copy(self, *, table: MyTable) -> None:
        assert copy(table).info() == table.info()

    def test_and(self, *, table: MyTable) -> None:
        with pytest.raises(NotImplementedError):
            _ = table & table

    def test__maybe_set_key_ttl(self, *, table: MyTable) -> None:
        table._should_expire_keys = Mock(return_value=False)
        table._maybe_set_key_ttl(('k', (100, 110)), 0)
        table._should_expire_keys = Mock(return_value=True)
        table._maybe_set_key_ttl(('k', (100, 110)), 0)

    def test__maybe_del_key_ttl(self, *, table: MyTable) -> None:
        table._partition_timestamp_keys[0, 110] = None
        table._should_expire_keys = Mock(return_value=False)
        table._maybe_del_key_ttl(('k', (100, 110)), 0)
        table._should_expire_keys = Mock(return_value=True)
        table._maybe_del_key_ttl(('k', (100, 110)), 0)
        table._partition_timestamp_keys[0, 110] = {('k', (100, 110)), ('v', (100, 110))}
        table._maybe_del_key_ttl(('k', (100, 110)), 0)
        assert table._partition_timestamp_keys[0, 110] == {('v', (100, 110))}

    def test_apply_window_op(self, *, table: MyTable) -> None:
        self.mock_ranges(table)
        table._set_key(('k', 1.1), 30)
        table._set_key(('k', 1.2), 40)
        table._set_key(('k', 1.3), 50)
        table._apply_window_op(operator.add, 'k', 12, 300.3)
        assert table._get_key(('k', 1.1)) == 42
        assert table._get_key(('k', 1.2)) == 52
        assert table._get_key(('k', 1.3)) == 62

    def test_set_del_windowed(self, *, table: MyTable) -> None:
        ranges = self.mock_ranges(table)
        table._set_windowed('k', 11, 300.3)
        for r in ranges:
            assert table._get_key(('k', r)) == 11
        table._del_windowed('k', 300.3)
        for r in ranges:
            assert table._get_key(('k', r)) is None

    def test_window_ranges(self, *, table: MyTable) -> None:
        table.window = Mock(name='window', autospec=Window)
        table.window.ranges.return_value = [1, 2, 3]
        assert list(table._window_ranges(300.3)) == [1, 2, 3]

    def mock_ranges(self, table: MyTable, ranges: List[float] = [1.1, 1.2, 1.3]) -> List[float]:
        table._window_ranges = Mock(name='_window_ranges')
        table._window_ranges.return_value = ranges
        return ranges

    def test_relative_now(self, *, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        table._partition_latest_timestamp[event.message.partition] = 30.3
        assert table._relative_now(event) == 30.3

    def test_relative_now__no_event(self, *, table: MyTable) -> None:
        with patch('faust.tables.base.current_event') as ce:
            ce.return_value = None
            with patch('time.time') as time_fn:
                assert table._relative_now(None) is time_fn()

    def test_relative_event(self, *, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        assert table._relative_event(event) is event.message.timestamp

    def test_relative_event__raises_if_no_event(self, *, table: MyTable) -> None:
        with patch('faust.tables.base.current_event') as current_event:
            current_event.return_value = None
            with pytest.raises(RuntimeError):
                table._relative_event(None)

    def test_relative_field(self, *, table: MyTable) -> None:
        user = User('foo', 'bar')
        event = Mock(name='event', autospec=Event)
        event.value = user
        assert table._relative_field(User.id)(event) == 'foo'

    def test_relative_field__raises_if_no_event(self, *, table: MyTable) -> None:
        with pytest.raises(RuntimeError):
            table._relative_field(User.id)(event=None)

    def test_relative_timestamp(self, *, table: MyTable) -> None:
        assert table._relative_timestamp(303.3)(Mock(name='event', autospec=Event)) == 303.3

    def test_windowed_now(self, *, table: MyTable) -> None:
        with patch('faust.tables.base.current_event'):
            table.window = Mock(name='window', autospec=Window)
            table.window.earliest.return_value = 42
            table._get_key = Mock(name='_get_key')
            table._windowed_now('k')
            table._get_key.assert_called_once_with(('k', 42))

    def test_windowed_timestamp(self, *, table: MyTable) -> None:
        table.window = Mock(name='window', autospec=Window)
        table.window.current.return_value = 10.1
        assert not table._windowed_contains('k', 303.3)
        table._set_key(('k', 10.1), 101.1)
        assert table._windowed_timestamp('k', 303.3) == 101.1
        assert table._windowed_contains('k', 303.3)

    def test_windowed_delta(self, *, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        table.window = Mock(name='window', autospec=Window)
        table.window.delta.return_value = 10.1
        table._set_key(('k', 10.1), 101.1)
        assert table._windowed_delta('k', 303.3, event=event) == 101.1

    @pytest.mark.asyncio
    async def test_on_rebalance(self, *, table: MyTable) -> None:
        table._data = Mock(name='data', autospec=Store, on_rebalance=AsyncMock())
        await table.on_rebalance({TP1}, set(), set())
        table._data.on_rebalance.assert_called_once_with(table, {TP1}, set(), set())

    @pytest.mark.asyncio
    async def test_on_changelog_event(self, *, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        table._on_changelog_event = None
        await table.on_changelog_event(event)
        table._on_changelog_event = AsyncMock(name='callback')
        await table.on_changelog_event(event)
        table._on_changelog_event.assert_called_once_with(event)

    def test_label(self, *, table: MyTable) -> None:
        assert label(table)

    def test_shortlabel(self, *, table: MyTable) -> None:
        assert shortlabel(table)

    def test_apply_changelog_batch(self, *, table: MyTable) -> None:
        table._data = Mock(name='data', autospec=Store)
        table.apply_changelog_batch([1, 2, 3])
        table._data.apply_changelog_batch.assert_called_once_with(
            [1, 2, 3], to_key=table._to_key, to_value=table._to_value
        )

    def test_to_key(self, *, table: MyTable) -> None:
        assert table._to_key([1, 2, 3]) == (1, 2, 3)
        assert table._to_key(1) == 1

    def test_to_value(self, *, table: MyTable) -> None:
        v = Mock(name='v')
        assert table._to_value(v) is v

    def test__human_channel(self, *, table: MyTable) -> None:
        assert table._human_channel()

    def test_repr_info(self, *, table: MyTable) -> None:
        assert table._repr_info() == table.name

    def test_partition_for_key__partitioner(self, *, table: MyTable, app: Any) -> None:
        table.use_partitioner = True
        partition: Optional[int] = None
        assert table.partition_for_key('k') is partition
