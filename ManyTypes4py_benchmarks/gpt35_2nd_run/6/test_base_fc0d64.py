from typing import Any, Dict, List, Set, Tuple, Union
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
    id: str
    name: str

class MyTable(Collection):
    datas: Dict[str, Any]

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.datas = {}

    def _has_key(self, key: str) -> bool:
        return key in self.datas

    def _get_key(self, key: str) -> Any:
        return self.datas.get(key)

    def _set_key(self, key: str, value: Any) -> None:
        self.datas[key] = value

    def _del_key(self, key: str) -> None:
        self.datas.pop(key, None)

    def hopping(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def tumbling(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def using_window(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def as_ansitable(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def info(self) -> Dict[str, Any]:
        return {'app': self.app, 'name': self.name, 'store': self._store, 'default': self.default, 'schema': self.schema, 'key_type': self.key_type, 'value_type': self.value_type, 'changelog_topic': self._changelog_topic, 'window': self.window, 'extra_topic_configs': self.extra_topic_configs, 'on_changelog_event': self._on_changelog_event, 'recover_callbacks': self._recover_callbacks, 'partitions': self.partitions, 'recovery_buffer_size': self.recovery_buffer_size, 'standby_buffer_size': self.standby_buffer_size, 'use_partitioner': self.use_partitioner}

    def persisted_offset(self, tp: TP) -> Any:
        data = self._data
        return data.persisted_offset()

    async def need_active_standby_for(self, tp: TP) -> Any:
        return await self._data.need_active_standby_for.coro()

    def reset_state(self) -> None:
        data = self._data
        data.reset_state()

    def send_changelog(self, event: Event, key: str, value: Any) -> None:
        self.changelog_topic.send_soon(key='k', value='v', partition=event.message.partition, key_serializer='json', value_serializer='json', callback=self._on_changelog_sent, eager_partitioning=True)

    def send_changelog__custom_serializers(self, event: Event, key: str, value: Any) -> None:
        self.changelog_topic.send_soon(key='k', value='v', partition=event.message.partition, key_serializer='raw', value_serializer='raw', callback=self._on_changelog_sent, eager_partitioning=True)

    def on_changelog_sent(self, fut: asyncio.Future) -> None:
        self._data.set_persisted_offset(fut.result().topic_partition, fut.result().offset)

    def on_changelog_sent__transactions(self, fut: Any) -> None:
        self.app.tables.persist_offset_on_commit(self.data, fut.result().topic_partition, fut.result().offset)

    async def del_old_keys__empty(self) -> None:
        self.window = Mock(name='window')
        await self._del_old_keys()

    async def del_old_keys(self) -> None:
        on_window_close = self._on_window_close
        self.window = Mock(name='window')
        self._data = {('boo', (1.1, 1.4)): 'BOO', ('moo', (1.4, 1.6)): 'MOO', ('faa', (1.9, 2.0)): 'FAA', ('bar', (4.1, 4.2)): 'BAR'}
        self._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        self._partition_timestamp_keys = {(TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))], (TP1, 5.0): [('bar', (4.1, 4.2))]}

        def get_stale(limit: float) -> Callable[[float, float], bool]:

            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale
        self.window.stale.side_effect = get_stale(4.0)
        await self._del_old_keys()
        assert self._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        assert self.data == {('bar', (4.1, 4.2)): 'BAR'}
        on_window_close.assert_has_calls([call(('boo', (1.1, 1.4)), 'BOO'), call.coro(('boo', (1.1, 1.4)), 'BOO'), call(('moo', (1.4, 1.6)), 'MOO'), call.coro(('moo', (1.4, 1.6)), 'MOO'), call(('faa', (1.9, 2.0)), 'FAA'), call.coro(('faa', (1.9, 2.0)), 'FAA')])
        self.last_closed_window = 8.0
        self.window.stale.side_effect = get_stale(6.0)
        await self._del_old_keys()
        assert not self.data

    async def del_old_keys_non_async_cb(self) -> None:
        on_window_close = self._on_window_close
        self.window = Mock(name='window')
        self._data = {('boo', (1.1, 1.4)): 'BOO', ('moo', (1.4, 1.6)): 'MOO', ('faa', (1.9, 2.0)): 'FAA', ('bar', (4.1, 4.2)): 'BAR'}
        self._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        self._partition_timestamp_keys = {(TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))], (TP1, 5.0): [('bar', (4.1, 4.2))]}

        def get_stale(limit: float) -> Callable[[float, float], bool]:

            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale
        self.window.stale.side_effect = get_stale(4.0)
        await self._del_old_keys()
        assert self._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        assert self.data == {('bar', (4.1, 4.2)): 'BAR'}
        on_window_close.assert_has_calls([call(('boo', (1.1, 1.4)), 'BOO'), call(('moo', (1.4, 1.6)), 'MOO'), call(('faa', (1.9, 2.0)), 'FAA')]
        self.last_closed_window = 8.0
        self.window.stale.side_effect = get_stale(6.0)
        await self._del_old_keys()
        assert not self.data

    def _verify_source_topic_partitions(self, source_n: int, change_n: int, expect_error: bool) -> None:
        event = Mock(name='event', autospec=Event)
        tps = {event.message.topic: source_n, self.changelog_topic.get_topic_name(): change_n}
        self.app.consumer.topic_partitions = Mock(side_effect=tps.get)
        if expect_error:
            with pytest.raises(PartitionsMismatch):
                self._verify_source_topic_partitions(event.message.topic)
        else:
            self._verify_source_topic_partitions(event.message.topic)

    async def clean_data(self) -> None:
        self._should_expire_keys = Mock(name='_should_expire_keys')
        self._should_expire_keys.return_value = False
        await self._clean_data()
        self._should_expire_keys.return_value = True
        self._del_old_keys = AsyncMock(name='_del_old_keys')

        def on_sleep(secs: float, **kwargs: Any) -> None:
            if self.sleep.call_count > 2:
                self._stopped.set()
        self.sleep = AsyncMock(name='sleep', side_effect=on_sleep)
        await self._clean_data()
        self._del_old_keys.assert_called_once_with()
        self.sleep.assert_called_with(pytest.approx(self.app.conf.table_cleanup_interval, rel=0.1)

    def should_expire_keys(self) -> bool:
        self.window = None
        return not self._should_expire_keys()

    def join(self, *fields: Any) -> Any:
        self._join = Mock(name='join')
        ret = self.join(fields)
        self._join.assert_called_once_with(joins.RightJoin(stream=self, fields=fields))
        return ret

    def left_join(self, *fields: Any) -> Any:
        self._join = Mock(name='join')
        ret = self.left_join(fields)
        self._join.assert_called_once_with(joins.LeftJoin(stream=self, fields=fields))
        return ret

    def inner_join(self, *fields: Any) -> Any:
        self._join = Mock(name='join')
        ret = self.inner_join(fields)
        self._join.assert_called_once_with(joins.InnerJoin(stream=self, fields=fields))
        return ret

    def outer_join(self, *fields: Any) -> Any:
        self._join = Mock(name='join')
        ret = self.outer_join(fields)
        self._join.assert_called_once_with(joins.OuterJoin(stream=self, fields=fields))
        return ret

    def _join(self, join_strategy: joins.Join) -> None:
        raise NotImplementedError()

    def clone(self) -> Any:
        t2 = self.clone()
        assert t2.info() == self.info()

    def combine(self, joinable: Stream) -> None:
        raise NotImplementedError()

    def contribute_to_stream(self, stream: Stream) -> None:
        self.contribute_to_stream(stream)

    async def remove_from_stream(self, stream: Stream) -> None:
        await self.remove_from_stream(stream)

    def new_changelog_topic(self, retention: float = None, compacting: bool = None, deleting: bool = None) -> Topic:
        if self.window.expires:
            return self._new_changelog_topic(retention=self.window.expires)
        if compacting is None:
            compacting = self._changelog_compacting
        if deleting is None:
            deleting = self._changelog_deleting
        return self._new_changelog_topic(compacting=compacting, deleting=deleting)

    def copy(self) -> Any:
        return copy(self)

    def __and__(self, other: Any) -> None:
        raise NotImplementedError()

    def _maybe_set_key_ttl(self, key: Tuple[str, Tuple[float, float]], timestamp: float) -> None:
        if not self._should_expire_keys():
            self._maybe_set_key_ttl(key, timestamp)

    def _maybe_del_key_ttl(self, key: Tuple[str, Tuple[float, float]], timestamp: float) -> None:
        if self._partition_timestamp_keys[0, 110] is None:
            if not self._should_expire_keys():
                self._maybe_del_key_ttl(key, timestamp)
            self._partition_timestamp_keys[0, 110] = {('k', (100, 110)), ('v', (100, 110))}
            self._maybe_del_key_ttl(key, timestamp)
            assert self._partition_timestamp_keys[0, 110] == {('v', (100, 110))}

    def apply_window_op(self, op: Callable, key: str, value: Any, timestamp: float) -> None:
        self.mock_ranges()
        self._set_key((key, 1.1), 30)
        self._set_key((key, 1.2), 40)
        self._set_key((key, 1.3), 50)
        self._apply_window_op(operator.add, key, 12, 300.3)
        assert self._get_key((key, 1.1)) == 42
        assert self._get_key((key, 1.2)) == 52
        assert self._get_key((key, 1.3)) == 62

    def set_del_windowed(self, key: str, value: Any, timestamp: float) -> None:
        ranges = self.mock_ranges()
        self._set_windowed(key, 11, 300.3)
        for r in ranges:
            assert self._get_key((key, r)) == 11
        self._del_windowed(key, 300.3)
        for r in ranges:
            assert self._get_key((key, r)) is None

    def window_ranges(self, timestamp: float) -> List[float]:
        self.window = Mock(name='window', autospec=Window)
        self.window.ranges.return_value = [1, 2, 3]
        return list(self._window_ranges(300.3))

    def mock_ranges(self, ranges: List[float] = [1.1, 1.2, 1.3]) -> List[float]:
        self._window_ranges = Mock(name='_window_ranges')
        self._window_ranges.return_value = ranges
        return ranges

    def relative_now(self, event: Event) -> float:
        if event:
            return self._partition_latest_timestamp[event.message.partition]
        return time()

    def relative_event(self, event: Event) -> float:
        if not event:
            raise RuntimeError
        return event.message.timestamp

    def relative_field(self, field: Any) -> Any:
        return field

    def relative_timestamp(self, timestamp: float) -> float:
        return timestamp

    def windowed_now(self, key: str) -> Any:
        self.window = Mock(name='window', autospec=Window)
        self.window.earliest.return_value = 42
        self._get_key = Mock(name='_get_key')
        self._windowed_now(key)
        self._get_key.assert_called_once_with((key, 42))

    def windowed_timestamp(self, key: str, timestamp: float) -> Any:
        if not self._windowed_contains(key, timestamp):
            self._set_key((key, 10.1), 101.1)
        return self._get_key((key, 10.1))

    def windowed_delta(self, key: str, timestamp: float, event: Event) -> Any:
        self.window = Mock(name='window', autospec=Window)
        self.window.delta.return_value = 10.1
        self._set_key((key, 10.1), 101.1)
        return self._get_key((key, 10.1))

    async def on_rebalance(self, tp: TP, assigned: Set[TP], revoked: Set[TP]) -> None:
        await self._data.on_rebalance(self, tp, assigned, revoked)

    async def on_changelog_event(self, event: Event) -> None:
        if self._on_changelog_event:
            await self._on_changelog_event(event)

    def apply_changelog_batch(self, batch: List[Any]) -> None:
        self._data.apply_changelog_batch(batch, to_key=self._to_key, to_value=self._to_value)

    def to_key(self, key: Any) -> Any:
        return key

    def to_value(self, value: Any) -> Any:
        return value

    def _human_channel(self) -> Any:
        return self.name

    def _repr_info(self) -> str:
        return self.name

    def partition_for_key(self, key: str) -> Any:
        if self.use_partitioner:
            return None
