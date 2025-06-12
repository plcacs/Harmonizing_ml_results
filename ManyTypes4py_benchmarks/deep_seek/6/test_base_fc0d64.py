import asyncio
import operator
from copy import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
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

TP1 = TP('foo', 0)

class User(Record):
    pass

class MyTable(Collection):
    datas: Dict[Any, Any]
    
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.datas = {}

    def _has_key(self, key: Any) -> bool:
        return key in self.datas

    def _get_key(self, key: Any) -> Any:
        return self.datas.get(key)

    def _set_key(self, key: Any, value: Any) -> None:
        self.datas[key] = value

    def _del_key(self, key: Any) -> None:
        self.datas.pop(key, None)

    def hopping(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def tumbling(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def using_window(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def as_ansitable(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

class test_Collection:
    @pytest.fixture
    def table(self, *, app: Any) -> MyTable:
        return MyTable(app, name='name')

    def test_key_type_bytes_implies_raw_serializer(self, *, app: Any) -> None:
        table = MyTable(app, name='name', key_type=bytes)
        assert table.key_serializer == 'raw'

    @pytest.mark.asyncio
    async def test_init_on_recover(self, *, app: Any) -> None:
        on_recover = AsyncMock(name='on_recover')
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
        assert table.info() == {'app': table.app, 'name': table.name, 'store': table._store, 'default': table.default, 'schema': table.schema, 'key_type': table.key_type, 'value_type': table.value_type, 'changelog_topic': table._changelog_topic, 'window': table.window, 'extra_topic_configs': table.extra_topic_configs, 'on_changelog_event': table._on_changelog_event, 'recover_callbacks': table._recover_callbacks, 'partitions': table.partitions, 'recovery_buffer_size': table.recovery_buffer_size, 'standby_buffer_size': table.standby_buffer_size, 'use_partitioner': table.use_partitioner}

    def test_persisted_offset(self, *, table: MyTable) -> None:
        data = table._data = Mock(name='_data')
        assert table.persisted_offset(TP1) == data.persisted_offset()

    @pytest.mark.asyncio
    async def test_need_active_standby_for(self, *, table: MyTable) -> None:
        table._data = Mock(name='_data', autospec=Store, need_active_standby_for=AsyncMock())
        assert await table.need_active_standby_for(TP1) == table._data.need_active_standby_for.coro()

    def test_reset_state(self, *, table: MyTable) -> None:
        data = table._data = Mock(name='_data', autospec=Store)
        table.reset_state()
        data.reset_state.assert_called_once_with()

    def test_send_changelog(self, *, table: MyTable) -> None:
        table.changelog_topic.send_soon = Mock(name='send_soon')
        event = Mock(name='event')
        table._send_changelog(event, 'k', 'v')
        table.changelog_topic.send_soon.assert_called_once_with(key='k', value='v', partition=event.message.partition, key_serializer='json', value_serializer='json', callback=table._on_changelog_sent, eager_partitioning=True)

    def test_send_changelog__custom_serializers(self, *, table: MyTable) -> None:
        event = Mock(name='event')
        table.changelog_topic.send_soon = Mock(name='send_soon')
        table._send_changelog(event, 'k', 'v', key_serializer='raw', value_serializer='raw')
        table.changelog_topic.send_soon.assert_called_once_with(key='k', value='v', partition=event.message.partition, key_serializer='raw', value_serializer='raw', callback=table._on_changelog_sent, eager_partitioning=True)

    def test_send_changelog__no_current_event(self, *, table: MyTable) -> None:
        with pytest.raises(RuntimeError):
            table._send_changelog(None, 'k', 'v')

    def test_on_changelog_sent(self, *, table: MyTable) -> None:
        fut = Mock(name='future', autospec=asyncio.Future)
        table._data = Mock(name='data', autospec=Store)
        table._on_changelog_sent(fut)
        table._data.set_persisted_offset.assert_called_once_with(fut.result().topic_partition, fut.result().offset)

    def test_on_changelog_sent__transactions(self, *, table: MyTable) -> None:
        table.app.in_transaction = True
        table.app.tables = Mock(name='tables')
        fut = Mock(name='fut')
        table._on_changelog_sent(fut)
        table.app.tables.persist_offset_on_commit.assert_called_once_with(table.data, fut.result().topic_partition, fut.result().offset)

    @pytest.mark.asyncio
    async def test_del_old_keys__empty(self, *, table: MyTable) -> None:
        table.window = Mock(name='window')
        await table._del_old_keys()

    @pytest.mark.asyncio
    async def test_del_old_keys(self, *, table: MyTable) -> None:
        on_window_close = table._on_window_close = AsyncMock(name='on_window_close')
        table.window = Mock(name='window')
        table._data = {('boo', (1.1, 1.4)): 'BOO', ('moo', (1.4, 1.6)): 'MOO', ('faa', (1.9, 2.0)): 'FAA', ('bar', (4.1, 4.2)): 'BAR'}
        table._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        table._partition_timestamp_keys = {(TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))], (TP1, 5.0): [('bar', (4.1, 4.2))]}

        def get_stale(limit: float) -> Any:
            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale
        table.window.stale.side_effect = get_stale(4.0)
        await table._del_old_keys()
        assert table._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        assert table.data == {('bar', (4.1, 4.2)): 'BAR'}
        on_window_close.assert_has_calls([call(('boo', (1.1, 1.4)), 'BOO'), call.coro(('boo', (1.1, 1.4)), 'BOO'), call(('moo', (1.4, 1.6)), 'MOO'), call.coro(('moo', (1.4, 1.6)), 'MOO'), call(('faa', (1.9, 2.0)), 'FAA'), call.coro(('faa', (1.9, 2.0)), 'FAA')])
        table.last_closed_window = 8.0
        table.window.stale.side_effect = get_stale(6.0)
        await table._del_old_keys()
        assert not table.data

    @pytest.mark.asyncio
    async def test_del_old_keys_non_async_cb(self, *, table: MyTable) -> None:
        on_window_close = table._on_window_close = Mock(name='on_window_close')
        table.window = Mock(name='window')
        table._data = {('boo', (1.1, 1.4)): 'BOO', ('moo', (1.4, 1.6)): 'MOO', ('faa', (1.9, 2.0)): 'FAA', ('bar', (4.1, 4.2)): 'BAR'}
        table._partition_timestamps = {TP1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        table._partition_timestamp_keys = {(TP1, 2.0): [('boo', (1.1, 1.4)), ('moo', (1.4, 1.6)), ('faa', (1.9, 2.0))], (TP1, 5.0): [('bar', (4.1, 4.2))]}

        def get_stale(limit: float) -> Any:
            def is_stale(timestamp: float, latest_timestamp: float) -> bool:
                return timestamp < limit
            return is_stale
        table.window.stale.side_effect = get_stale(4.0)
        await table._del_old_keys()
        assert table._partition_timestamps[TP1] == [4.0, 5.0, 6.0, 7.0]
        assert table.data == {('bar', (4.1, 4.2)): 'BAR'}
        on_window_close.assert_has_calls([call(('boo', (1.1, 1.4)), 'BOO'), call(('moo', (1.4, 1.6)), 'MOO'), call(('faa', (1.9, 2.0)), 'FAA')])
        table.last_closed_window = 8.0
        table.window.stale.side_effect = get_stale(6.0)
        await table._del_old_keys()
        assert not table.data

    @pytest.mark.asyncio
    async def test_on_window_close__default(self, *, table: MyTable) -> None:
        assert table._on_window_close is None
        await table.on_window_close(('boo', (1.1, 1.4)), 'BOO')

    @pytest.mark.parametrize('source_n,change_n,expect_error', [(3, 3, False), (3, None, False), (None, 3, False), (3, 6, True), (6, 3, True)])
    def test__verify_source_topic_partitions(self, source_n: Optional[int], change_n: Optional[int], expect_error: bool, *, app: Any, table: MyTable) -> None:
        event = Mock(name='event', autospec=Event)
        tps = {event.message.topic: source_n, table.changelog_topic.get_topic_name(): change_n}
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
        await table._clean_data(table)
        table._should_expire_keys.return_value = True
        table._del_old_keys = AsyncMock(name='_del_old_keys')

        def on_sleep(secs: float, **kwargs: Any) -> None:
            if table.sleep.call_count > 2:
                table._stopped.set()
        table.sleep = AsyncMock(name='sleep', side_effect=on_sleep)
        await table._clean_data(table)
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

