#!/usr/bin/env python3
import datetime
from typing import Any, Optional

import faust
import pytest
from faust.events import Event
from faust.types import Message
from faust.tables.wrappers import WindowSet, WindowWrapper
from mode.utils.mocks import Mock, patch


class TableKey(faust.Record):
    pass


class TableValue(faust.Record):
    pass


KEY1: TableKey = TableKey('foo')
VALUE1: TableValue = TableValue('id', 3)
WINDOW1: Any = faust.HoppingWindow(size=10, step=2, expires=3600.0)


def event() -> Event:
    message: Message = Message(
        topic='test-topic',
        key='key',
        value='value',
        partition=3,
        offset=0,
        checksum=None,
        timestamp=datetime.datetime.now().timestamp(),
        timestamp_type=0,
        headers={}
    )
    return Event(app='test-app', key='key', value='value', headers={}, message=message)


class test_Table:

    @pytest.fixture
    def table(self, *, app: faust.App) -> Any:
        return self.create_table(app, name='foo', default=int)

    @pytest.fixture
    def strict_table(self, *, app: faust.App) -> Any:
        return self.create_table(app, name='strict')

    def create_table(self, app: faust.App, *, name: str = 'foo', key_type: type = TableKey,
                     value_type: type = TableValue, **kwargs: Any) -> Any:
        return app.Table(name, key_type=key_type, value_type=value_type, **kwargs)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_using_window(self, patch_current: Any, *, table: Any) -> None:
        with_wrapper: Any = table.using_window(WINDOW1)
        self.assert_wrapper(with_wrapper, table, WINDOW1)
        self.assert_current(with_wrapper, patch_current)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_hopping(self, patch_current: Any, *, table: Any) -> None:
        with_wrapper: Any = table.hopping(10, 2, 3600)
        self.assert_wrapper(with_wrapper, table)
        self.assert_current(with_wrapper, patch_current)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_tumbling(self, patch_current: Any, *, table: Any) -> None:
        with_wrapper: Any = table.tumbling(10, 3600)
        self.assert_wrapper(with_wrapper, table)
        self.assert_current(with_wrapper, patch_current)

    def assert_wrapper(self, wrapper: Any, table: Any, window: Optional[Any] = None) -> None:
        assert wrapper.table is table
        t: Any = wrapper.table
        if window is not None:
            assert t.window is window
        assert t._changelog_compacting
        assert t._changelog_deleting
        assert t._changelog_topic is None
        assert isinstance(wrapper, WindowWrapper)

    def assert_current(self, wrapper: Any, patch_current: Any) -> None:
        value: WindowSet = wrapper['test']
        assert isinstance(value, WindowSet)
        patch_current.asssert_called_once_with()
        assert value.current() == 0

    def test_missing__when_default(self, *, table: Any) -> None:
        assert table['foo'] == 0
        table.data['foo'] = 3
        assert table['foo'] == 3

    def test_missing__no_default(self, *, strict_table: Any) -> None:
        with pytest.raises(KeyError):
            strict_table['foo']
        strict_table.data['foo'] = 3
        assert strict_table['foo'] == 3

    def test_has_key(self, *, table: Any) -> None:
        assert not table._has_key('foo')
        table.data['foo'] = 3
        assert table._has_key('foo')

    def test_get_key(self, *, table: Any) -> None:
        assert table._get_key('foo') == 0
        table.data['foo'] = 3
        assert table._get_key('foo') == 3

    def test_set_key(self, *, table: Any) -> None:
        with patch('faust.tables.base.current_event') as current_event:
            event_obj: Any = current_event.return_value
            partition: Any = event_obj.message.partition
            table.send_changelog = Mock(name='send_changelog')
            table._set_key('foo', 'val')
            table.send_changelog.asssert_called_once_with(partition, 'foo', 'val')
            assert table['foo'] == 'val'

    def test_del_key(self, *, table: Any) -> None:
        with patch('faust.tables.base.current_event') as current_event:
            event_obj: Any = current_event.return_value
            partition: Any = event_obj.message.partition
            table.send_changelog = Mock(name='send_changelog')
            table.data['foo'] = 3
            table._del_key('foo')
            table.send_changelog.asssert_called_once_with(partition, 'foo', None)
            assert 'foo' not in table.data

    def test_as_ansitable(self, *, table: Any) -> None:
        table.data['foo'] = 'bar'
        table.data['bar'] = 'baz'
        assert table.as_ansitable(sort=True)
        assert table.as_ansitable(sort=False)

    def test_on_key_set__no_event(self, *, table: Any) -> None:
        with patch('faust.tables.base.current_event') as ce:
            ce.return_value = None
            with pytest.raises(TypeError):
                table.on_key_set('k', 'v')

    def test_on_key_del__no_event(self, *, table: Any) -> None:
        with patch('faust.tables.base.current_event') as ce:
            ce.return_value = None
            with pytest.raises(TypeError):
                table.on_key_del('k')