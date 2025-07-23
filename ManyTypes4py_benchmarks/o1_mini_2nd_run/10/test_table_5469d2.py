import datetime
from typing import Any, Optional, Type

import faust
import pytest
from faust.events import Event
from faust.types import AppT, Message
from faust.tables import Table
from faust.windows import HoppingWindow, Window
from faust.tables.wrappers import WindowSet, WindowWrapper
from mode.utils.mocks import Mock, patch


class TableKey(faust.Record):
    pass


class TableValue(faust.Record):
    id: str
    value: int


KEY1: TableKey = TableKey(foo='foo')
VALUE1: TableValue = TableValue(id='id', value=3)
WINDOW1: HoppingWindow = faust.HoppingWindow(size=10, step=2, expires=3600.0)


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
    return Event(
        app='test-app',
        key='key',
        value='value',
        headers={},
        message=message
    )


class test_Table:

    @pytest.fixture
    def table(self, *, app: AppT) -> Table:
        return self.create_table(app, name='foo', default=int)

    @pytest.fixture
    def strict_table(self, *, app: AppT) -> Table:
        return self.create_table(app, name='strict')

    def create_table(
        self,
        app: AppT,
        *,
        name: str = 'foo',
        key_type: Type[TableKey] = TableKey,
        value_type: Type[TableValue] = TableValue,
        **kwargs: Any
    ) -> Table:
        return app.Table(name, key_type=key_type, value_type=value_type, **kwargs)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_using_window(
        self,
        patch_current: Mock,
        *,
        table: Table
    ) -> None:
        with_wrapper: WindowWrapper = table.using_window(WINDOW1)
        self.assert_wrapper(with_wrapper, table, WINDOW1)
        self.assert_current(with_wrapper, patch_current)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_hopping(
        self,
        patch_current: Mock,
        *,
        table: Table
    ) -> None:
        with_wrapper: WindowWrapper = table.hopping(10, 2, 3600)
        self.assert_wrapper(with_wrapper, table)
        self.assert_current(with_wrapper, patch_current)

    @patch('faust.tables.wrappers.current_event', return_value=event())
    def test_tumbling(
        self,
        patch_current: Mock,
        *,
        table: Table
    ) -> None:
        with_wrapper: WindowWrapper = table.tumbling(10, 3600)
        self.assert_wrapper(with_wrapper, table)
        self.assert_current(with_wrapper, patch_current)

    def assert_wrapper(
        self,
        wrapper: WindowWrapper,
        table: Table,
        window: Optional[Window] = None
    ) -> None:
        assert wrapper.table is table
        t: Table = wrapper.table
        if window is not None:
            assert t.window is window
        assert t._changelog_compacting
        assert t._changelog_deleting
        assert t._changelog_topic is None
        assert isinstance(wrapper, WindowWrapper)

    def assert_current(
        self,
        wrapper: WindowWrapper,
        patch_current: Mock
    ) -> None:
        value: WindowSet = wrapper['test']
        assert isinstance(value, WindowSet)
        patch_current.asssert_called_once_with()
        assert value.current() == 0

    def test_missing__when_default(self, *, table: Table) -> None:
        assert table['foo'] == 0
        table.data['foo'] = 3
        assert table['foo'] == 3

    def test_missing__no_default(self, *, strict_table: Table) -> None:
        with pytest.raises(KeyError):
            strict_table['foo']
        strict_table.data['foo'] = 3
        assert strict_table['foo'] == 3

    def test_has_key(self, *, table: Table) -> None:
        assert not table._has_key('foo')
        table.data['foo'] = 3
        assert table._has_key('foo')

    def test_get_key(self, *, table: Table) -> None:
        assert table._get_key('foo') == 0
        table.data['foo'] = 3
        assert table._get_key('foo') == 3

    def test_set_key(self, *, table: Table) -> None:
        with patch('faust.tables.base.current_event') as current_event:
            event_instance: Optional[Event] = current_event.return_value
            partition: int = event_instance.message.partition if event_instance else 0
            table.send_changelog = Mock(name='send_changelog')
            table._set_key('foo', 'val')
            table.send_changelog.asssert_called_once_with(partition, 'foo', 'val')
            assert table['foo'] == 'val'

    def test_del_key(self, *, table: Table) -> None:
        with patch('faust.tables.base.current_event') as current_event:
            event_instance: Optional[Event] = current_event.return_value
            partition: int = event_instance.message.partition if event_instance else 0
            table.send_changelog = Mock(name='send_changelog')
            table.data['foo'] = 3
            table._del_key('foo')
            table.send_changelog.asssert_called_once_with(partition, 'foo', None)
            assert 'foo' not in table.data

    def test_as_ansitable(self, *, table: Table) -> None:
        table.data['foo'] = 'bar'
        table.data['bar'] = 'baz'
        assert table.as_ansitable(sort=True)
        assert table.as_ansitable(sort=False)

    def test_on_key_set__no_event(self, *, table: Table) -> None:
        with patch('faust.tables.base.current_event') as ce:
            ce.return_value = None
            with pytest.raises(TypeError):
                table.on_key_set('k', 'v')

    def test_on_key_del__no_event(self, *, table: Table) -> None:
        with patch('faust.tables.base.current_event') as ce:
            ce.return_value = None
            with pytest.raises(TypeError):
                table.on_key_del('k')
