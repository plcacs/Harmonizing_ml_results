from typing import Any, Dict, List, Optional, Union
import operator
import random
from datetime import datetime
import faust
import pytest
from faust.events import Event
from faust.exceptions import ImproperlyConfigured
from faust.tables.wrappers import WindowSet
from faust.types import Message
from mode.utils.mocks import Mock, patch

DATETIME: datetime = datetime.utcnow()
DATETIME_TS: float = DATETIME.timestamp()

class User(faust.Record):
    pass

def same_items(a: Any, b: Any) -> bool:
    a_list = _maybe_items(a)
    b_list = _maybe_items(b)
    return same(a_list, b_list)

def _maybe_items(d: Any) -> Any:
    try:
        items = d.items
    except AttributeError:
        return d
    else:
        return items()

def same(a: List, b: List) -> bool:
    return sorted(a) == sorted(b)

@pytest.yield_fixture()
def current_event(*, freeze_time: Any) -> Any:
    with patch('faust.tables.wrappers.current_event') as current_event:
        with patch('faust.tables.base.current_event', current_event):
            current_event.return_value.message.timestamp = freeze_time.time
            yield current_event

class test_WindowSet:

    @pytest.fixture
    def wset(self, *, wtable: Any, event: Any) -> Any:
        return WindowSet('k', wtable.table, wtable, event)

    def test_constructor(self, *, event: Any, table: Any, wset: Any, wtable: Any) -> None:
        assert wset.key == 'k'
        assert wset.table is table
        assert wset.wrapper is wtable
        assert wset.event is event

    def test_apply(self, *, wset: Any, event: Any) -> Any:
        Mock(name='event2', autospec=Event)
        wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        ret = wset.apply(operator.add, 'val')
        wset.wrapper.get_timestamp.assert_called_once_with(wset.event)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', wset.wrapper.get_timestamp())
        assert ret is wset

    def mock_get_timestamp(self, wset: Any) -> Any:
        m = wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        return m

    def test_apply__custom_event(self, *, wset: Any, event: Any) -> Any:
        event2 = Mock(name='event2', autospec=Event)
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        get_timestamp = self.mock_get_timestamp(wset)
        ret = wset.apply(operator.add, 'val', event2)
        get_timestamp.assert_called_once_with(event2)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', get_timestamp())
        assert ret is wset

    def test_value(self, *, event: Any, wset: Any) -> Any:
        get_timestamp = self.mock_get_timestamp(wset)
        wset.table._windowed_timestamp = Mock(name='_windowed_timestamp')
        assert wset.value(event)
        wset.table._windowed_timestamp.assert_called_once_with('k', get_timestamp())

    def test_now(self, *, wset: Any) -> Any:
        wset.table._windowed_now = Mock(name='_windowed_now')
        ret = wset.now()
        wset.table._windowed_now.assert_called_once_with('k')
        assert ret is wset.table._windowed_now()

    def test_current(self, *, table: Any, wset: Any) -> Any:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current(event2)
        table._relative_event.assert_called_once_with(event2)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_current__default_event(self, *, table: Any, wset: Any) -> Any:
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current()
        table._relative_event.assert_called_once_with(wset.event)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_delta(self, *, table: Any, wset: Any) -> Any:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3, event2)
        table._windowed_delta.assert_called_once_with('k', 30.3, event2)
        assert ret is table._windowed_delta()

    def test_delta__default_event(self, *, table: Any, wset: Any) -> Any:
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3)
        table._windowed_delta.assert_called_once_with('k', 30.3, wset.event)
        assert ret is table._windowed_delta()

    def test_getitem(self, *, wset: Any) -> Any:
        wset.table = {(wset.key, 30.3): 101.1}
        assert wset[30.3] == 101.1

    def test_getitem__event(self, *, app: Any, wset: Any) -> Any:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        ret = wset[e]
        assert isinstance(ret, WindowSet)
        assert ret.key == wset.key
        assert ret.table is wset.table
        assert ret.wrapper is wset.wrapper
        assert ret.event is e

    def test_setitem(self, *, wset: Any) -> Any:
        wset.table = {}
        wset[30.3] = 'val'
        assert wset.table[wset.key, 30.3] == 'val'

    def test_delitem(self, *, wset: Any) -> Any:
        wset.table = {(wset.key, 30.3): 'val'}
        del wset[30.3]
        assert not wset.table

    @pytest.mark.parametrize('meth,expected_op', [('__iadd__', operator.add), ('__isub__', operator.sub), ('__imul__', operator.mul), ('__itruediv__', operator.truediv), ('__ifloordiv__', operator.floordiv), ('__imod__', operator.mod), ('__ipow__', operator.pow), ('__ilshift__', operator.lshift), ('__irshift__', operator.rshift), ('__iand__', operator.and_), ('__ixor__', operator.xor), ('__ior__', operator.or_)])
    def test_operators(self, meth: str, expected_op: Any, *, wset: Any) -> Any:
        other = Mock(name='other')
        op = getattr(wset, meth)
        wset.apply = Mock(name='apply')
        result = op(other)
        wset.apply.assert_called_once_with(expected_op, other)
        assert result is wset.apply()

    def test_repr(self, *, wset: Any) -> Any:
        assert repr(wset)

class test_WindowWrapper:

    def test_name(self, *, wtable: Any) -> Any:
        assert wtable.name == wtable.table.name

    def test_relative_to(self, *, wtable: Any) -> Any:
        relative_to = Mock(name='relative_to')
        w2 = wtable.relative_to(relative_to)
        assert w2.table is wtable.table
        assert w2._get_relative_timestamp is relative_to

    def test_relative_to_now(self, *, table: Any, wtable: Any) -> Any:
        w2 = wtable.relative_to_now()
        assert w2._get_relative_timestamp == wtable.table._relative_now

    def test_relative_to_field(self, *, table: Any, wtable: Any) -> Any:
        table._relative_field = Mock(name='_relative_field')
        field = Mock(name='field')
        w2 = wtable.relative_to_field(field)
        table._relative_field.assert_called_once_with(field)
        assert w2._get_relative_timestamp == table._relative_field()

    def test_relative_to_stream(self, *, table: Any, wtable: Any) -> Any:
        w2 = wtable.relative_to_stream()
        assert w2._get_relative_timestamp == wtable.table._relative_event

    @pytest.mark.parametrize('input,expected', [(DATETIME, DATETIME_TS), (303.333, 303.333), (None, 99999.6)])
    def test_get_timestamp(self, input: Any, expected: float, *, event: Any, wtable: Any) -> Any:
        event.message.timestamp = 99999.6
        if input is not None:
            wtable.get_relative_timestamp = lambda e=None: input
        else:
            wtable.get_relative_timestamp = None
        assert wtable.get_timestamp(event) == expected

    def test_get_timestamp__event_is_None(self, *, event: Any, wtable: Any) -> Any:
        wtable.get_relative_timestamp = None
        with patch('faust.tables.wrappers.current_event') as ce:
            ce.return_value = None
            with pytest.raises(RuntimeError):
                assert wtable.get_timestamp(None)

    def test_on_recover(self, *, wtable: Any, table: Any) -> Any:
        cb = Mock(name='callback')
        wtable.on_recover(cb)
        assert cb in table._recover_callbacks

    def test_contains(self, *, table: Any, wtable: Any) -> Any:
        table._windowed_contains = Mock(name='windowed_contains')
        wtable.get_timestamp = Mock(name='get_timestamp')
        ret = wtable.__contains__('k')
        wtable.get_timestamp.assert_called_once_with()
        table._windowed_contains.assert_called_once_with('k', wtable.get_timestamp())
        assert ret is table._windowed_contains()

    def test_getitem(self, *, wtable: Any) -> Any:
        w = wtable['k2']
        assert isinstance(w, WindowSet)
        assert w.key == 'k2'
        assert w.table is wtable.table
        assert w.wrapper is wtable

    def test_setitem(self, *, table: Any, wtable: Any) -> Any:
        table._set_windowed = Mock(name='set_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        wtable['foo'] = 300
        wtable.get_timestamp.assert_called_once_with()
        table._set_windowed.assert_called_once_with('foo', 300, wtable.get_timestamp())

    def test_delitem(self, *, table: Any, wtable: Any) -> Any:
        table._del_windowed = Mock(name='del_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        del wtable['foo']
        wtable.get_timestamp.assert_called_once_with()
        table._del_windowed.assert_called_once_with('foo', wtable.get_timestamp())

    def test_len__no_key_index_raises(self, *, wtable: Any) -> Any:
        with pytest.raises(NotImplementedError):
            len(wtable)

    def test_as_ansitable__raises(self, *, wtable: Any) -> Any:
        with pytest.raises(NotImplementedError):
            wtable.as_ansitable()

    def test_keys_raises(self, *, wtable: Any) -> Any:
        with pytest.raises(NotImplementedError):
            list(wtable._keys())

    @pytest.mark.parametrize('input', [datetime.now(), 103.33, User.id, lambda s: s])
    def test_relative_handler(self, input: Any, *, wtable: Any) -> Any:
        wtable.get_relative_timestamp = input
        assert wtable.get_relative_timestamp

    def test_relative_handler__invalid_handler(self, *, wtable: Any) -> Any:
        with pytest.raises(ImproperlyConfigured):
            wtable._relative_handler(object())

class test_WindowWrapper_using_key_index:

    @pytest.fixture
    def wset(self, *, iwtable: Any, event: Any) -> Any:
        return WindowSet('k', iwtable.table, iwtable, event)

    @pytest.fixture()
    def data(self, *, freeze_time: Any, iwtable: Any) -> Dict:
        iwtable.key_index_table = {k: 1 for k in self.TABLE_DATA}
        iwtable.table._data = {}
        for w in iwtable.table._window_ranges(freeze_time.time):
            iwtable.table._data.update({(k, w): v for k, v in self.TABLE_DATA.items()})
        return iwtable.table._data

    @pytest.fixture()
    def data_with_30s_delta(self, *, freeze_time: Any, iwtable: Any, data: Dict) -> Dict:
        window = iwtable.table.window
        for key, value in self.TABLE_DATA.items():
            data[key, window.delta(freeze_time.time, 30)] = value + 'delta1'

    @pytest.fixture()
    def remove_a_key(self, *, iwtable: Any, data: Dict) -> Dict:
        remove_key = random.choice(list(self.TABLE_DATA))
        items_leftover = {k: v for k, v in self.TABLE_DATA.items() if k != remove_key}
        iwtable.table._data = {k: v for k, v in iwtable.table._data.items() if k[0] != remove_key}
        return items_leftover

    def test_len(self, *, iwtable: Any) -> Any:
        iwtable.key_index_table = {1: 'A', 2: 'B'}
        assert len(iwtable) == 2

    def test_as_ansitable(self, *, iwtable: Any, data: Dict) -> Any:
        table = iwtable.relative_to_now().as_ansitable()
        print(table)
        assert table
        assert 'foobar' in table
        assert 'AUNIQSTR' in table

    def test_items(self, *, iwtable: Any, data: Dict) -> Any:
        assert same_items(iwtable.relative_to_now(), self.TABLE_DATA)

    def test_items_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict) -> Any:
        assert same_items(iwtable.relative_to_now(), remove_a_key)

    def test_items_now(self, *, iwtable: Any, data: Dict) -> Any:
        assert same_items(iwtable.items().now(), self.TABLE_DATA)

    def test_items_now_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict) -> Any:
        assert same_items(iwtable.items().now(), remove_a_key)

    def test_items_current(self, *, iwtable: Any, data: Dict, current_event: Any) -> Any:
        assert same_items(iwtable.items().current(), self.TABLE_DATA)

    def test_items_current_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict, current_event: Any) -> Any:
        assert same_items(iwtable.items().current(), remove_a_key)

    def test_items_delta(self, *, iwtable: Any, data_with_30s_delta: Dict, current_event: Any) -> Any:
        assert same_items(iwtable.items().delta(30), self.TABLE_DATA_DELTA)

    def test_items_delta_key_not_in_table(self, *, iwtable: Any, data_with_30s_delta: Dict, remove_a_key: Dict, current_event: Any) -> Any:
        expected = {k: v for k, v in self.TABLE_DATA_DELTA.items() if k in remove_a_key}
        assert same_items(iwtable.items().delta(30), expected)

    def test_keys(self, *, iwtable: Any, data: Dict) -> Any:
        assert same(iwtable.relative_to_now().keys(), self.TABLE_DATA)

    def test_keys__now(self, *, iwtable: Any, data: Dict) -> Any:
        assert same(iwtable.relative_to_now().keys().now(), self.TABLE_DATA)

    def test_keys__current(self, *, iwtable: Any, data: Dict, current_event: Any) -> Any:
        keys = iwtable.relative_to_now().keys().current()
        assert same(keys, self.TABLE_DATA)

    def test_keys__delta(self, *, iwtable: Any, data: Dict, current_event: Any) -> Any:
        keys = iwtable.relative_to_now().keys().delta(1000)
        assert same(keys, [])
        keys = iwtable.relative_to_now().keys().delta(10)
        assert same(keys, self.TABLE_DATA)

    def test_iter(self, *, iwtable: Any, data: Dict) -> Any:
        assert same(iwtable.relative_to_now(), self.TABLE_DATA)

    def test_values(self, *, iwtable: Any, data: Dict) -> Any:
        assert same(iwtable.relative_to_now().values(), self.TABLE_DATA.values())

    def test_values_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict) -> Any:
        assert same(iwtable.relative_to_now().values(), remove_a_key.values())

    def test_values_now(self, *, iwtable: Any, data: Dict) -> Any:
        assert same(iwtable.values().now(), self.TABLE_DATA.values())

    def test_values_now_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict) -> Any:
        assert same(iwtable.values().now(), remove_a_key.values())

    def test_values_current(self, *, iwtable: Any, data: Dict, current_event: Any) -> Any:
        assert same(iwtable.values().current(), self.TABLE_DATA.values())

    def test_values_current_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Dict, current_event: Any) -> Any:
        assert same(iwtable.values().current(), remove_a_key.values())

    def test_values_delta(self, *, iwtable: Any, data_with_30s_delta: Dict, current_event: Any) -> Any:
        assert same(iwtable.values().delta(30), self.TABLE_DATA_DELTA.values())

    def test_values_delta_key_not_in_table(self, *, iwtable: Any, data_with_30s_delta: Dict, remove_a_key: Dict, current_event: Any) -> Any:
        expected = {k: v for k, v in self.TABLE_DATA_DELTA.items() if k in remove_a_key}
        assert same(iwtable.values().delta(30), expected.values())

    def test_setitem(self, *, wset: Any) -> Any:
        wset.table = {}
        wset.wrapper.key_index_table = {}
        wset[30.3] = 'val'
        assert wset