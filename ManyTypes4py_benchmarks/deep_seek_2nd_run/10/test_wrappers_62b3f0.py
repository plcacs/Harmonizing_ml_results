import operator
import random
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import faust
import pytest
from faust.events import Event
from faust.exceptions import ImproperlyConfigured
from faust.tables.wrappers import WindowSet
from faust.types import Message, TableT
from mode.utils.mocks import Mock, patch

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

DATETIME: datetime = datetime.utcnow()
DATETIME_TS: float = DATETIME.timestamp()

class User(faust.Record):
    pass

@pytest.fixture
def table(*, app: faust.App) -> TableT:
    return app.Table('name')

@pytest.fixture
def wtable(*, table: TableT) -> faust.Table:
    return table.hopping(60, 1, 3600.0)

@pytest.fixture
def iwtable(*, table: TableT) -> faust.Table:
    return table.hopping(60, 1, 3600.0, key_index=True)

@pytest.fixture
def event() -> Mock:
    return Mock(name='event', autospec=Event)

def same_items(a: Union[Dict[K, V], Iterable[Tuple[K, V]]], b: Union[Dict[K, V], Iterable[Tuple[K, V]]]) -> bool:
    a_list = _maybe_items(a)
    b_list = _maybe_items(b)
    return same(a_list, b_list)

def _maybe_items(d: Union[Dict[K, V], Iterable[Tuple[K, V]]]) -> List[Tuple[K, V]]:
    try:
        items = d.items
    except AttributeError:
        return list(d)
    else:
        return list(items())

def same(a: Iterable[T], b: Iterable[T]) -> bool:
    return sorted(a) == sorted(b)

@pytest.yield_fixture()
def current_event(*, freeze_time: Any) -> Mock:
    with patch('faust.tables.wrappers.current_event') as current_event:
        with patch('faust.tables.base.current_event', current_event):
            current_event.return_value.message.timestamp = freeze_time.time
            yield current_event

class test_WindowSet:

    @pytest.fixture
    def wset(self, *, wtable: faust.Table, event: Mock) -> WindowSet:
        return WindowSet('k', wtable.table, wtable, event)

    def test_constructor(self, *, event: Mock, table: TableT, wset: WindowSet, wtable: faust.Table) -> None:
        assert wset.key == 'k'
        assert wset.table is table
        assert wset.wrapper is wtable
        assert wset.event is event

    def test_apply(self, *, wset: WindowSet, event: Mock) -> None:
        Mock(name='event2', autospec=Event)
        wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        ret = wset.apply(operator.add, 'val')
        wset.wrapper.get_timestamp.assert_called_once_with(wset.event)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', wset.wrapper.get_timestamp())
        assert ret is wset

    def mock_get_timestamp(self, wset: WindowSet) -> Mock:
        m = wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        return m

    def test_apply__custom_event(self, *, wset: WindowSet, event: Mock) -> None:
        event2 = Mock(name='event2', autospec=Event)
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        get_timestamp = self.mock_get_timestamp(wset)
        ret = wset.apply(operator.add, 'val', event2)
        get_timestamp.assert_called_once_with(event2)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', get_timestamp())
        assert ret is wset

    def test_value(self, *, event: Mock, wset: WindowSet) -> None:
        get_timestamp = self.mock_get_timestamp(wset)
        wset.table._windowed_timestamp = Mock(name='_windowed_timestamp')
        assert wset.value(event)
        wset.table._windowed_timestamp.assert_called_once_with('k', get_timestamp())

    def test_now(self, *, wset: WindowSet) -> None:
        wset.table._windowed_now = Mock(name='_windowed_now')
        ret = wset.now()
        wset.table._windowed_now.assert_called_once_with('k')
        assert ret is wset.table._windowed_now()

    def test_current(self, *, table: TableT, wset: WindowSet) -> None:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current(event2)
        table._relative_event.assert_called_once_with(event2)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_current__default_event(self, *, table: TableT, wset: WindowSet) -> None:
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current()
        table._relative_event.assert_called_once_with(wset.event)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_delta(self, *, table: TableT, wset: WindowSet) -> None:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3, event2)
        table._windowed_delta.assert_called_once_with('k', 30.3, event2)
        assert ret is table._windowed_delta()

    def test_delta__default_event(self, *, table: TableT, wset: WindowSet) -> None:
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3)
        table._windowed_delta.assert_called_once_with('k', 30.3, wset.event)
        assert ret is table._windowed_delta()

    def test_getitem(self, *, wset: WindowSet) -> None:
        wset.table = {(wset.key, 30.3): 101.1}
        assert wset[30.3] == 101.1

    def test_getitem__event(self, *, app: faust.App, wset: WindowSet) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        ret = wset[e]
        assert isinstance(ret, WindowSet)
        assert ret.key == wset.key
        assert ret.table is wset.table
        assert ret.wrapper is wset.wrapper
        assert ret.event is e

    def test_setitem(self, *, wset: WindowSet) -> None:
        wset.table = {}
        wset[30.3] = 'val'
        assert wset.table[wset.key, 30.3] == 'val'

    def test_setitem__event(self, *, app: faust.App, wset: WindowSet) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        with pytest.raises(NotImplementedError):
            wset[e] = 'val'

    def test_delitem(self, *, wset: WindowSet) -> None:
        wset.table = {(wset.key, 30.3): 'val'}
        del wset[30.3]
        assert not wset.table

    def test_delitem__event(self, *, app: faust.App, wset: WindowSet) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        with pytest.raises(NotImplementedError):
            del wset[e]

    @pytest.mark.parametrize('meth,expected_op', [
        ('__iadd__', operator.add),
        ('__isub__', operator.sub),
        ('__imul__', operator.mul),
        ('__itruediv__', operator.truediv),
        ('__ifloordiv__', operator.floordiv),
        ('__imod__', operator.mod),
        ('__ipow__', operator.pow),
        ('__ilshift__', operator.lshift),
        ('__irshift__', operator.rshift),
        ('__iand__', operator.and_),
        ('__ixor__', operator.xor),
        ('__ior__', operator.or_),
    ])
    def test_operators(self, meth: str, expected_op: Callable, *, wset: WindowSet) -> None:
        other = Mock(name='other')
        op = getattr(wset, meth)
        wset.apply = Mock(name='apply')
        result = op(other)
        wset.apply.assert_called_once_with(expected_op, other)
        assert result is wset.apply()

    def test_repr(self, *, wset: WindowSet) -> None:
        assert repr(wset)

class test_WindowWrapper:

    def test_name(self, *, wtable: faust.Table) -> None:
        assert wtable.name == wtable.table.name

    def test_relative_to(self, *, wtable: faust.Table) -> None:
        relative_to = Mock(name='relative_to')
        w2 = wtable.relative_to(relative_to)
        assert w2.table is wtable.table
        assert w2._get_relative_timestamp is relative_to

    def test_relative_to_now(self, *, table: TableT, wtable: faust.Table) -> None:
        w2 = wtable.relative_to_now()
        assert w2._get_relative_timestamp == wtable.table._relative_now

    def test_relative_to_field(self, *, table: TableT, wtable: faust.Table) -> None:
        table._relative_field = Mock(name='_relative_field')
        field = Mock(name='field')
        w2 = wtable.relative_to_field(field)
        table._relative_field.assert_called_once_with(field)
        assert w2._get_relative_timestamp == table._relative_field()

    def test_relative_to_stream(self, *, table: TableT, wtable: faust.Table) -> None:
        w2 = wtable.relative_to_stream()
        assert w2._get_relative_timestamp == wtable.table._relative_event

    @pytest.mark.parametrize('input,expected', [
        (DATETIME, DATETIME_TS),
        (303.333, 303.333),
        (None, 99999.6),
    ])
    def test_get_timestamp(self, input: Optional[Union[datetime, float]], expected: float, *, event: Mock, wtable: faust.Table) -> None:
        event.message.timestamp = 99999.6
        if input is not None:
            wtable.get_relative_timestamp = lambda e=None: input
        else:
            wtable.get_relative_timestamp = None
        assert wtable.get_timestamp(event) == expected

    def test_get_timestamp__event_is_None(self, *, event: Mock, wtable: faust.Table) -> None:
        wtable.get_relative_timestamp = None
        with patch('faust.tables.wrappers.current_event') as ce:
            ce.return_value = None
            with pytest.raises(RuntimeError):
                assert wtable.get_timestamp(None)

    def test_on_recover(self, *, wtable: faust.Table, table: TableT) -> None:
        cb = Mock(name='callback')
        wtable.on_recover(cb)
        assert cb in table._recover_callbacks

    def test_contains(self, *, table: TableT, wtable: faust.Table) -> None:
        table._windowed_contains = Mock(name='windowed_contains')
        wtable.get_timestamp = Mock(name='get_timestamp')
        ret = wtable.__contains__('k')
        wtable.get_timestamp.assert_called_once_with()
        table._windowed_contains.assert_called_once_with('k', wtable.get_timestamp())
        assert ret is table._windowed_contains()

    def test_getitem(self, *, wtable: faust.Table) -> None:
        w = wtable['k2']
        assert isinstance(w, WindowSet)
        assert w.key == 'k2'
        assert w.table is wtable.table
        assert w.wrapper is wtable

    def test_setitem(self, *, table: TableT, wtable: faust.Table) -> None:
        table._set_windowed = Mock(name='set_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        wtable['foo'] = 300
        wtable.get_timestamp.assert_called_once_with()
        table._set_windowed.assert_called_once_with('foo', 300, wtable.get_timestamp())

    def test_setitem__key_is_WindowSet(self, *, wtable: faust.Table) -> None:
        wtable['k2'] = wtable['k2']

    def test_delitem(self, *, table: TableT, wtable: faust.Table) -> None:
        table._del_windowed = Mock(name='del_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        del wtable['foo']
        wtable.get_timestamp.assert_called_once_with()
        table._del_windowed.assert_called_once_with('foo', wtable.get_timestamp())

    def test_len__no_key_index_raises(self, *, wtable: faust.Table) -> None:
        with pytest.raises(NotImplementedError):
            len(wtable)

    def test_as_ansitable__raises(self, *, wtable: faust.Table) -> None:
        with pytest.raises(NotImplementedError):
            wtable.as_ansitable()

    def test_keys_raises(self, *, wtable: faust.Table) -> None:
        with pytest.raises(NotImplementedError):
            list(wtable._keys())

    @pytest.mark.parametrize('input', [
        datetime.now(),
        103.33,
        User.id,
        lambda s: s,
    ])
    def test_relative_handler(self, input: Any, *, wtable: faust.Table) -> None:
        wtable.get_relative_timestamp = input
        assert wtable.get_relative_timestamp

    def test_relative_handler__invalid_handler(self, *, wtable: faust.Table) -> None:
        with pytest.raises(ImproperlyConfigured):
            wtable._relative_handler(object())

class test_WindowWrapper_using_key_index:
    TABLE_DATA: Dict[str, str] = {'foobar': 'AUNIQSTR', 'xuzzy': 'BUNIQSTR'}
    TABLE_DATA_DELTA: Dict[str, str] = {'foobar': 'AUNIQSTRdelta1', 'xuzzy': 'BUNIQSTRdelta1'}

    @pytest.fixture
    def wset(self, *, iwtable: faust.Table, event: Mock) -> WindowSet:
        return WindowSet('k', iwtable.table, iwtable, event)

    @pytest.fixture()
    def data(self, *, freeze_time: Any, iwtable: faust.Table) -> Dict[Tuple[str, float], str]:
        iwtable.key_index_table = {k: 1 for k in self.TABLE_DATA}
        iwtable.table._data = {}
        for w in iwtable.table._window_ranges(freeze_time.time):
            iwtable.table._data.update({(k, w): v for k, v in self.TABLE_DATA.items()})
        return iwtable.table._data

    @pytest.fixture()
    def data_with_30s_delta(self, *, freeze_time: Any, iwtable: faust.Table, data: Dict[Tuple[str, float], str]) -> None:
        window = iwtable.table.window
        for key, value in self.TABLE_DATA.items():
            data[key, window.delta(freeze_time.time, 30)] = value + 'delta1'

    @pytest.fixture()
    def remove_a_key(self, *, iwtable: faust.Table, data: Dict[Tuple[str, float], str]) -> Dict[str, str]:
        remove_key = random.choice(list(self.TABLE_DATA))
        items_leftover = {k: v for k, v in self.TABLE_DATA.items() if k != remove_key}
        iwtable.table._data = {k: v for k, v in iwtable.table._data.items() if k[0] != remove_key}
        return items_leftover

    def test_len(self, *, iwtable: faust.Table) -> None:
        iwtable.key_index_table = {1: 'A', 2: 'B'}
        assert len(iwtable) == 2

    def test_as_ansitable(self, *, iwtable: faust.Table, data: Dict[Tuple[str, float], str]) -> None:
        table = iwtable.relative_to_now().as_ansitable()
        print(table)
        assert table
        assert 'foobar' in table
        assert 'AUNIQSTR' in table

    def test_items(self, *, iwtable: faust.Table, data: Dict[Tuple[str, float], str]) -> None:
        assert same_items(iwtable.relative_to_now(), self.TABLE_DATA)

    def test_items_keys_in_index_not_in_table(self, *, iwtable: faust.Table, remove_a_key: Dict[str, str]) -> None:
        assert same_items(iwtable.relative_to_now(), remove_a_key)

    def test_items_now(self, *, iwtable: faust.Table, data: Dict[Tuple