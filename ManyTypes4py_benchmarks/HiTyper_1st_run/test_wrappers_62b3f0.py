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
DATETIME = datetime.utcnow()
DATETIME_TS = DATETIME.timestamp()

class User(faust.Record):
    pass

@pytest.fixture
def table(*, app: Any):
    return app.Table('name')

@pytest.fixture
def wtable(*, table: Union[list[str], dict[str, typing.Any], list[int]]) -> Union[str, list[tuple[str]], dict]:
    return table.hopping(60, 1, 3600.0)

@pytest.fixture
def iwtable(*, table: Union[list[list[str]], list[dict], None, dict]) -> Union[str, list[tuple[typing.Union[bool,typing.Any]]], list[str]]:
    return table.hopping(60, 1, 3600.0, key_index=True)

@pytest.fixture
def event() -> Mock:
    return Mock(name='event', autospec=Event)

def same_items(a: Union[int, T, typing.Iterable[typing.Any]], b: Union[typing.Iterable, T]) -> bool:
    a_list = _maybe_items(a)
    b_list = _maybe_items(b)
    return same(a_list, b_list)

def _maybe_items(d: Union[dict, dict[str, typing.Any], dict[str, str]]) -> Union[dict, dict[str, typing.Any], dict[str, str], str, dict[str, dict[str, str]]]:
    try:
        items = d.items
    except AttributeError:
        return d
    else:
        return items()

def same(a: Any, b: Any) -> bool:
    return sorted(a) == sorted(b)

@pytest.yield_fixture()
def current_event(*, freeze_time: Union[float, int, datetime.timedelta]) -> typing.Generator:
    with patch('faust.tables.wrappers.current_event') as current_event:
        with patch('faust.tables.base.current_event', current_event):
            current_event.return_value.message.timestamp = freeze_time.time
            yield current_event

class test_WindowSet:

    @pytest.fixture
    def wset(self, *, wtable, event: Union[T, str, set[str]]) -> WindowSet:
        return WindowSet('k', wtable.table, wtable, event)

    def test_constructor(self, *, event: Union[str, dict[str, typing.Any], dict[str, bool]], table: str, wset: Union[dict, str], wtable: Union[str, bytes, None]) -> None:
        assert wset.key == 'k'
        assert wset.table is table
        assert wset.wrapper is wtable
        assert wset.event is event

    def test_apply(self, *, wset: Union[unittesmock.Mock, typing.Callable], event: unittesmock.Mock) -> None:
        Mock(name='event2', autospec=Event)
        wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        ret = wset.apply(operator.add, 'val')
        wset.wrapper.get_timestamp.assert_called_once_with(wset.event)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', wset.wrapper.get_timestamp())
        assert ret is wset

    def mock_get_timestamp(self, wset: Union[str, list[str]]) -> Mock:
        m = wset.wrapper.get_timestamp = Mock(name='wrapper.get_timestamp')
        return m

    def test_apply__custom_event(self, *, wset: Any, event: Message) -> None:
        event2 = Mock(name='event2', autospec=Event)
        wset.table._apply_window_op = Mock(name='_apply_window_op')
        get_timestamp = self.mock_get_timestamp(wset)
        ret = wset.apply(operator.add, 'val', event2)
        get_timestamp.assert_called_once_with(event2)
        wset.table._apply_window_op.assert_called_once_with(operator.add, 'k', 'val', get_timestamp())
        assert ret is wset

    def test_value(self, *, event: Union[dict[str, typing.Any], events.events_base.EventType, edp.journal.Event], wset: Union[Event, aioquic.quic.events.QuicEvent, typing.MutableMapping]) -> None:
        get_timestamp = self.mock_get_timestamp(wset)
        wset.table._windowed_timestamp = Mock(name='_windowed_timestamp')
        assert wset.value(event)
        wset.table._windowed_timestamp.assert_called_once_with('k', get_timestamp())

    def test_now(self, *, wset: unittesmock.Mock) -> None:
        wset.table._windowed_now = Mock(name='_windowed_now')
        ret = wset.now()
        wset.table._windowed_now.assert_called_once_with('k')
        assert ret is wset.table._windowed_now()

    def test_current(self, *, table: dict, wset: Union[str, types.LocatorType, bool]) -> None:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current(event2)
        table._relative_event.assert_called_once_with(event2)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_current__default_event(self, *, table: dict, wset: Union[dict[str, typing.Sequence[str]], typing.Collection]) -> None:
        table._windowed_timestamp = Mock(name='_windowed_timestamp')
        table._relative_event = Mock(name='_relative_event')
        ret = wset.current()
        table._relative_event.assert_called_once_with(wset.event)
        table._windowed_timestamp.assert_called_once_with('k', table._relative_event())
        assert ret is table._windowed_timestamp()

    def test_delta(self, *, table: Union[typing.Collection, list[str]], wset: Any) -> None:
        event2 = Mock(name='event2', autospec=Event)
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3, event2)
        table._windowed_delta.assert_called_once_with('k', 30.3, event2)
        assert ret is table._windowed_delta()

    def test_delta__default_event(self, *, table: Any, wset: Union[dict[str, typing.Any], dict, list[str]]) -> None:
        table._windowed_delta = Mock(name='_windowed_delta')
        ret = wset.delta(30.3)
        table._windowed_delta.assert_called_once_with('k', 30.3, wset.event)
        assert ret is table._windowed_delta()

    def test_getitem(self, *, wset) -> None:
        wset.table = {(wset.key, 30.3): 101.1}
        assert wset[30.3] == 101.1

    def test_getitem__event(self, *, app: Any, wset: str) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        ret = wset[e]
        assert isinstance(ret, WindowSet)
        assert ret.key == wset.key
        assert ret.table is wset.table
        assert ret.wrapper is wset.wrapper
        assert ret.event is e

    def test_setitem(self, *, wset: Any) -> None:
        wset.table = {}
        wset[30.3] = 'val'
        assert wset.table[wset.key, 30.3] == 'val'

    def test_setitem__event(self, *, app: Any, wset: Any) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        with pytest.raises(NotImplementedError):
            wset[e] = 'val'

    def test_delitem(self, *, wset: Any) -> None:
        wset.table = {(wset.key, 30.3): 'val'}
        del wset[30.3]
        assert not wset.table

    def test_delitem__event(self, *, app: Any, wset: Any) -> None:
        e = Event(app, key='KK', value='VV', headers={}, message=Mock(name='message', autospec=Message))
        with pytest.raises(NotImplementedError):
            del wset[e]

    @pytest.mark.parametrize('meth,expected_op', [('__iadd__', operator.add), ('__isub__', operator.sub), ('__imul__', operator.mul), ('__itruediv__', operator.truediv), ('__ifloordiv__', operator.floordiv), ('__imod__', operator.mod), ('__ipow__', operator.pow), ('__ilshift__', operator.lshift), ('__irshift__', operator.rshift), ('__iand__', operator.and_), ('__ixor__', operator.xor), ('__ior__', operator.or_)])
    def test_operators(self, meth: Union[str, typing.Type, dict[str, typing.Any]], expected_op: Union[str, int], *, wset: Union[str, typing.Callable, eggman.types.Handler]) -> None:
        other = Mock(name='other')
        op = getattr(wset, meth)
        wset.apply = Mock(name='apply')
        result = op(other)
        wset.apply.assert_called_once_with(expected_op, other)
        assert result is wset.apply()

    def test_repr(self, *, wset: Union[float, str, int]) -> None:
        assert repr(wset)

class test_WindowWrapper:

    def test_name(self, *, wtable: Any) -> None:
        assert wtable.name == wtable.table.name

    def test_relative_to(self, *, wtable: Union[str, float, int]) -> None:
        relative_to = Mock(name='relative_to')
        w2 = wtable.relative_to(relative_to)
        assert w2.table is wtable.table
        assert w2._get_relative_timestamp is relative_to

    def test_relative_to_now(self, *, table: Any, wtable: list[str]) -> None:
        w2 = wtable.relative_to_now()
        assert w2._get_relative_timestamp == wtable.table._relative_now

    def test_relative_to_field(self, *, table: Union[dict[str, typing.Sequence[str]], dict[str, typing.Sequence[typing.Any]], typing.Collection], wtable: Union[dict[str, typing.Sequence[typing.Any]], dict[str, dict[str, str]], dict[str, typing.Sequence[str]]]) -> None:
        table._relative_field = Mock(name='_relative_field')
        field = Mock(name='field')
        w2 = wtable.relative_to_field(field)
        table._relative_field.assert_called_once_with(field)
        assert w2._get_relative_timestamp == table._relative_field()

    def test_relative_to_stream(self, *, table: Any, wtable: list[str]) -> None:
        w2 = wtable.relative_to_stream()
        assert w2._get_relative_timestamp == wtable.table._relative_event

    @pytest.mark.parametrize('input,expected', [(DATETIME, DATETIME_TS), (303.333, 303.333), (None, 99999.6)])
    def test_get_timestamp(self, input: Union[dict[str, typing.Any], None, str], expected: Union[dict, tuple, core_lib.core.models.Event], *, event: Union[dict, dict[str, typing.Any]], wtable: Union[str, None, dict, dict[str, typing.Any]]) -> None:
        event.message.timestamp = 99999.6
        if input is not None:
            wtable.get_relative_timestamp = lambda e=None: input
        else:
            wtable.get_relative_timestamp = None
        assert wtable.get_timestamp(event) == expected

    def test_get_timestamp__event_is_None(self, *, event: Union[widark.Event, dict], wtable: Union[dict, widark.Event, apscheduler.events.JobExecutionEvent]) -> None:
        wtable.get_relative_timestamp = None
        with patch('faust.tables.wrappers.current_event') as ce:
            ce.return_value = None
            with pytest.raises(RuntimeError):
                assert wtable.get_timestamp(None)

    def test_on_recover(self, *, wtable: Union[bool, typing.Callable[..., None], dict], table: Any) -> None:
        cb = Mock(name='callback')
        wtable.on_recover(cb)
        assert cb in table._recover_callbacks

    def test_contains(self, *, table: dict, wtable: dict) -> None:
        table._windowed_contains = Mock(name='windowed_contains')
        wtable.get_timestamp = Mock(name='get_timestamp')
        ret = wtable.__contains__('k')
        wtable.get_timestamp.assert_called_once_with()
        table._windowed_contains.assert_called_once_with('k', wtable.get_timestamp())
        assert ret is table._windowed_contains()

    def test_getitem(self, *, wtable: Any) -> None:
        w = wtable['k2']
        assert isinstance(w, WindowSet)
        assert w.key == 'k2'
        assert w.table is wtable.table
        assert w.wrapper is wtable

    def test_setitem(self, *, table, wtable) -> None:
        table._set_windowed = Mock(name='set_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        wtable['foo'] = 300
        wtable.get_timestamp.assert_called_once_with()
        table._set_windowed.assert_called_once_with('foo', 300, wtable.get_timestamp())

    def test_setitem__key_is_WindowSet(self, *, wtable: Any) -> None:
        wtable['k2'] = wtable['k2']

    def test_delitem(self, *, table, wtable) -> None:
        table._del_windowed = Mock(name='del_windowed')
        wtable.get_timestamp = Mock(name='get_timestamp')
        del wtable['foo']
        wtable.get_timestamp.assert_called_once_with()
        table._del_windowed.assert_called_once_with('foo', wtable.get_timestamp())

    def test_len__no_key_index_raises(self, *, wtable: str) -> None:
        with pytest.raises(NotImplementedError):
            len(wtable)

    def test_as_ansitable__raises(self, *, wtable: Any) -> None:
        with pytest.raises(NotImplementedError):
            wtable.as_ansitable()

    def test_keys_raises(self, *, wtable: int) -> None:
        with pytest.raises(NotImplementedError):
            list(wtable._keys())

    @pytest.mark.parametrize('input', [datetime.now(), 103.33, User.id, lambda s: s])
    def test_relative_handler(self, input: Union[T, str, dict], *, wtable: Union[str, list, float]) -> None:
        wtable.get_relative_timestamp = input
        assert wtable.get_relative_timestamp

    def test_relative_handler__invalid_handler(self, *, wtable: unittesmock.Mock) -> None:
        with pytest.raises(ImproperlyConfigured):
            wtable._relative_handler(object())

class test_WindowWrapper_using_key_index:
    TABLE_DATA = {'foobar': 'AUNIQSTR', 'xuzzy': 'BUNIQSTR'}
    TABLE_DATA_DELTA = {'foobar': 'AUNIQSTRdelta1', 'xuzzy': 'BUNIQSTRdelta1'}

    @pytest.fixture
    def wset(self, *, iwtable: Union[T, str, set[str]], event: Union[T, str, set[str]]) -> WindowSet:
        return WindowSet('k', iwtable.table, iwtable, event)

    @pytest.fixture()
    def data(self, *, freeze_time: Union[int, float], iwtable: Union[str, float, None, int]) -> dict:
        iwtable.key_index_table = {k: 1 for k in self.TABLE_DATA}
        iwtable.table._data = {}
        for w in iwtable.table._window_ranges(freeze_time.time):
            iwtable.table._data.update({(k, w): v for k, v in self.TABLE_DATA.items()})
        return iwtable.table._data

    @pytest.fixture()
    def data_with_30s_delta(self, *, freeze_time: Union[str, float, None], iwtable: Union[float, str, None, typing.Callable], data: Union[str, float, None]) -> None:
        window = iwtable.table.window
        for key, value in self.TABLE_DATA.items():
            data[key, window.delta(freeze_time.time, 30)] = value + 'delta1'

    @pytest.fixture()
    def remove_a_key(self, *, iwtable: Union[dict, typing.Mapping, list], data: Union[dict, dict[str, typing.Any]]) -> dict:
        remove_key = random.choice(list(self.TABLE_DATA))
        items_leftover = {k: v for k, v in self.TABLE_DATA.items() if k != remove_key}
        iwtable.table._data = {k: v for k, v in iwtable.table._data.items() if k[0] != remove_key}
        return items_leftover

    def test_len(self, *, iwtable: Union[int, str, float]) -> None:
        iwtable.key_index_table = {1: 'A', 2: 'B'}
        assert len(iwtable) == 2

    def test_as_ansitable(self, *, iwtable: Union[bytes, dict], data: Union[dict[str, typing.Any], bytes, list[dict[str, typing.Any]]]) -> None:
        table = iwtable.relative_to_now().as_ansitable()
        print(table)
        assert table
        assert 'foobar' in table
        assert 'AUNIQSTR' in table

    def test_items(self, *, iwtable: Union[dict, dict[str, typing.Any], typing.Mapping], data: Union[list[dict[str, typing.Any]], dict[str, typing.Any], dict]) -> None:
        assert same_items(iwtable.relative_to_now(), self.TABLE_DATA)

    def test_items_keys_in_index_not_in_table(self, *, iwtable: Union[dict, list[str]], remove_a_key: Union[dict, list[str]]) -> None:
        assert same_items(iwtable.relative_to_now(), remove_a_key)

    def test_items_now(self, *, iwtable: Union[list[dict], dict[str, typing.Any], dict], data: Union[list[dict[str, typing.Any]], str, dict]) -> None:
        assert same_items(iwtable.items().now(), self.TABLE_DATA)

    def test_items_now_keys_in_index_not_in_table(self, *, iwtable: Union[str, list[str], int], remove_a_key: Union[str, list[str], int]) -> None:
        assert same_items(iwtable.items().now(), remove_a_key)

    def test_items_current(self, *, iwtable: Union[dict[str, typing.Any], dict, tuple], data: Union[watchdog.events.FileSystemEvent, dict[str, typing.Any], events.EventT], current_event: Union[watchdog.events.FileSystemEvent, dict[str, typing.Any], events.EventT]) -> None:
        assert same_items(iwtable.items().current(), self.TABLE_DATA)

    def test_items_current_keys_in_index_not_in_table(self, *, iwtable: Union[str, list[dict[str, typing.Any]], list[str]], remove_a_key: Union[str, list[dict[str, typing.Any]], list[str]], current_event: bool) -> None:
        assert same_items(iwtable.items().current(), remove_a_key)

    def test_items_delta(self, *, iwtable: Union[dict[str, typing.Any], float, list], data_with_30s_delta: Union[str, bool, tuple], current_event: Union[str, bool, tuple]) -> None:
        assert same_items(iwtable.items().delta(30), self.TABLE_DATA_DELTA)

    def test_items_delta_key_not_in_table(self, *, iwtable: Union[bool, dict], data_with_30s_delta: bool, remove_a_key: Union[bool, None, dict[str, dict[str, typing.Any]]], current_event: bool) -> None:
        expected = {k: v for k, v in self.TABLE_DATA_DELTA.items() if k in remove_a_key}
        assert same_items(iwtable.items().delta(30), expected)

    def test_keys(self, *, iwtable: Union[dict, slp.util.types.Embeddings, dict[str, typing.Any]], data: Union[bytes, dict, str]) -> None:
        assert same(iwtable.relative_to_now().keys(), self.TABLE_DATA)

    def test_keys__now(self, *, iwtable: Union[dict, dict[str, typing.Any], str], data: Union[bytes, dict]) -> None:
        assert same(iwtable.relative_to_now().keys().now(), self.TABLE_DATA)

    def test_keys__current(self, *, iwtable: dict, data: Union[dict[str, typing.Any], tuple[int], bytes], current_event: Union[dict[str, typing.Any], tuple[int], bytes]) -> None:
        keys = iwtable.relative_to_now().keys().current()
        assert same(keys, self.TABLE_DATA)

    def test_keys__delta(self, *, iwtable: dict, data: Union[str, dict[str, typing.Any], typing.Type], current_event: Union[str, dict[str, typing.Any], typing.Type]) -> None:
        keys = iwtable.relative_to_now().keys().delta(1000)
        assert same(keys, [])
        keys = iwtable.relative_to_now().keys().delta(10)
        assert same(keys, self.TABLE_DATA)

    def test_iter(self, *, iwtable: Union[dict, bytes], data: Union[dict, str]) -> None:
        assert same(iwtable.relative_to_now(), self.TABLE_DATA)

    def test_values(self, *, iwtable: Union[dict, dict[str, typing.Any], str], data: Union[dict, None, dict[str, typing.Any], bytes]) -> None:
        assert same(iwtable.relative_to_now().values(), self.TABLE_DATA.values())

    def test_values_keys_in_index_not_in_table(self, *, iwtable: Union[str, tuple[typing.Union[float,int]], int], remove_a_key: Union[str, tuple[typing.Union[float,int]], int]) -> None:
        assert same(iwtable.relative_to_now().values(), remove_a_key.values())

    def test_values_now(self, *, iwtable: Union[dict, None, dict[str, typing.Any]], data: Union[dict[str, typing.Any], dict, list[tuple]]) -> None:
        assert same(iwtable.values().now(), self.TABLE_DATA.values())

    def test_values_now_keys_in_index_not_in_table(self, *, iwtable: Union[str, bool, typing.Callable], remove_a_key: Union[str, bool, typing.Callable]) -> None:
        assert same(iwtable.values().now(), remove_a_key.values())

    def test_values_current(self, *, iwtable: Union[int, str, Event], data: Union[dict[str, str], list[dict[str, typing.Any]], int], current_event: Union[dict[str, str], list[dict[str, typing.Any]], int]) -> None:
        assert same(iwtable.values().current(), self.TABLE_DATA.values())

    def test_values_current_keys_in_index_not_in_table(self, *, iwtable: Union[str, Message, typing.Callable], remove_a_key: Union[str, Message, typing.Callable], current_event: Union[bool, dict, typing.Callable]) -> None:
        assert same(iwtable.values().current(), remove_a_key.values())

    def test_values_delta(self, *, iwtable: Union[dict[str, typing.Any], float, dict], data_with_30s_delta: Union[int, str], current_event: Union[int, str]) -> None:
        assert same(iwtable.values().delta(30), self.TABLE_DATA_DELTA.values())

    def test_values_delta_key_not_in_table(self, *, iwtable: Union[float, str, dict[str, int]], data_with_30s_delta: bool, remove_a_key: Union[bool, None, dict[str, dict[str, typing.Any]]], current_event: bool) -> None:
        expected = {k: v for k, v in self.TABLE_DATA_DELTA.items() if k in remove_a_key}
        assert same(iwtable.values().delta(30), expected.values())

    def test_setitem(self, *, wset: Any) -> None:
        wset.table = {}
        wset.wrapper.key_index_table = {}
        wset[30.3] = 'val'
        assert wset.table[wset.key, 30.3] == 'val'
        assert wset.key in wset.wrapper.key_index_table
        wset[30.3] = 'val2'
        assert wset.table[wset.key, 30.3] == 'val2'

    def test_delitem(self, *, wset: Any) -> None:
        wset.table = {(wset.key, 30.3): 'val'}
        wset.wrapper.key_index_table = {wset.key: 1}
        del wset[30.3]
        assert not wset.table
        assert not wset.wrapper.key_index_table