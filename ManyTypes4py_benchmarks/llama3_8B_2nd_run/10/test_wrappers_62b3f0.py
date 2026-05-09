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
from typing import Any, Dict, List, Tuple

DATETIME = datetime.utcnow()
DATETIME_TS = DATETIME.timestamp()

class User(faust.Record):
    pass

@pytest.fixture
def table(*, app: Any) -> faust.Table:
    return app.Table('name')

@pytest.fixture
def wtable(*, table: faust.Table) -> WindowSet:
    return table.hopping(60, 1, 3600.0)

@pytest.fixture
def iwtable(*, table: faust.Table) -> WindowSet:
    return table.hopping(60, 1, 3600.0, key_index=True)

@pytest.fixture
def event() -> Event:
    return Mock(name='event', autospec=Event)

def same_items(a: Dict[Tuple[str, float], str], b: Dict[Tuple[str, float], str]) -> bool:
    a_list = _maybe_items(a)
    b_list = _maybe_items(b)
    return same(a_list, b_list)

def _maybe_items(d: Dict[Tuple[str, float], str]) -> List[Tuple[str, float]]:
    try:
        items = d.items
    except AttributeError:
        return list(d.items())
    else:
        return list(items())

def same(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> bool:
    return sorted(a) == sorted(b)

@pytest.yield_fixture
def current_event(*, freeze_time: float) -> Event:
    with patch('faust.tables.wrappers.current_event') as current_event:
        with patch('faust.tables.base.current_event', current_event):
            current_event.return_value.message.timestamp = freeze_time
            yield current_event

class test_WindowSet:
    # ... rest of the code ...

class test_WindowWrapper:
    # ... rest of the code ...

class test_WindowWrapper_using_key_index:
    TABLE_DATA: Dict[str, str] = {'foobar': 'AUNIQSTR', 'xuzzy': 'BUNIQSTR'}
    TABLE_DATA_DELTA: Dict[str, str] = {'foobar': 'AUNIQSTRdelta1', 'xuzzy': 'BUNIQSTRdelta1'}

    @pytest.fixture
    def wset(self, *, iwtable: WindowSet, event: Event) -> WindowSet:
        return WindowSet('k', iwtable.table, iwtable, event)

    # ... rest of the code ...
