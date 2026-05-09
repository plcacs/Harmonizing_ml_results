import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generator, Iterable, cast
import gevent
import ulid
from eth_utils import to_normalized_address
from ulid import MAX_ULID, MIN_ULID, ULID
from raiden.constants import RAIDEN_DB_VERSION, SQLITE_MIN_REQUIRED_VERSION
from raiden.exceptions import InvalidDBData, InvalidNumberInput
from raiden.storage.serialization import SerializationBase
from raiden.storage.utils import DB_SCRIPT_CREATE_TABLES, TimestampedEvent
from raiden.transfer.architecture import Event, State, StateChange
from raiden.utils.system import get_system_spec
from raiden.utils.typing import Address, Any, DatabasePath, Dict, Generic, Iterator, List, NamedTuple, NewType, Optional, RaidenDBVersion, TokenNetworkAddress, Tuple, Type, TypeVar, Union

@dataclass
class Range(Generic[ID]):
    """Inclusive range used to filter database entries."""

    def __post_init__(self):
        if self.first > self.last:
            raise ValueError('last must be larger or equal to first')
LOW_STATECHANGE_ULID = StateChangeID(MIN_ULID)
HIGH_STATECHANGE_ULID = StateChangeID(MAX_ULID)
RANGE_ALL_STATE_CHANGES = Range(LOW_STATECHANGE_ULID, HIGH_STATECHANGE_ULID)

class Operator(Enum):
    NONE = ''
    AND = 'AND'
    OR = 'OR'

class FilteredDBQuery(NamedTuple):
    """
    FilteredDBQuery is a datastructure that helps
    form a list of conditions and how they're grouped
    in order to form more complicated queries
    on the internal JSON representation
    of states / state changes and events.
    Note that it is not used to search
    the top-level attributes of the sqlite tables.
    """

class EventEncodedRecord(NamedTuple):
    pass

class StateChangeEncodedRecord(NamedTuple):
    pass

class SnapshotEncodedRecord(NamedTuple):
    pass

class EventRecord(NamedTuple):
    pass

class StateChangeRecord(NamedTuple):
    pass

class SnapshotRecord(NamedTuple):
    pass

def assert_sqlite_version():
    if sqlite3.sqlite_version_info < SQLITE_MIN_REQUIRED_VERSION:
        return False
    return True

def adapt_ulid_identifier(ulid: ULID) -> bytes:
    return ulid.bytes

def convert_ulid_identifier(data: bytes) -> ULID:
    return ulid.from_bytes(data)

def _sanitize_limit_and_offset(limit: Optional[int], offset: Optional[int]) -> Tuple[int, int]:
    if limit is not None and limit < 0:
        raise InvalidNumberInput('limit must be a positive integer')
    if offset is not None and offset < 0:
        raise InvalidNumberInput('offset must be a positive integer')
    limit = -1 if limit is None else limit
    offset = 0 if offset is None else offset
    return (limit, offset)

def _filter_from_dict(current: Dict[str, Any]) -> Dict[str, Any]:
    """Takes in a nested dictionary as a filter and returns a flattened filter dictionary"""
    filter_ = {}
    for k, v in current.items():
        if isinstance(v, dict):
            for sub, v2 in _filter_from_dict(v).items():
                filter_[f'{k}.{sub}'] = v2
        else:
            filter_[k] = v
    return filter_

def _query_to_string(query: FilteredDBQuery) -> Tuple[str, List[Any]]:
    """
    Converts a query object to a valid SQL string
    which can be used in the WHERE clause.
    A query object will contain a list of dictionaries
    where each key-value pair is used to filter records.
    All the key-value pairs in a dictionary are grouped
    together by `inner_operator` so that they form a SQL condition
    indepedently from other dictionaries in the list.

    Examples:
    - Performing a query with 1 filter
    FilteredDBQuery(
      filters=[{'a': 1, 'b': 2}],
      main_operator=NONE,
      inner_operator='AND'
    )
    Will result in:
    (a=1 AND b=2)

    - Performing a query with multiple filters
      `inner_operator` is used in the inner subqueries
      of the key-value pairs in a single dictionary.
      While `main_operator` is used in the outer query.

    FilteredDBQuery(
      filters=[
        {'a': 1, 'b': 2},
        {'c': 3, 'd': 4},
      ],
      main_operator='OR',
      inner_operator='AND'
    )
    Will result in:
    (a=1 AND b=2) OR (c=3 AND d=4)
    """
    query_where = []
    args = []
    for filter_set in query.filters:
        where_clauses = []
        filters = _filter_from_dict(filter_set)
        for field, value in filters.items():
            where_clauses.append('json_extract(data, ?)=?')
            args.append(f'$.{field}')
            args.append(value)
        filter_set_str = f' {query.inner_operator.value} '.join(where_clauses)
        query_where.append(f'({filter_set_str}) ')
    query_where_str = f' {query.main_operator.value} '.join(query_where)
    return (query_where_str, args)

def _prepend_and_save_ids(ulid_factory: SerializationBase, ids: List[ULID], items: Iterable[Tuple[ULID, Any]]) -> Generator[Tuple[ULID, Any], None, None]:
    for item in items:
        next_id = cast(ULID, ulid_factory.new())
        ids.append(next_id)
        yield (next_id, *item)

def write_state_change(ulid_factory: SerializationBase, cursor: sqlite3.Cursor, state_change: Any) -> ULID:
    """Write `state_change` to the database and returns the corresponding ID."""
    query = 'INSERT INTO state_changes(identifier, data) VALUES(?, ?)'
    new_id = StateChangeID(ulid_factory.new())
    cursor.execute(query, (new_id, state_change))
    return new_id

def write_events(ulid_factory: SerializationBase, cursor: sqlite3.Cursor, events: Iterable[Tuple[ULID, Any]]) -> List[ULID]:
    events_ids = []
    query = 'INSERT INTO state_events(identifier, source_statechange_id, data) VALUES(?, ?, ?)'
    cursor.executemany(query, _prepend_and_save_ids(ulid_factory, events_ids, events))
    return events_ids

class SQLiteStorage:
    def __init__(self, database_path: DatabasePath):
        sqlite3.register_adapter(ULID, adapt_ulid_identifier)
        sqlite3.register_converter('ULID', convert_ulid_identifier)
        conn = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.text_factory = str
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA locking_mode=EXCLUSIVE')
        try:
            conn.execute('PRAGMA journal_mode=PERSIST')
        except sqlite3.DatabaseError:
            raise InvalidDBData(f'Existing DB {database_path} was found to be corrupt at Raiden startup. Manual user intervention required. Bailing.')
        with conn:
            conn.executescript(DB_SCRIPT_CREATE_TABLES)
        self.conn = conn
        self.in_transaction = False
        self._ulid_factories = {}

    # ... rest of the class definition ...
