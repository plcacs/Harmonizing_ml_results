import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import (
    Any, Dict, Generator, Generic, Iterable, Iterator, List, NamedTuple, NewType, 
    Optional, Sequence, Tuple, Type, TypeVar, Union, cast, overload
)
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
from raiden.utils.typing import (
    Address, Any, DatabasePath, Dict, Generic, Iterator, List, NamedTuple, 
    NewType, Optional, RaidenDBVersion, TokenNetworkAddress, Tuple, Type, 
    TypeVar, Union
)

StateChangeID = NewType('StateChangeID', ULID)
SnapshotID = NewType('SnapshotID', ULID)
EventID = NewType('EventID', ULID)
ID = TypeVar('ID', StateChangeID, SnapshotID, EventID)

@dataclass
class Range(Generic[ID]):
    first: ID
    last: ID

    def __post_init__(self) -> None:
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
    filters: List[Dict[str, Any]]
    main_operator: Operator
    inner_operator: Operator

class EventEncodedRecord(NamedTuple):
    event_identifier: EventID
    state_change_identifier: StateChangeID
    data: str

class StateChangeEncodedRecord(NamedTuple):
    state_change_identifier: StateChangeID
    data: str

class SnapshotEncodedRecord(NamedTuple):
    identifier: SnapshotID
    state_change_qty: int
    state_change_identifier: Optional[StateChangeID]
    data: str

class EventRecord(NamedTuple):
    event_identifier: EventID
    state_change_identifier: StateChangeID
    data: Event

class StateChangeRecord(NamedTuple):
    state_change_identifier: StateChangeID
    data: StateChange

class SnapshotRecord(NamedTuple):
    identifier: SnapshotID
    state_change_qty: int
    state_change_identifier: StateChangeID
    data: State

def assert_sqlite_version() -> bool:
    if sqlite3.sqlite_version_info < SQLITE_MIN_REQUIRED_VERSION:
        return False
    return True

def adapt_ulid_identifier(ulid: ULID) -> bytes:
    return ulid.bytes

def convert_ulid_identifier(data: bytes) -> ULID:
    return ulid.from_bytes(data)

def _sanitize_limit_and_offset(
    limit: Optional[int] = None, 
    offset: Optional[int] = None
) -> Tuple[int, int]:
    if limit is not None and limit < 0:
        raise InvalidNumberInput('limit must be a positive integer')
    if offset is not None and offset < 0:
        raise InvalidNumberInput('offset must be a positive integer')
    limit = -1 if limit is None else limit
    offset = 0 if offset is None else offset
    return (limit, offset)

def _filter_from_dict(current: Dict[str, Any]) -> Dict[str, Any]:
    """Takes in a nested dictionary as a filter and returns a flattened filter dictionary"""
    filter_: Dict[str, Any] = {}
    for k, v in current.items():
        if isinstance(v, dict):
            for sub, v2 in _filter_from_dict(v).items():
                filter_[f'{k}.{sub}'] = v2
        else:
            filter_[k] = v
    return filter_

def _query_to_string(query: FilteredDBQuery) -> Tuple[str, List[Any]]:
    query_where: List[str] = []
    args: List[Any] = []
    for filter_set in query.filters:
        where_clauses: List[str] = []
        filters = _filter_from_dict(filter_set)
        for field, value in filters.items():
            where_clauses.append('json_extract(data, ?)=?')
            args.append(f'$.{field}')
            args.append(value)
        filter_set_str = f' {query.inner_operator.value} '.join(where_clauses)
        query_where.append(f'({filter_set_str}) ')
    query_where_str = f' {query.main_operator.value} '.join(query_where)
    return (query_where_str, args)

def _prepend_and_save_ids(
    ulid_factory: ulid.api.api.Api, 
    ids: List[ID], 
    items: Iterable[Tuple[Any, ...]]
) -> Generator[Tuple[ID, ...], None, None]:
    for item in items:
        next_id = cast(ID, ulid_factory.new())
        ids.append(next_id)
        yield (next_id, *item)

def write_state_change(
    ulid_factory: ulid.api.api.Api, 
    cursor: sqlite3.Cursor, 
    state_change: str
) -> StateChangeID:
    """Write `state_change` to the database and returns the corresponding ID."""
    query = 'INSERT INTO state_changes(identifier, data) VALUES(?, ?)'
    new_id = StateChangeID(ulid_factory.new())
    cursor.execute(query, (new_id, state_change))
    return new_id

def write_events(
    ulid_factory: ulid.api.api.Api, 
    cursor: sqlite3.Cursor, 
    events: Iterable[Tuple[StateChangeID, str]]
) -> List[EventID]:
    events_ids: List[EventID] = []
    query = 'INSERT INTO state_events(identifier, source_statechange_id, data) VALUES(?, ?, ?)'
    cursor.executemany(query, _prepend_and_save_ids(ulid_factory, events_ids, events))
    return events_ids

class SQLiteStorage:
    def __init__(self, database_path: DatabasePath) -> None:
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
        self._ulid_factories: Dict[str, ulid.api.api.Api] = {}

    def _ulid_factory(self, id_type: Type[ID]) -> ulid.api.api.Api:
        expected_types = {
            StateChangeID: 'state_changes', 
            EventID: 'state_events', 
            SnapshotID: 'state_snapshot'
        }
        table_name = expected_types.get(id_type)
        if not table_name:
            raise ValueError(f'Unexpected ID type {id_type}')
        factory = self._ulid_factories.get(table_name)
        if factory is None:
            cursor = self.conn.cursor()
            query_table_exists = cursor.execute('SELECT name FROM sqlite_master WHERE name=?', (table_name,))
            assert query_table_exists.fetchone(), f'The table {table_name} does not exist.'
            query_last_id = cursor.execute(f'SELECT identifier FROM {table_name} ORDER BY identifier DESC LIMIT 1')
            result = query_last_id.fetchone()
            provider = ulid.providers.monotonic.Provider(ulid.providers.default.Provider())
            if result:
                timestamp = result[0].timestamp()
                provider.prev_timestamp = ulid.codec.decode_timestamp(timestamp)
            factory = ulid.api.api.Api(provider)
            self._ulid_factories[table_name] = factory
        return factory

    def update_version(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO settings(name, value) VALUES("version", ?)', (str(RAIDEN_DB_VERSION),))
        self.maybe_commit()

    def log_run(self) -> None:
        version = get_system_spec()['raiden']
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO runs(raiden_version) VALUES (?)', [version])
        self.maybe_commit()

    def get_version(self) -> RaidenDBVersion:
        cursor = self.conn.cursor()
        query = cursor.execute('SELECT value FROM settings WHERE name="version";')
        result = query.fetchall()
        if len(result) == 0:
            return RAIDEN_DB_VERSION
        return RaidenDBVersion(int(result[0][0]))

    def count_state_changes(self) -> int:
        cursor = self.conn.cursor()
        query = cursor.execute('SELECT COUNT(1) FROM state_changes')
        result = query.fetchall()
        if len(result) == 0:
            return 0
        return int(result[0][0])

    def has_snapshot(self) -> bool:
        cursor = self.conn.cursor()
        query = cursor.execute('SELECT EXISTS(SELECT 1 FROM state_snapshot)')
        result = query.fetchone()
        return bool(result[0])

    def write_state_changes(self, state_changes: List[str]) -> List[StateChangeID]:
        ulid_factory = self._ulid_factory(StateChangeID)
        state_change_data: List[Tuple[ULID, str]] = []
        state_change_ids: List[StateChangeID] = []
        for state_change in state_changes:
            new_id = ulid_factory.new()
            state_change_ids.append(StateChangeID(new_id))
            state_change_data.append((new_id, state_change))
        query = 'INSERT INTO state_changes(identifier, data) VALUES(?, ?)'
        self.conn.executemany(query, state_change_data)
        self.maybe_commit()
        return state_change_ids

    def write_first_state_snapshot(self, snapshot: str) -> SnapshotID:
        if self.has_snapshot():
            raise RuntimeError('write_first_state_snapshot can only be used for an unitialized node.')
        snapshot_id = SnapshotID(self._ulid_factory(SnapshotID).new())
        query = 'INSERT INTO state_snapshot (identifier, statechange_id, statechange_qty, data) VALUES(?, NULL, 0, ?)'
        self.conn.execute(query, (snapshot_id, snapshot))
        self.maybe_commit()
        return snapshot_id

    def write_state_snapshot(
        self, 
        snapshot: str, 
        statechange_id: StateChangeID, 
        statechange_qty: int
    ) -> SnapshotID:
        snapshot_id = SnapshotID(self._ulid_factory(SnapshotID).new())
        query = 'INSERT INTO state_snapshot (identifier, statechange_id, statechange_qty, data) VALUES(?, ?, ?, ?)'
        self.conn.execute(query, (snapshot_id, statechange_id, statechange_qty, snapshot))
        self.maybe_commit()
        return snapshot_id

    def write_events(self, events: Iterable[Tuple[StateChangeID, str]]) -> List[EventID]:
        ulid_factory = self._ulid_factory(EventID)
        events_ids: List[EventID] = []
        query = 'INSERT INTO state_events(identifier, source_statechange_id, data) VALUES(?, ?, ?)'
        self.conn.executemany(query, _prepend_and_save_ids(ulid_factory, events_ids, events))
        self.maybe_commit()
        return events_ids

    def delete_state_changes(self, state_changes_to_delete: List[StateChangeID]) -> None:
        self.conn.executemany('DELETE FROM state_changes WHERE identifier = ?', [(id,) for id in state_changes_to_delete])
        self.maybe_commit()

    def get_snapshot_before_state_change(
        self, 
        state_change_identifier: StateChangeID
    ) -> Optional[SnapshotEncodedRecord]:
        if not isinstance(state_change_identifier, ULID):
            raise ValueError('from_identifier must be an ULID')
        cursor = self.conn.execute(
            'SELECT identifier, statechange_qty, statechange_id, data FROM state_snapshot '
            'WHERE statechange_id <= ? OR statechange_id IS NULL '
            'ORDER BY identifier DESC LIMIT 1',
            (state_change_identifier,)
        )
        rows = cursor.fetchall()
        result = None
        if rows:
            assert len(rows) == 1, 'LIMIT 1 must return one element'
            identifier = rows[0][0]
            statechange_qty = rows[0][1]
            last_applied_state_change_id = rows[0][2]
            snapshot_state = rows[0][3]
            result = SnapshotEncodedRecord(
                identifier, 
                statechange_qty, 
                last_applied_state_change_id, 
                snapshot_state
            )
        return result

    def get_latest_event_by_data_field(
        self, 
        query: FilteredDBQuery
    ) -> Optional[EventEncodedRecord]:
        cursor = self.conn.cursor()
        query_str, args = _query_to_string(query)
        cursor.execute(
            f'SELECT identifier, source_statechange_id, data FROM state_events '
            f'WHERE {query_str}ORDER BY identifier DESC LIMIT 1',
            args
        )
        result = None
        row = cursor.fetchone()
        if row:
            event_id = row[0]
            state_change_identifier = row[1]
            event = row[2]
            result = EventEncodedRecord(
                event_identifier=event_id, 
                state_change_identifier=state_change_identifier, 
                data=event
            )
        return result

    def _form_and_execute_json_query(
        self, 
        query: str, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None, 
        filters: Optional[List[Tuple[str, str]]] = None, 
        logical_and: bool = True
    ) -> sqlite3.Cursor:
        limit, offset = _sanitize_limit_and_offset(limit, offset)
        cursor = self.conn.cursor()
        where_clauses: List[str] = []
        args: List[Any] = []
        if filters:
            for field, value in filters:
                where_clauses.append('json_extract(data, ?) LIKE ?')
                args.append(f'$.{field}')
                args.append(value)
            if logical_and:
                query += f'WHERE {" AND ".join(where_clauses)}'
            else:
                query += f'WHERE {" OR ".join(where_clauses)}'
        query += 'ORDER BY identifier ASC LIMIT ? OFFSET ?'
        args.append(limit)
        args.append(offset)
        cursor.execute(query, args)
        return cursor

    def get_latest_state_change_by_data_field(
        self, 
        query: FilteredDBQuery
    ) -> Optional[StateChangeEncodedRecord]:
        cursor = self.conn.cursor()
        query_str, args = _query_to_string(query)
        sql = (
            f'SELECT identifier, data FROM state_changes '
            f'WHERE {query_str} ORDER BY identifier DESC LIMIT 1'
        )
        cursor.execute(sql, args)
        result = None
        row = cursor.fetchone()
        if row:
            state_change_identifier = row[0]
            state_change = row[1]
            result = StateChangeEncodedRecord(
                state_change_identifier=state_change_identifier, 
                data=state_change
            )
        return result

    def _get_state_changes(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None, 
        filters: Optional[List[Tuple[str, str]]] = None, 
        logical_and: bool = True
    ) -> List[StateChangeEncodedRecord]:
        cursor = self._form_and_execute_json_query(
            query='SELECT identifier, data FROM state_changes ',
            limit=limit,
            offset=offset,
            filters=filters,
            logical_and=logical_and
        )
        return [
            StateChangeEncodedRecord(state_change_identifier=row[0], data=row[1]) 
            for row in cursor
        ]

    def batch_query_state_changes(
        self, 
        batch_size: int, 
        filters: Optional[List[Tuple[str, str]]] = None, 
        logical_and: bool = True
    ) -> Generator[List[StateChangeEncodedRecord], None, None]:
        limit = batch_size
        offset = 0
        result_length = 1
        while result_length != 0:
            result = self._get_state_changes(
                limit=limit,
                offset=offset,
                filters=filters,
                logical_and=logical_and
            )
            result_length = len(result)
            offset += result_length
            yield result

    def update_state_changes(
        self, 
        state_changes_data: List[Tuple[str, StateChangeID]]
    ) -> None:
        cursor = self.conn.cursor()
        cursor.executemany(
            'UPDATE state_changes SET data=? WHERE identifier=?', 
            state_ch