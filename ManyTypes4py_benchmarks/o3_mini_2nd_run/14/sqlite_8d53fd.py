#!/usr/bin/env python3
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
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
from raiden.utils.typing import Address, DatabasePath, RaidenDBVersion

StateChangeID = NewType('StateChangeID', ULID)
SnapshotID = NewType('SnapshotID', ULID)
EventID = NewType('EventID', ULID)
ID = TypeVar('ID', StateChangeID, SnapshotID, EventID)

@dataclass
class Range:
    """Inclusive range used to filter database entries."""
    first: ID
    last: ID

    def __post_init__(self) -> None:
        if self.first > self.last:
            raise ValueError('last must be larger or equal to first')

LOW_STATECHANGE_ULID: StateChangeID = StateChangeID(MIN_ULID)
HIGH_STATECHANGE_ULID: StateChangeID = StateChangeID(MAX_ULID)
RANGE_ALL_STATE_CHANGES: Range = Range(LOW_STATECHANGE_ULID, HIGH_STATECHANGE_ULID)

class Operator(Enum):
    NONE = ''
    AND = 'AND'
    OR = 'OR'

class FilteredDBQuery(NamedTuple):
    """
    FilteredDBQuery is a datastructure that helps
    form a list of conditions and how they're grouped
    in order to form more complicated queries
    on the internal JSON representation of states / state changes and events.
    Note that it is not used to search the top-level attributes of the sqlite tables.
    """
    filters: List[Dict[str, Any]]
    main_operator: Operator
    inner_operator: Operator

class EventEncodedRecord(NamedTuple):
    event_identifier: ULID
    state_change_identifier: ULID
    data: Any

class StateChangeEncodedRecord(NamedTuple):
    state_change_identifier: ULID
    data: Any

class SnapshotEncodedRecord(NamedTuple):
    identifier: ULID
    state_change_qty: int
    state_change_identifier: Optional[ULID]
    data: Any

class EventRecord(NamedTuple):
    event_identifier: ULID
    state_change_identifier: ULID
    data: Any

class StateChangeRecord(NamedTuple):
    state_change_identifier: ULID
    data: Any

class SnapshotRecord(NamedTuple):
    identifier: ULID
    state_change_qty: int
    state_change_identifier: ULID
    data: Any

def assert_sqlite_version() -> bool:
    if sqlite3.sqlite_version_info < SQLITE_MIN_REQUIRED_VERSION:
        return False
    return True

def adapt_ulid_identifier(ulid_obj: ULID) -> bytes:
    return ulid_obj.bytes

def convert_ulid_identifier(data: bytes) -> ULID:
    return ulid.from_bytes(data)

def _sanitize_limit_and_offset(limit: Optional[int] = None, offset: Optional[int] = None) -> Tuple[int, int]:
    if limit is not None and limit < 0:
        raise InvalidNumberInput('limit must be a positive integer')
    if offset is not None and offset < 0:
        raise InvalidNumberInput('offset must be a positive integer')
    limit_val: int = -1 if limit is None else limit
    offset_val: int = 0 if offset is None else offset
    return (limit_val, offset_val)

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
    """
    Converts a query object to a valid SQL string which can be used in the WHERE clause.
    """
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
    ulid_factory: Any, ids: List[ID], items: Iterable[Tuple[Any, ...]]
) -> Generator[Tuple[ID, Any], None, None]:
    for item in items:
        next_id: ID = cast(ID, ulid_factory.new())
        ids.append(next_id)
        yield (next_id, *item)

def write_state_change(ulid_factory: Any, cursor: sqlite3.Cursor, state_change: Any) -> StateChangeID:
    """Write `state_change` to the database and returns the corresponding ID."""
    query: str = 'INSERT INTO state_changes(identifier, data) VALUES(?, ?)'
    new_id: StateChangeID = StateChangeID(ulid_factory.new())
    cursor.execute(query, (new_id, state_change))
    return new_id

def write_events(ulid_factory: Any, cursor: sqlite3.Cursor, events: Iterable[Tuple[Any, ...]]) -> List[EventID]:
    events_ids: List[EventID] = []
    query: str = 'INSERT INTO state_events(identifier, source_statechange_id, data) VALUES(?, ?, ?)'
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
        self.conn: sqlite3.Connection = conn
        self.in_transaction: bool = False
        self._ulid_factories: Dict[str, Any] = {}

    def _ulid_factory(self, id_type: Type[ID]) -> Any:
        """Return an ULID Factory for a specific table."""
        expected_types: Dict[Type[ID], str] = {
            StateChangeID: 'state_changes',
            EventID: 'state_events',
            SnapshotID: 'state_snapshot',
        }
        table_name: Optional[str] = expected_types.get(id_type)
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
        """Log timestamp and raiden version to help with debugging"""
        version: str = get_system_spec()['raiden']
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

    def write_state_changes(self, state_changes: Iterable[Any]) -> List[StateChangeID]:
        """Write `state_changes` to the database and returns the corresponding IDs."""
        ulid_factory = self._ulid_factory(StateChangeID)
        state_change_data: List[Tuple[Any, Any]] = []
        state_change_ids: List[StateChangeID] = []
        for state_change in state_changes:
            new_id: ULID = ulid_factory.new()
            state_change_ids.append(StateChangeID(new_id))
            state_change_data.append((new_id, state_change))
        query = 'INSERT INTO state_changes(identifier, data) VALUES(?, ?)'
        self.conn.executemany(query, state_change_data)
        self.maybe_commit()
        return state_change_ids

    def write_first_state_snapshot(self, snapshot: Any) -> SnapshotID:
        if self.has_snapshot():
            raise RuntimeError('write_first_state_snapshot can only be used for an unitialized node.')
        snapshot_id: SnapshotID = SnapshotID(self._ulid_factory(SnapshotID).new())
        query = 'INSERT INTO state_snapshot (identifier, statechange_id, statechange_qty, data) VALUES(?, NULL, 0, ?)'
        self.conn.execute(query, (snapshot_id, snapshot))
        self.maybe_commit()
        return snapshot_id

    def write_state_snapshot(self, snapshot: Any, statechange_id: ULID, statechange_qty: int) -> SnapshotID:
        snapshot_id: SnapshotID = SnapshotID(self._ulid_factory(SnapshotID).new())
        query = 'INSERT INTO state_snapshot (identifier, statechange_id, statechange_qty, data) VALUES(?, ?, ?, ?)'
        self.conn.execute(query, (snapshot_id, statechange_id, statechange_qty, snapshot))
        self.maybe_commit()
        return snapshot_id

    def write_events(self, events: Iterable[Tuple[Any, Any]]) -> List[EventID]:
        ulid_factory = self._ulid_factory(EventID)
        events_ids: List[EventID] = []
        query = 'INSERT INTO state_events(identifier, source_statechange_id, data) VALUES(?, ?, ?)'
        self.conn.executemany(query, _prepend_and_save_ids(ulid_factory, events_ids, events))
        self.maybe_commit()
        return events_ids

    def delete_state_changes(self, state_changes_to_delete: Iterable[Tuple[Any]]) -> None:
        self.conn.executemany('DELETE FROM state_changes WHERE identifier = ?', state_changes_to_delete)
        self.maybe_commit()

    def get_snapshot_before_state_change(self, state_change_identifier: ULID) -> Optional[SnapshotEncodedRecord]:
        if not isinstance(state_change_identifier, ULID):
            raise ValueError('from_identifier must be an ULID')
        cursor = self.conn.execute(
            'SELECT identifier, statechange_qty, statechange_id, data FROM state_snapshot '
            'WHERE statechange_id <= ? OR statechange_id IS NULL ORDER BY identifier DESC LIMIT 1',
            (state_change_identifier,)
        )
        rows = cursor.fetchall()
        result: Optional[SnapshotEncodedRecord] = None
        if rows:
            assert len(rows) == 1, 'LIMIT 1 must return one element'
            identifier = rows[0][0]
            statechange_qty = rows[0][1]
            last_applied_state_change_id = rows[0][2]
            snapshot_state = rows[0][3]
            result = SnapshotEncodedRecord(identifier, statechange_qty, last_applied_state_change_id, snapshot_state)
        return result

    def get_latest_event_by_data_field(self, query: FilteredDBQuery) -> Optional[EventEncodedRecord]:
        cursor = self.conn.cursor()
        query_str, args = _query_to_string(query)
        cursor.execute(
            f'SELECT identifier, source_statechange_id, data FROM state_events WHERE {query_str}ORDER BY identifier DESC LIMIT 1',
            args
        )
        result: Optional[EventEncodedRecord] = None
        row = cursor.fetchone()
        if row:
            event_id = row[0]
            state_change_identifier = row[1]
            event = row[2]
            result = EventEncodedRecord(event_identifier=event_id, state_change_identifier=state_change_identifier, data=event)
        return result

    def _form_and_execute_json_query(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> sqlite3.Cursor:
        limit_val, offset_val = _sanitize_limit_and_offset(limit, offset)
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
        args.append(limit_val)
        args.append(offset_val)
        cursor.execute(query, args)
        return cursor

    def get_latest_state_change_by_data_field(self, query: FilteredDBQuery) -> Optional[StateChangeEncodedRecord]:
        cursor = self.conn.cursor()
        query_str, args = _query_to_string(query)
        sql = f'SELECT identifier, data FROM state_changes WHERE {query_str} ORDER BY identifier DESC LIMIT 1'
        cursor.execute(sql, args)
        result: Optional[StateChangeEncodedRecord] = None
        row = cursor.fetchone()
        if row:
            state_change_identifier = row[0]
            state_change = row[1]
            result = StateChangeEncodedRecord(state_change_identifier=state_change_identifier, data=state_change)
        return result

    def _get_state_changes(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> List[StateChangeEncodedRecord]:
        cursor = self._form_and_execute_json_query(
            query='SELECT identifier, data FROM state_changes ',
            limit=limit,
            offset=offset,
            filters=filters,
            logical_and=logical_and
        )
        result = [StateChangeEncodedRecord(state_change_identifier=row[0], data=row[1]) for row in cursor]
        return result

    def batch_query_state_changes(
        self,
        batch_size: int,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> Generator[List[StateChangeEncodedRecord], None, None]:
        limit = batch_size
        offset = 0
        result_length = 1
        while result_length != 0:
            result = self._get_state_changes(limit=limit, offset=offset, filters=filters, logical_and=logical_and)
            result_length = len(result)
            offset += result_length
            yield result

    def update_state_changes(self, state_changes_data: Iterable[Tuple[Any, Any]]) -> None:
        cursor = self.conn.cursor()
        cursor.executemany('UPDATE state_changes SET data=? WHERE identifier=?', state_changes_data)
        self.maybe_commit()

    def get_statechanges_records_by_range(self, db_range: Range) -> List[StateChangeEncodedRecord]:
        if not isinstance(db_range, Range):
            raise ValueError('db_range must be an Range')
        cursor = self.conn.cursor()
        query = 'SELECT identifier, data FROM state_changes WHERE identifier BETWEEN ? AND ? ORDER BY identifier ASC'
        cursor.execute(query, (db_range.first, db_range.last))
        return [StateChangeEncodedRecord(state_change_identifier=entry[0], data=entry[1]) for entry in cursor]

    def _query_events(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> List[Tuple[Any, Any]]:
        cursor = self._form_and_execute_json_query(
            query='SELECT data, timestamp FROM state_events ',
            limit=limit,
            offset=offset,
            filters=filters,
            logical_and=logical_and
        )
        return cursor.fetchall()

    def _get_event_records(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> List[EventEncodedRecord]:
        cursor = self._form_and_execute_json_query(
            query='SELECT identifier, source_statechange_id, data FROM state_events ',
            limit=limit,
            offset=offset,
            filters=filters,
            logical_and=logical_and
        )
        result = [EventEncodedRecord(event_identifier=row[0], state_change_identifier=row[1], data=row[2]) for row in cursor]
        return result

    def batch_query_event_records(
        self,
        batch_size: int,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> Generator[List[EventEncodedRecord], None, None]:
        limit = batch_size
        offset = 0
        result_length = 1
        while result_length != 0:
            result = self._get_event_records(limit=limit, offset=offset, filters=filters, logical_and=logical_and)
            result_length = len(result)
            offset += result_length
            yield result

    def update_events(self, events_data: Iterable[Tuple[Any, Any]]) -> None:
        cursor = self.conn.cursor()
        cursor.executemany('UPDATE state_events SET data=? WHERE identifier=?', events_data)
        self.maybe_commit()

    def get_raiden_events_payment_history_with_timestamps(
        self,
        event_types: List[str],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        token_network_address: Optional[Address] = None,
        partner_address: Optional[Address] = None
    ) -> List[TimestampedEvent]:
        limit_val, offset_val = _sanitize_limit_and_offset(limit, offset)
        cursor = self.conn.cursor()
        args: List[Any] = list(event_types)
        sql_helper: str = ','.join('?' * len(event_types))
        if token_network_address and partner_address:
            query = (
                "\n                SELECT\n                    data, timestamp\n                FROM\n                    state_events\n                WHERE\n                    json_extract(data, '$._type') IN ({})\n                AND\n                    json_extract(data, '$.token_network_address') LIKE ?\n                AND\n                    (\n                    json_extract(data, '$.target') LIKE ?\n                    OR\n                    json_extract(data, '$.initiator') LIKE ?\n                    )\n                ORDER BY identifier\n                ASC LIMIT ? OFFSET ?\n                "
            )
            args.append(to_normalized_address(token_network_address))
            args.append(to_normalized_address(partner_address))
            args.append(to_normalized_address(partner_address))
        elif token_network_address and (not partner_address):
            query = (
                "\n                SELECT\n                    data, timestamp\n                FROM\n                    state_events\n                WHERE\n                    json_extract(data, '$._type') IN ({})\n                AND\n                    json_extract(data, '$.token_network_address') LIKE ?\n                ORDER BY identifier\n                ASC LIMIT ? OFFSET ?\n                "
            )
            args.append(to_normalized_address(token_network_address))
        elif partner_address and (not token_network_address):
            query = (
                "\n                SELECT\n                    data, timestamp\n                FROM\n                    state_events\n                WHERE\n                    json_extract(data, '$._type') IN ({})\n                AND\n                    (\n                    json_extract(data, '$.target') LIKE ?\n                    OR\n                    json_extract(data, '$.initiator') LIKE ?\n                    )\n                ORDER BY identifier\n                ASC LIMIT ? OFFSET ?\n                "
            )
            args.append(to_normalized_address(partner_address))
            args.append(to_normalized_address(partner_address))
        else:
            query = (
                "\n                SELECT\n                    data, timestamp\n                FROM\n                    state_events\n                WHERE\n                    json_extract(data, '$._type') IN ({})\n                ORDER BY identifier\n                ASC LIMIT ? OFFSET ?\n                "
            )
        query = query.format(sql_helper)
        args.append(limit_val)
        args.append(offset_val)
        cursor.execute(query, args)
        return [(entry[0], entry[1]) for entry in cursor]

    def get_events_with_timestamps(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> List[Tuple[Any, Any]]:
        entries = self._query_events(limit=limit, offset=offset, filters=filters, logical_and=logical_and)
        return [(entry[0], entry[1]) for entry in entries]

    def get_events(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
        entries = self._query_events(limit, offset)
        return [entry[0] for entry in entries]

    def get_state_changes(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
        entries = self._get_state_changes(limit, offset)
        return [entry.data for entry in entries]

    def get_snapshots(self) -> List[SnapshotEncodedRecord]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT identifier, statechange_qty, statechange_id, data FROM state_snapshot')
        return [SnapshotEncodedRecord(snapshot[0], snapshot[1], snapshot[2], snapshot[3]) for snapshot in cursor]

    def update_snapshot(self, identifier: ULID, new_snapshot: Any) -> None:
        cursor = self.conn.cursor()
        cursor.execute('UPDATE state_snapshot SET data=? WHERE identifier=?', (new_snapshot, identifier))
        self.maybe_commit()

    def update_snapshots(self, snapshots_data: Iterable[Tuple[Any, Any]]) -> None:
        """Given a list of snapshot data, update them in the DB.
        The snapshots_data should be a list of tuples of snapshots data and identifiers in that order.
        """
        cursor = self.conn.cursor()
        cursor.executemany('UPDATE state_snapshot SET data=? WHERE identifier=?', snapshots_data)
        self.maybe_commit()

    def maybe_commit(self) -> None:
        if not self.in_transaction:
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        cursor = self.conn.cursor()
        self.in_transaction = True
        try:
            cursor.execute('BEGIN')
            yield
            cursor.execute('COMMIT')
        except:
            cursor.execute('ROLLBACK')
            raise
        finally:
            self.in_transaction = False

    def close(self) -> None:
        if not hasattr(self, 'conn'):
            raise RuntimeError('The database connection was closed already.')
        self.conn.close()
        del self.conn

    def __enter__(self) -> "SQLiteStorage":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        self.close()

class SerializedSQLiteStorage:
    """A wrapper around SQLiteStorage that automatically serializes and deserializes the data.
    SQLiteStorage is necessary for database upgrades. Upgrades are necessary when the data model changes,
    and as a consequence before the upgrades are applied the automatic encoding/decoding will not work.
    """
    def __init__(self, database_path: DatabasePath, serializer: SerializationBase) -> None:
        self.database: SQLiteStorage = SQLiteStorage(database_path)
        self.serializer: SerializationBase = serializer

    def update_version(self) -> None:
        self.database.update_version()

    def count_state_changes(self) -> int:
        return self.database.count_state_changes()

    def get_version(self) -> RaidenDBVersion:
        return self.database.get_version()

    def log_run(self) -> None:
        self.database.log_run()

    def write_state_changes(self, state_changes: Iterable[Any]) -> List[StateChangeID]:
        serialized_data = [self.serializer.serialize(state_change) for state_change in state_changes]
        return self.database.write_state_changes(serialized_data)

    def write_first_state_snapshot(self, snapshot: Any) -> SnapshotID:
        serialized_data = self.serializer.serialize(snapshot)
        return self.database.write_first_state_snapshot(serialized_data)

    def write_state_snapshot(self, snapshot: Any, statechange_id: ULID, statechange_qty: int) -> SnapshotID:
        serialized_data = self.serializer.serialize(snapshot)
        return self.database.write_state_snapshot(serialized_data, statechange_id, statechange_qty)

    def write_events(self, events: List[Tuple[ULID, Any]]) -> List[EventID]:
        """Save events.
        Args:
            events: List of tuples where the first element is state_change_id and the second is an Event.
        """
        events_data = [(state_change_id, self.serializer.serialize(event)) for state_change_id, event in events]
        return self.database.write_events(events_data)

    def get_snapshot_before_state_change(self, state_change_identifier: ULID) -> Optional[SnapshotRecord]:
        row = self.database.get_snapshot_before_state_change(state_change_identifier)
        if row is not None:
            deserialized_data = self.serializer.deserialize(row.data)
            result = SnapshotRecord(row.identifier, row.state_change_qty, row.state_change_identifier or LOW_STATECHANGE_ULID, deserialized_data)
        else:
            result = None
        return result

    def get_latest_event_by_data_field(self, query: FilteredDBQuery) -> Optional[EventRecord]:
        encoded_event = self.database.get_latest_event_by_data_field(query)
        event: Optional[EventRecord] = None
        if encoded_event is not None:
            event = EventRecord(
                event_identifier=encoded_event.event_identifier,
                state_change_identifier=encoded_event.state_change_identifier,
                data=self.serializer.deserialize(encoded_event.data)
            )
        return event

    def get_latest_state_change_by_data_field(self, query: FilteredDBQuery) -> Optional[StateChangeRecord]:
        encoded_state_change = self.database.get_latest_state_change_by_data_field(query)
        state_change: Optional[StateChangeRecord] = None
        if encoded_state_change is not None:
            state_change = StateChangeRecord(
                state_change_identifier=encoded_state_change.state_change_identifier,
                data=self.serializer.deserialize(encoded_state_change.data)
            )
        return state_change

    def get_statechanges_records_by_range(self, db_range: Range) -> List[StateChangeRecord]:
        state_changes = self.database.get_statechanges_records_by_range(db_range=db_range)
        return [StateChangeRecord(state_change_identifier=state_change.state_change_identifier, data=self.serializer.deserialize(state_change.data)) for state_change in state_changes]

    def get_statechanges_by_range(self, db_range: Range) -> List[Any]:
        return [state_change_record.data for state_change_record in self.get_statechanges_records_by_range(db_range=db_range)]

    def get_raiden_events_payment_history_with_timestamps(
        self,
        event_types: List[str],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        token_network_address: Optional[Address] = None,
        partner_address: Optional[Address] = None
    ) -> List[TimestampedEvent]:
        events = self.database.get_raiden_events_payment_history_with_timestamps(
            event_types=event_types,
            limit=limit,
            offset=offset,
            token_network_address=token_network_address,
            partner_address=partner_address
        )
        return [TimestampedEvent(self.serializer.deserialize(data), timestamp) for data, timestamp in events]

    def get_events_with_timestamps(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[List[Tuple[str, Any]]] = None,
        logical_and: bool = True
    ) -> List[TimestampedEvent]:
        events = self.database.get_events_with_timestamps(limit=limit, offset=offset, filters=filters, logical_and=logical_and)
        return [TimestampedEvent(self.serializer.deserialize(data), timestamp) for data, timestamp in events]

    def get_events(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
        events = self.database.get_events(limit, offset)
        return [self.serializer.deserialize(event) for event in events]

    def get_state_changes_stream(
        self,
        retry_timeout: float,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> Generator[List[Any], None, None]:
        while True:
            state_changes = self.database.get_state_changes(limit, offset)
            yield [self.serializer.deserialize(state_change) for state_change in state_changes]
            offset += len(state_changes)
            gevent.sleep(retry_timeout)

    def close(self) -> None:
        self.database.close()