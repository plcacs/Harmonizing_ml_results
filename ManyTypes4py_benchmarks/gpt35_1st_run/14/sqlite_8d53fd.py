from typing import List, Tuple, Optional

class SQLiteStorage:
    def __init__(self, database_path: str):
    def _ulid_factory(self, id_type: Type[ID]) -> ulid.api.api.Api:
    def update_version(self) -> None:
    def log_run(self) -> None:
    def get_version(self) -> RaidenDBVersion:
    def count_state_changes(self) -> int:
    def has_snapshot(self) -> bool:
    def write_state_changes(self, state_changes: List[Any]) -> List[StateChangeID]:
    def write_first_state_snapshot(self, snapshot: Any) -> SnapshotID:
    def write_state_snapshot(self, snapshot: Any, statechange_id: StateChangeID, statechange_qty: int) -> SnapshotID:
    def write_events(self, events: List[Tuple[StateChangeID, Any]]) -> List[EventID]:
    def delete_state_changes(self, state_changes_to_delete: List[StateChangeID]) -> None:
    def get_snapshot_before_state_change(self, state_change_identifier: ULID) -> Optional[SnapshotEncodedRecord]:
    def get_latest_event_by_data_field(self, query: FilteredDBQuery) -> Optional[EventEncodedRecord]:
    def batch_query_state_changes(self, batch_size: int, filters: Optional[List[Tuple[str, Any]]] = None, logical_and: bool = True) -> Generator[List[StateChangeEncodedRecord], None, None]:
    def update_state_changes(self, state_changes_data: List[Tuple[Any, StateChangeID]]) -> None:
    def get_statechanges_records_by_range(self, db_range: Range[ID]) -> List[StateChangeEncodedRecord]:
    def batch_query_event_records(self, batch_size: int, filters: Optional[List[Tuple[str, Any]]] = None, logical_and: bool = True) -> Generator[List[EventEncodedRecord], None, None]:
    def update_events(self, events_data: List[Tuple[Any, EventID]]) -> None:
    def get_raiden_events_payment_history_with_timestamps(self, event_types: List[str], limit: Optional[int] = None, offset: Optional[int] = None, token_network_address: Optional[Address] = None, partner_address: Optional[Address] = None) -> List[TimestampedEvent]:
    def get_events_with_timestamps(self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[List[Tuple[str, Any]]] = None, logical_and: bool = True) -> List[TimestampedEvent]:
    def get_state_changes(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
    def get_snapshots(self) -> List[SnapshotEncodedRecord]:
    def update_snapshot(self, identifier: SnapshotID, new_snapshot: Any) -> None:
    def update_snapshots(self, snapshots_data: List[Tuple[Any, SnapshotID]]) -> None
    def maybe_commit(self) -> None:
    def transaction(self) -> Generator[None, None, None]:
    def close(self) -> None

class SerializedSQLiteStorage:
    def __init__(self, database_path: str, serializer: SerializationBase):
    def update_version(self) -> None:
    def count_state_changes(self) -> int:
    def get_version(self) -> RaidenDBVersion:
    def log_run(self) -> None:
    def write_state_changes(self, state_changes: List[Any]) -> List[StateChangeID]:
    def write_first_state_snapshot(self, snapshot: Any) -> SnapshotID:
    def write_state_snapshot(self, snapshot: Any, statechange_id: StateChangeID, statechange_qty: int) -> SnapshotID:
    def write_events(self, events: List[Tuple[StateChangeID, Any]]) -> List[EventID]:
    def get_snapshot_before_state_change(self, state_change_identifier: ULID) -> Optional[SnapshotRecord]:
    def get_latest_event_by_data_field(self, query: FilteredDBQuery) -> Optional[EventRecord]:
    def get_latest_state_change_by_data_field(self, query: FilteredDBQuery) -> Optional[StateChangeRecord]:
    def get_statechanges_records_by_range(self, db_range: Range[ID]) -> List[StateChangeRecord]:
    def get_statechanges_by_range(self, db_range: Range[ID]) -> List[Any]:
    def get_raiden_events_payment_history_with_timestamps(self, event_types: List[str], limit: Optional[int] = None, offset: Optional[int] = None, token_network_address: Optional[Address] = None, partner_address: Optional[Address] = None) -> List[TimestampedEvent]:
    def get_events_with_timestamps(self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[List[Tuple[str, Any]]] = None, logical_and: bool = True) -> List[TimestampedEvent]:
    def get_events(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
    def get_state_changes_stream(self, retry_timeout: int, limit: Optional[int] = None, offset: int = 0) -> Generator[List[Any], None, None]:
    def close(self) -> None
