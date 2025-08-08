from typing import Any, Dict, Set, Tuple

class ValueType:
    man: ChangeloggedObjectManager
    key: Any
    synced: Set[Any]
    changes: List[Tuple[str, Any]]

    def sync_from_storage(self, value: Any) -> None:
        ...

    def as_stored_value(self) -> str:
        ...

    def apply_changelog_event(self, operation: str, value: Any) -> None:
        ...

class ChangeloggedObjectManager:
    _dirty: Set[Any]
    _table_type_name: str
    storage: Any

    def send_changelog_event(self, key: Any, partition: int, value: Any) -> None:
        ...

    def __getitem__(self, key: Any) -> ValueType:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def __delitem__(self, key: Any) -> None:
        ...

    def persisted_offset(self, tp: TP) -> Any:
        ...

    def set_persisted_offset(self, tp: TP, offset: int) -> None:
        ...

    async def on_rebalance(self, table: Any, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    async def on_recovery_completed(self, assigned: Set[TP], revoked: Set[TP]) -> None:
        ...

    def sync_from_storage(self) -> None:
        ...

    def flush_to_storage(self) -> None:
        ...

    def reset_state(self) -> None:
        ...

    def apply_changelog_batch(self, events: List[Any], key_serializer: Callable, value_serializer: Callable) -> None:
        ...
