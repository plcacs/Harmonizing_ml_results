from typing import Any, Dict, Set, Tuple

class ValueType:

    def __init__(self, man: 'ChangeloggedObjectManager', key: Any) -> None:
        self.man: 'ChangeloggedObjectManager' = man
        self.key: Any = key
        self.synced: Set[Any] = set()
        self.changes: List[Tuple[str, Any]] = []

    def sync_from_storage(self, value: Any) -> None:
        self.synced.add(value)

    def as_stored_value(self) -> str:
        return f'{self.key}-stored'

    def apply_changelog_event(self, operation: str, value: Any) -> None:
        self.changes.append((operation, value))

class ChangeloggedObjectManager:

    def __init__(self, table: Any) -> None:
        self.table: Any = table
        self.ValueType: Any = ValueType
        self._dirty: Set[Any] = set()
        self.data: Dict[str, ValueType] = {}
        self._storage: Dict[str, str] = {}

    def send_changelog_event(self, key: Any, num: int, value: Any) -> None:
        pass

    def __getitem__(self, key: str) -> ValueType:
        pass

    def __setitem__(self, key: str, value: Any) -> None:
        pass

    def __delitem__(self, key: str) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def persisted_offset(self, tp: Any) -> Any:
        pass

    def set_persisted_offset(self, tp: Any, offset: int) -> None:
        pass

    def on_rebalance(self, table: Any, assigned: Set[Any], revoked: Set[Any], newly_assigned: Set[Any]) -> None:
        pass

    def on_recovery_completed(self, assigned: Set[Any], revoked: Set[Any]) -> None:
        pass

    def sync_from_storage(self) -> None:
        pass

    def flush_to_storage(self) -> None:
        pass

    def reset_state(self) -> None:
        pass

    def apply_changelog_batch(self, events: List[Any], key_serializer: Any, value_serializer: Any) -> None:
        pass
