from typing import Any, Callable, Iterable, MutableMapping, Optional, Set, Tuple
from faust.types import EventT, TP
from . import base

class Store(base.Store):
    data: MutableMapping[Any, Any]

    def __post_init__(self):
        self.data: MutableMapping[Any, Any] = {}

    def _clear(self) -> None:
        self.data.clear()

    def apply_changelog_batch(self, batch: Iterable[EventT], to_key: Callable[[Any], Any], to_value: Callable[[Any], Any]) -> None:
        to_delete: Set[Any] = set()
        delete_key: Callable[[Any], Any] = self.data.pop
        self.data.update(self._create_batch_iterator(to_delete.add, to_key, to_value, batch))
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(self, mark_as_delete: Callable[[Any], Any], to_key: Callable[[Any], Any], to_value: Callable[[Any], Any], batch: Iterable[EventT]) -> Iterable[Tuple[Any, Any]]:
        for event in batch:
            key = to_key(event.key)
            if event.message.value is None:
                mark_as_delete(key)
                continue
            yield (key, to_value(event.value))

    def persisted_offset(self, tp: TP) -> Optional[Any]:
        return None

    def reset_state(self) -> None:
        ...
