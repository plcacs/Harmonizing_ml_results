"""In-memory table storage."""
from typing import Any, Callable, Iterable, MutableMapping, Optional, Set, Tuple
from faust.types import EventT, TP
from . import base

class Store(base.Store):
    """Table storage using an in-memory dictionary."""

    def __post_init__(self) -> None:
        self.data: MutableMapping[Any, Any] = {}

    def _clear(self) -> None:
        self.data.clear()

    def apply_changelog_batch(
        self,
        batch: Iterable[EventT],
        to_key: Callable[[EventT], Any],
        to_value: Callable[[EventT], Any],
    ) -> None:
        """Apply batch of changelog events to in-memory table."""
        to_delete: Set[Any] = set()
        delete_key: Callable[[Any, Optional[Any]], None] = self.data.pop
        self.data.update(
            self._create_batch_iterator(
                mark_as_delete=to_delete.add,
                to_key=to_key,
                to_value=to_value,
                batch=batch,
            )
        )
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(
        self,
        mark_as_delete: Callable[[Any], None],
        to_key: Callable[[EventT], Any],
        to_value: Callable[[EventT], Any],
        batch: Iterable[EventT],
    ) -> Iterable[Tuple[Any, Any]]:
        for event in batch:
            key = to_key(event)
            if event.message.value is None:
                mark_as_delete(key)
                continue
            yield (key, to_value(event.value))

    def persisted_offset(self, tp: TP) -> Optional[Any]:
        """Return the persisted offset.

        This always returns :const:`None` when using the in-memory store.
        """
        return None

    def reset_state(self) -> None:
        """Remove local file system state.

        This does nothing when using the in-memory store.

        """
        ...
