from typing import Any, Callable, Generator, Iterable, MutableMapping, Optional, Set, Tuple
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
        batch: Iterable[Any],
        to_key: Callable[[Any], Any],
        to_value: Callable[[Any], Any],
    ) -> None:
        """Apply batch of changelog events to in-memory table."""
        to_delete: Set[Any] = set()
        delete_key = self.data.pop
        self.data.update(
            self._create_batch_iterator(to_delete.add, to_key, to_value, batch)
        )
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(
        self,
        mark_as_delete: Callable[[Any], None],
        to_key: Callable[[Any], Any],
        to_value: Callable[[Any], Any],
        batch: Iterable[Any],
    ) -> Generator[Tuple[Any, Any], None, None]:
        for event in batch:
            key = to_key(event.key)
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