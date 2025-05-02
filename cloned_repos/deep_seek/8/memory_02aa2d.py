"""In-memory table storage."""
from typing import Any, Callable, Dict, Iterable, Iterator, MutableMapping, Optional, Set, Tuple, TypeVar
from faust.types import EventT, TP
from . import base

KT = TypeVar('KT')
VT = TypeVar('VT')

class Store(base.Store):
    """Table storage using an in-memory dictionary."""

    def __post_init__(self) -> None:
        self.data: Dict[Any, Any] = {}

    def _clear(self) -> None:
        self.data.clear()

    def apply_changelog_batch(
        self,
        batch: Iterable[EventT],
        to_key: Callable[[Any], KT],
        to_value: Callable[[Any], VT],
    ) -> None:
        """Apply batch of changelog events to in-memory table."""
        to_delete: Set[KT] = set()
        delete_key = self.data.pop
        self.data.update(self._create_batch_iterator(to_delete.add, to_key, to_value, batch))
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(
        self,
        mark_as_delete: Callable[[KT], None],
        to_key: Callable[[Any], KT],
        to_value: Callable[[Any], VT],
        batch: Iterable[EventT],
    ) -> Iterator[Tuple[KT, VT]]:
        for event in batch:
            key = to_key(event.key)
            if event.message.value is None:
                mark_as_delete(key)
                continue
            yield (key, to_value(event.value))

    def persisted_offset(self, tp: TP) -> Optional[int]:
        """Return the persisted offset.

        This always returns :const:`None` when using the in-memory store.
        """
        return None

    def reset_state(self) -> None:
        """Remove local file system state.

        This does nothing when using the in-memory store.

        """
        ...
