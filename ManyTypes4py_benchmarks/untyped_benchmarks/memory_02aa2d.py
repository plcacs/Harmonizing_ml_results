"""In-memory table storage."""
from typing import Any, Callable, Iterable, MutableMapping, Optional, Set, Tuple
from faust.types import EventT, TP
from . import base

class Store(base.Store):
    """Table storage using an in-memory dictionary."""

    def __post_init__(self):
        self.data = {}

    def _clear(self):
        self.data.clear()

    def apply_changelog_batch(self, batch, to_key, to_value):
        """Apply batch of changelog events to in-memory table."""
        to_delete = set()
        delete_key = self.data.pop
        self.data.update(self._create_batch_iterator(to_delete.add, to_key, to_value, batch))
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(self, mark_as_delete, to_key, to_value, batch):
        for event in batch:
            key = to_key(event.key)
            if event.message.value is None:
                mark_as_delete(key)
                continue
            yield (key, to_value(event.value))

    def persisted_offset(self, tp):
        """Return the persisted offset.

        This always returns :const:`None` when using the in-memory store.
        """
        return None

    def reset_state(self):
        """Remove local file system state.

        This does nothing when using the in-memory store.

        """
        ...