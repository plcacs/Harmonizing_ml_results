"""In-memory table storage."""
from typing import Any, Callable, Iterable, MutableMapping, Optional, Set, Tuple
from faust.types import EventT, TP
from . import base

class Store(base.Store):
    """Table storage using an in-memory dictionary."""

    def __post_init__(self) -> None:
        self.data = {}

    def _clear(self) -> None:
        self.data.clear()

    def apply_changelog_batch(self, batch: Union[typing.Callable, typing.Iterable[faustypes.EventT]], to_key: Union[typing.Callable, typing.Iterable[faustypes.EventT]], to_value: Union[typing.Callable, typing.Iterable[faustypes.EventT]]) -> None:
        """Apply batch of changelog events to in-memory table."""
        to_delete = set()
        delete_key = self.data.pop
        self.data.update(self._create_batch_iterator(to_delete.add, to_key, to_value, batch))
        for key in to_delete:
            delete_key(key, None)

    def _create_batch_iterator(self, mark_as_delete: Union[typing.Callable[typing.Any, None], int, str], to_key: Union[dict, typing.Callable, str], to_value: Union[typing.Callable, None], batch: Any) -> typing.Generator[tuple]:
        for event in batch:
            key = to_key(event.key)
            if event.message.value is None:
                mark_as_delete(key)
                continue
            yield (key, to_value(event.value))

    def persisted_offset(self, tp: Union[tuples.TP, typing.Type, None]) -> None:
        """Return the persisted offset.

        This always returns :const:`None` when using the in-memory store.
        """
        return None

    def reset_state(self) -> None:
        """Remove local file system state.

        This does nothing when using the in-memory store.

        """
        ...