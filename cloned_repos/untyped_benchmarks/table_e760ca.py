"""Table (key/value changelog stream)."""
from typing import Any, ClassVar, Type
from mode import Seconds
from faust import windows
from faust.types.tables import KT, TableT, VT, WindowWrapperT
from faust.types.windows import WindowT
from faust.utils.terminal.tables import dict_as_ansitable
from . import wrappers
from .base import Collection
__all__ = ['Table']

class Table(TableT[KT, VT], Collection):
    """Table (non-windowed)."""
    WindowWrapper = wrappers.WindowWrapper

    def using_window(self, window, *, key_index=False):
        """Wrap table using a specific window type."""
        self.window = window
        self._changelog_compacting = True
        self._changelog_deleting = True
        self._changelog_topic = None
        return self.WindowWrapper(self, key_index=key_index)

    def hopping(self, size, step, expires=None, key_index=False):
        """Wrap table in a hopping window."""
        return self.using_window(windows.HoppingWindow(size, step, expires), key_index=key_index)

    def tumbling(self, size, expires=None, key_index=False):
        """Wrap table in a tumbling window."""
        return self.using_window(windows.TumblingWindow(size, expires), key_index=key_index)

    def __missing__(self, key):
        if self.default is not None:
            return self.default()
        raise KeyError(key)

    def _has_key(self, key):
        return key in self

    def _get_key(self, key):
        return self[key]

    def _set_key(self, key, value):
        self[key] = value

    def _del_key(self, key):
        del self[key]

    def on_key_get(self, key):
        """Call when the value for a key in this table is retrieved."""
        self._sensor_on_get(self, key)

    def on_key_set(self, key, value):
        """Call when the value for a key in this table is set."""
        fut = self.send_changelog(self.partition_for_key(key), key, value)
        partition = fut.message.partition
        assert partition is not None
        self._maybe_set_key_ttl(key, partition)
        self._sensor_on_set(self, key, value)

    def on_key_del(self, key):
        """Call when a key in this table is removed."""
        fut = self.send_changelog(self.partition_for_key(key), key, value=None, value_serializer='raw')
        partition = fut.message.partition
        assert partition is not None
        self._maybe_del_key_ttl(key, partition)
        self._sensor_on_del(self, key)

    def as_ansitable(self, title='{table.name}', **kwargs):
        """Draw table as a a terminal ANSI table."""
        return dict_as_ansitable(self, title=title.format(table=self), **kwargs)