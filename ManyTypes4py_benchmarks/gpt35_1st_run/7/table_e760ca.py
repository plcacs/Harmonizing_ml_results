from typing import Any, Type
from mode import Seconds
from faust import windows
from faust.types.tables import KT, TableT, VT
from faust.types.windows import WindowT
from faust.utils.terminal.tables import dict_as_ansitable
from . import wrappers
from .base import Collection

class Table(TableT[KT, VT], Collection):
    WindowWrapper: ClassVar[Type[WindowWrapperT]] = wrappers.WindowWrapper

    def using_window(self, window: WindowT, *, key_index: bool = False) -> WindowWrapperT:
        self.window = window
        self._changelog_compacting = True
        self._changelog_deleting = True
        self._changelog_topic = None
        return self.WindowWrapper(self, key_index=key_index)

    def hopping(self, size: Seconds, step: Seconds, expires: Seconds = None, key_index: bool = False) -> WindowWrapperT:
        return self.using_window(windows.HoppingWindow(size, step, expires), key_index=key_index)

    def tumbling(self, size: Seconds, expires: Seconds = None, key_index: bool = False) -> WindowWrapperT:
        return self.using_window(windows.TumblingWindow(size, expires), key_index=key_index)

    def __missing__(self, key: KT) -> VT:
        if self.default is not None:
            return self.default()
        raise KeyError(key)

    def _has_key(self, key: KT) -> bool:
        return key in self

    def _get_key(self, key: KT) -> VT:
        return self[key]

    def _set_key(self, key: KT, value: VT) -> None:
        self[key] = value

    def _del_key(self, key: KT) -> None:
        del self[key]

    def on_key_get(self, key: KT) -> None:
        self._sensor_on_get(self, key)

    def on_key_set(self, key: KT, value: VT) -> None:
        fut = self.send_changelog(self.partition_for_key(key), key, value)
        partition = fut.message.partition
        assert partition is not None
        self._maybe_set_key_ttl(key, partition)
        self._sensor_on_set(self, key, value)

    def on_key_del(self, key: KT) -> None:
        fut = self.send_changelog(self.partition_for_key(key), key, value=None, value_serializer='raw')
        partition = fut.message.partition
        assert partition is not None
        self._maybe_del_key_ttl(key, partition)
        self._sensor_on_del(self, key)

    def as_ansitable(self, title: str = '{table.name}', **kwargs: Any) -> Any:
        return dict_as_ansitable(self, title=title.format(table=self), **kwargs)
