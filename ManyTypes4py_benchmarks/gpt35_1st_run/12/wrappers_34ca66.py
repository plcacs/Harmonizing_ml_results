from typing import Any, Callable, ItemsView, Iterator, KeysView, Optional, Tuple, Type, ValuesView
from faust.types.tables import KT, VT, WindowSetT, WindowWrapperT, WindowedItemsViewT, WindowedValuesViewT
from faust.streams import current_event
from faust.types import EventT, FieldDescriptorT
from faust.exceptions import ImproperlyConfigured

class WindowedKeysView(KeysView):
    def __init__(self, mapping, event=None) -> None:
    def __iter__(self) -> Iterator:
    def __len__(self) -> int:
    def now(self) -> Iterator:
    def current(self, event=None) -> Iterator:
    def delta(self, d, event=None) -> Iterator:

class WindowedItemsView(WindowedItemsViewT):
    def __init__(self, mapping, event=None) -> None:
    def __iter__(self) -> Iterator:
    def now(self) -> ItemsView:
    def current(self, event=None) -> ItemsView:
    def delta(self, d, event=None) -> ItemsView:

class WindowedValuesView(WindowedValuesViewT):
    def __init__(self, mapping, event=None) -> None:
    def __iter__(self) -> Iterator:
    def now(self) -> Iterator:
    def current(self, event=None) -> Iterator:
    def delta(self, d, event=None) -> Iterator:

class WindowSet(WindowSetT[KT, VT]):
    def __init__(self, key, table, wrapper, event=None) -> None:
    def apply(self, op, value, event=None) -> 'WindowSet':
    def value(self, event=None) -> VT:
    def now(self) -> VT:
    def current(self, event=None) -> VT:
    def delta(self, d, event=None) -> VT:
    def __unauthorized_dict_operation(self, operation) -> None:
    def keys(self) -> None:
    def items(self) -> None:
    def values(self) -> None:
    def __getitem__(self, w) -> VT:
    def __setitem__(self, w, value) -> None:
    def __delitem__(self, w) -> None:
    def __iadd__(self, other) -> 'WindowSet':
    def __isub__(self, other) -> 'WindowSet':
    def __imul__(self, other) -> 'WindowSet':
    def __itruediv__(self, other) -> 'WindowSet':
    def __ifloordiv__(self, other) -> 'WindowSet':
    def __imod__(self, other) -> 'WindowSet':
    def __ipow__(self, other) -> 'WindowSet':
    def __ilshift__(self, other) -> 'WindowSet':
    def __irshift__(self, other) -> 'WindowSet':
    def __iand__(self, other) -> 'WindowSet':
    def __ixor__(self, other) -> 'WindowSet':
    def __ior__(self, other) -> 'WindowSet':
    def __repr__(self) -> str:

class WindowWrapper(WindowWrapperT):
    def __init__(self, table, *, relative_to=None, key_index=False, key_index_table=None) -> None:
    def clone(self, relative_to) -> 'WindowWrapper':
    def name(self) -> str:
    def relative_to(self, ts) -> 'WindowWrapper':
    def relative_to_now(self) -> 'WindowWrapper':
    def relative_to_field(self, field) -> 'WindowWrapper':
    def relative_to_stream(self) -> 'WindowWrapper':
    def get_timestamp(self, event=None) -> float:
    def on_recover(self, fun) -> None:
    def __contains__(self, key) -> bool:
    def __getitem__(self, key) -> WindowSet:
    def __setitem__(self, key, value) -> None:
    def on_set_key(self, key, value) -> None:
    def on_del_key(self, key) -> None:
    def __delitem__(self, key) -> None:
    def __len__(self) -> int:
    def _relative_handler(self, relative_to) -> Optional[Callable]:
    def __iter__(self) -> Iterator:
    def keys(self) -> KeysView:
    def _keys(self) -> Iterator:
    def values(self, event=None) -> ValuesView:
    def items(self, event=None) -> ItemsView:
    def _items(self, event=None) -> ItemsView:
    def _items_now(self) -> ItemsView:
    def _items_current(self, event=None) -> ItemsView:
    def _items_delta(self, d, event=None) -> ItemsView:
    def as_ansitable(self, title='{table.name}', **kwargs) -> Any:
    @property
    def get_relative_timestamp(self) -> Optional[Callable]:
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to) -> None:
