"""Wrappers for windowed tables."""

import datetime
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    KeysView,
    Optional,
    Tuple,
    Type,
    Union,
    ValuesView,
    cast,
)

from faust.types import EventT
from faust.types.tables import (
    KT,
    RecoverCallback,
    RelativeArg,
    RelativeHandler,
    TableT,
    VT,
    WindowSetT,
    WindowWrapperT,
    WindowedItemsViewT,
    WindowedValuesViewT,
)

__all__ = [
    'WindowSet',
    'WindowWrapper',
    'WindowedItemsView',
    'WindowedKeysView',
    'WindowedValuesView',
]

class WindowedKeysView(KeysView[KT]):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        ...

    def __iter__(self) -> Iterator[KT]:
        ...

    def __len__(self) -> int:
        ...

    def now(self) -> Iterator[KT]:
        ...

    def current(self, event: Optional[EventT] = None) -> Iterator[KT]:
        ...

    def delta(self, d: RelativeArg, event: Optional[EventT] = None) -> Iterator[KT]:
        ...

class WindowedItemsView(WindowedItemsViewT[KT, VT]):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        ...

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        ...

    def now(self) -> Iterator[Tuple[KT, VT]]:
        ...

    def current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        ...

    def delta(self, d: RelativeArg, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        ...

class WindowedValuesView(WindowedValuesViewT[VT]):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        ...

    def __iter__(self) -> Iterator[VT]:
        ...

    def now(self) -> Iterator[VT]:
        ...

    def current(self, event: Optional[EventT] = None) -> Iterator[VT]:
        ...

    def delta(self, d: RelativeArg, event: Optional[EventT] = None) -> Iterator[VT]:
        ...

class WindowSet(WindowSetT[KT, VT]):
    def __init__(self, key: KT, table: TableT, wrapper: WindowWrapperT, event: Optional[EventT] = None) -> None:
        ...

    def apply(self, op: Callable[[VT, VT], VT], value: VT, event: Optional[EventT] = None) -> 'Self':
        ...

    def value(self, event: Optional[EventT] = None) -> VT:
        ...

    def now(self) -> Union[VT, None]:
        ...

    def current(self, event: Optional[EventT] = None) -> Union[VT, None]:
        ...

    def delta(self, d: RelativeArg, event: Optional[EventT] = None) -> Union[VT, None]:
        ...

    def __getitem__(self, w: Any) -> Union['Self', VT]:
        ...

    def __setitem__(self, w: Any, value: VT) -> None:
        ...

    def __delitem__(self, w: Any) -> None:
        ...

    def __iadd__(self, other: Any) -> 'Self':
        ...

    def __isub__(self, other: Any) -> 'Self':
        ...

    def __imul__(self, other: Any) -> 'Self':
        ...

    def __itruediv__(self, other: Any) -> 'Self':
        ...

    def __ifloordiv__(self, other: Any) -> 'Self':
        ...

    def __imod__(self, other: Any) -> 'Self':
        ...

    def __ipow__(self, other: Any) -> 'Self':
        ...

    def __ilshift__(self, other: Any) -> 'Self':
        ...

    def __irshift__(self, other: Any) -> 'Self':
        ...

    def __iand__(self, other: Any) -> 'Self':
        ...

    def __ixor__(self, other: Any) -> 'Self':
        ...

    def __ior__(self, other: Any) -> 'Self':
        ...

    def __repr__(self) -> str:
        ...

class WindowWrapper(WindowWrapperT[KT, VT]):
    ValueType: Type[WindowSetT[KT, VT]] = ...

    def __init__(self, table: TableT, *, relative_to: Optional[RelativeArg] = None, key_index: bool = False, key_index_table: Optional[TableT] = None) -> None:
        ...

    def clone(self, relative_to: Optional[RelativeArg] = None) -> 'Self':
        ...

    @property
    def name(self) -> str:
        ...

    def relative_to(self, ts: RelativeArg) -> 'Self':
        ...

    def relative_to_now(self) -> 'Self':
        ...

    def relative_to_field(self, field: Any) -> 'Self':
        ...

    def relative_to_stream(self) -> 'Self':
        ...

    def get_timestamp(self, event: Optional[EventT] = None) -> Optional[float]:
        ...

    def on_recover(self, fun: RecoverCallback) -> RecoverCallback:
        ...

    def __contains__(self, key: KT) -> bool:
        ...

    def __getitem__(self, key: KT) -> WindowSetT[KT, VT]:
        ...

    def __setitem__(self, key: KT, value: Any) -> None:
        ...

    def __delitem__(self, key: KT) -> None:
        ...

    def __len__(self) -> int:
        ...

    def _relative_handler(self, relative_to: Optional[RelativeArg]) -> Optional[RelativeHandler]:
        ...

    def __iter__(self) -> Iterator[KT]:
        ...

    def keys(self) -> WindowedKeysView[KT]:
        ...

    def _keys(self) -> Iterator[KT]:
        ...

    def values(self, event: Optional[EventT] = None) -> WindowedValuesView[VT]:
        ...

    def items(self, event: Optional[EventT] = None) -> WindowedItemsView[KT, VT]:
        ...

    def _items(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        ...

    def _items_now(self) -> Iterator[Tuple[KT, VT]]:
        ...

    def _items_current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        ...

    def _items_delta(self, d: RelativeArg, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        ...

    def as_ansitable(self, title: str = '{table.name}', **kwargs: Any) -> str:
        ...

    @property
    def get_relative_timestamp(self) -> Optional[RelativeHandler]:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: RelativeArg) -> None:
        ...