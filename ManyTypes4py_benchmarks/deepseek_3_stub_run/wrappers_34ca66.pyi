"""Wrappers for windowed tables."""
import operator
import typing
from datetime import datetime
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
    cast,
    overload,
)

from mode import Seconds
from mode.utils.typing import NoReturn
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import EventT, FieldDescriptorT
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
from faust.utils.terminal.tables import dict_as_ansitable

if typing.TYPE_CHECKING:
    from .table import Table as _Table
else:
    class _Table: ...

__all__: Tuple[str, ...] = (
    'WindowSet',
    'WindowWrapper',
    'WindowedItemsView',
    'WindowedKeysView',
    'WindowedValuesView',
)

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

class WindowedKeysView(KeysView[_KT]):
    """The object returned by ``windowed_table.keys()``."""
    
    def __init__(self, mapping: Any, event: Optional[EventT] = ...) -> None: ...
    
    def __iter__(self) -> Iterator[_KT]: ...
    
    def __len__(self) -> int: ...
    
    def now(self) -> Iterator[_KT]: ...
    
    def current(self, event: Optional[EventT] = ...) -> Iterator[_KT]: ...
    
    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> Iterator[_KT]: ...

class WindowedItemsView(WindowedItemsViewT[_KT, _VT]):
    """The object returned by ``windowed_table.items()``."""
    
    def __init__(self, mapping: Any, event: Optional[EventT] = ...) -> None: ...
    
    def __iter__(self) -> Iterator[Tuple[_KT, _VT]]: ...
    
    def now(self) -> Iterator[Tuple[_KT, _VT]]: ...
    
    def current(self, event: Optional[EventT] = ...) -> Iterator[Tuple[_KT, _VT]]: ...
    
    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> Iterator[Tuple[_KT, _VT]]: ...

class WindowedValuesView(WindowedValuesViewT[_VT]):
    """The object returned by ``windowed_table.values()``."""
    
    def __init__(self, mapping: Any, event: Optional[EventT] = ...) -> None: ...
    
    def __iter__(self) -> Iterator[_VT]: ...
    
    def now(self) -> Iterator[_VT]: ...
    
    def current(self, event: Optional[EventT] = ...) -> Iterator[_VT]: ...
    
    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> Iterator[_VT]: ...

class WindowSet(WindowSetT[KT, VT]):
    """Represents the windows available for table key."""
    
    def __init__(
        self,
        key: KT,
        table: TableT[KT, VT],
        wrapper: WindowWrapperT,
        event: Optional[EventT] = ...,
    ) -> None: ...
    
    def apply(
        self,
        op: Callable[[Any, Any], Any],
        value: VT,
        event: Optional[EventT] = ...,
    ) -> "WindowSet[KT, VT]": ...
    
    def value(self, event: Optional[EventT] = ...) -> VT: ...
    
    def now(self) -> VT: ...
    
    def current(self, event: Optional[EventT] = ...) -> VT: ...
    
    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> VT: ...
    
    def keys(self) -> NoReturn: ...
    
    def items(self) -> NoReturn: ...
    
    def values(self) -> NoReturn: ...
    
    @overload
    def __getitem__(self, w: EventT) -> "WindowSet[KT, VT]": ...
    
    @overload
    def __getitem__(self, w: Any) -> VT: ...
    
    def __getitem__(self, w: Any) -> Union["WindowSet[KT, VT]", VT]: ...
    
    def __setitem__(self, w: Any, value: VT) -> None: ...
    
    def __delitem__(self, w: Any) -> None: ...
    
    def __iadd__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __isub__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __imul__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __itruediv__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __ifloordiv__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __imod__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __ipow__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __ilshift__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __irshift__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __iand__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __ixor__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __ior__(self, other: VT) -> "WindowSet[KT, VT]": ...
    
    def __repr__(self) -> str: ...

class WindowWrapper(WindowWrapperT):
    """Windowed table wrapper."""
    
    ValueType: ClassVar[Type[WindowSet]] = ...
    key_index: bool = ...
    key_index_table: Optional[TableT[Any, int]] = ...
    
    def __init__(
        self,
        table: TableT[KT, VT],
        *,
        relative_to: Optional[RelativeArg] = ...,
        key_index: bool = ...,
        key_index_table: Optional[TableT[Any, int]] = ...,
    ) -> None: ...
    
    def clone(self, relative_to: Optional[RelativeArg]) -> "WindowWrapper": ...
    
    @property
    def name(self) -> str: ...
    
    def relative_to(self, ts: RelativeArg) -> "WindowWrapper": ...
    
    def relative_to_now(self) -> "WindowWrapper": ...
    
    def relative_to_field(self, field: FieldDescriptorT) -> "WindowWrapper": ...
    
    def relative_to_stream(self) -> "WindowWrapper": ...
    
    def get_timestamp(self, event: Optional[EventT] = ...) -> float: ...
    
    def on_recover(self, fun: RecoverCallback) -> RecoverCallback: ...
    
    def __contains__(self, key: KT) -> bool: ...
    
    def __getitem__(self, key: KT) -> WindowSet[KT, VT]: ...
    
    def __setitem__(self, key: KT, value: VT) -> None: ...
    
    def on_set_key(self, key: KT, value: VT) -> None: ...
    
    def on_del_key(self, key: KT) -> None: ...
    
    def __delitem__(self, key: KT) -> None: ...
    
    def __len__(self) -> int: ...
    
    def _relative_handler(
        self,
        relative_to: Optional[RelativeArg],
    ) -> Optional[RelativeHandler]: ...
    
    def __iter__(self) -> Iterator[KT]: ...
    
    def keys(self) -> WindowedKeysView[KT]: ...
    
    def _keys(self) -> Iterator[KT]: ...
    
    def values(self, event: Optional[EventT] = ...) -> WindowedValuesView[VT]: ...
    
    def items(self, event: Optional[EventT] = ...) -> WindowedItemsView[KT, VT]: ...
    
    def _items(
        self,
        event: Optional[EventT] = ...,
    ) -> Iterator[Tuple[KT, VT]]: ...
    
    def _items_now(self) -> Iterator[Tuple[KT, VT]]: ...
    
    def _items_current(
        self,
        event: Optional[EventT] = ...,
    ) -> Iterator[Tuple[KT, VT]]: ...
    
    def _items_delta(
        self,
        d: Seconds,
        event: Optional[EventT] = ...,
    ) -> Iterator[Tuple[KT, VT]]: ...
    
    def as_ansitable(
        self,
        title: str = ...,
        **kwargs: Any,
    ) -> str: ...
    
    @property
    def get_relative_timestamp(self) -> Optional[RelativeHandler]: ...
    
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: RelativeArg) -> None: ...