"""Wrappers for windowed tables."""
import operator
import typing
from datetime import datetime
from typing import (
    Any, Callable, ClassVar, Dict, ItemsView, Iterator, KeysView, Optional,
    Tuple, Type, ValuesView, cast, Generic, TypeVar, Union, overload
)
from mode import Seconds
from mode.utils.typing import NoReturn
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import (
    EventT, FieldDescriptorT, KT, RecoverCallback, RelativeArg, RelativeHandler,
    TableT, VT, WindowSetT, WindowWrapperT, WindowedItemsViewT,
    WindowedValuesViewT
)
from faust.utils.terminal.tables import dict_as_ansitable

if typing.TYPE_CHECKING:
    from .table import Table as _Table
else:
    class _Table:
        ...

__all__ = [
    'WindowSet', 'WindowWrapper', 'WindowedItemsView',
    'WindowedKeysView', 'WindowedValuesView'
]

T = TypeVar('T')
RT = TypeVar('RT')

class WindowedKeysView(KeysView[KT]):
    """The object returned by ``windowed_table.keys()``."""

    def __init__(self, mapping: 'WindowWrapperT[KT, VT]', event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[KT]:
        """Iterate over keys."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items(self.event):
            yield key

    def __len__(self) -> int:
        return len(self._mapping)

    def func_91gt37gp(self) -> Iterator[KT]:
        """Return all keys present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_now():
            yield key

    def func_u0563mzf(self, event: Optional[EventT] = None) -> Iterator[KT]:
        """Return all keys present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_current(event or self.event):
            yield key

    def func_0bu2wteu(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[KT]:
        """Return all keys present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_delta(d, event or self.event):
            yield key


class WindowedItemsView(WindowedItemsViewT[KT, VT]):
    """The object returned by ``windowed_table.items()``."""

    def __init__(self, mapping: 'WindowWrapperT[KT, VT]', event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        """Iterate over items."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items(self.event)

    def func_91gt37gp(self) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_now()

    def func_u0563mzf(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_current(event or self.event)

    def func_0bu2wteu(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_delta(d, event or self.event)


class WindowedValuesView(WindowedValuesViewT[VT]):
    """The object returned by ``windowed_table.values()``."""

    def __init__(self, mapping: 'WindowWrapperT[KT, VT]', event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[VT]:
        """Iterate over values."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items(self.event):
            yield value

    def func_91gt37gp(self) -> Iterator[VT]:
        """Return all values present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_now():
            yield value

    def func_u0563mzf(self, event: Optional[EventT] = None) -> Iterator[VT]:
        """Return all values present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_current(event or self.event):
            yield value

    def func_0bu2wteu(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[VT]:
        """Return all values present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_delta(d, event or self.event):
            yield value


class WindowSet(WindowSetT[KT, VT]):
    """Represents the windows available for table key."""

    def __init__(
        self,
        key: KT,
        table: TableT[KT, VT],
        wrapper: 'WindowWrapperT[KT, VT]',
        event: Optional[EventT] = None
    ) -> None:
        self.key = key
        self.table = cast(_Table, table)
        self.wrapper = wrapper
        self.event = event
        self.data = table

    def func_x3lewxyp(
        self,
        op: Callable[[T, RT], T],
        value: RT,
        event: Optional[EventT] = None
    ) -> 'WindowSet[KT, VT]':
        """Apply operation to all affected windows."""
        table = cast(_Table, self.table)
        wrapper = cast(WindowWrapper, self.wrapper)
        timestamp = wrapper.get_timestamp(event or self.event)
        wrapper.on_set_key(self.key, value)
        table._apply_window_op(op, self.key, value, timestamp)
        return self

    def func_zikj1qgb(self, event: Optional[EventT] = None) -> VT:
        """Return current value."""
        return cast(_Table, self.table)._windowed_timestamp(
            self.key, self.wrapper.get_timestamp(event or self.event))

    def func_91gt37gp(self) -> VT:
        """Return current value, using the current system time."""
        return cast(_Table, self.table)._windowed_now(self.key)

    def func_u0563mzf(self, event: Optional[EventT] = None) -> VT:
        """Return current value, using stream time-relativity."""
        t = cast(_Table, self.table)
        return t._windowed_timestamp(self.key, t._relative_event(event or self.event))

    def func_0bu2wteu(self, d: Seconds, event: Optional[EventT] = None) -> VT:
        """Return value as it was ±n seconds ago."""
        table = cast(_Table, self.table)
        return table._windowed_delta(self.key, d, event or self.event)

    def __unauthorized_dict_operation(self, operation: str) -> NoReturn:
        raise NotImplementedError(
            f'Accessing {operation} on a WindowSet is not implemented. '
            f'Try using the underlying table directly'
        )

    def func_e58hwo3t(self) -> NoReturn:
        self.__unauthorized_dict_operation('keys')

    def func_afeoycrb(self) -> NoReturn:
        self.__unauthorized_dict_operation('items')

    def func_tg8imvjd(self) -> NoReturn:
        self.__unauthorized_dict_operation('values')

    @overload
    def __getitem__(self, w: EventT) -> 'WindowSet[KT, VT]': ...
    @overload
    def __getitem__(self, w: Any) -> VT: ...

    def __getitem__(self, w: Any) -> Union['WindowSet[KT, VT]', VT]:
        if isinstance(w, EventT):
            return cast(VT, type(self)(self.key, self.table, self.wrapper, w))
        return self.table[self.key, w]

    def __setitem__(self, w: Any, value: VT) -> None:
        if isinstance(w, EventT):
            raise NotImplementedError(
                'Cannot set WindowSet key, when key is an event')
        self.table[self.key, w] = value
        self.wrapper.on_set_key(self.key, value)

    def __delitem__(self, w: Any) -> None:
        if isinstance(w, EventT):
            raise NotImplementedError(
                'Cannot delete WindowSet key, when key is an event')
        del self.table[self.key, w]
        self.wrapper.on_del_key(self.key)

    def __iadd__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.add, other)

    def __isub__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.sub, other)

    def __imul__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.mul, other)

    def __itruediv__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.truediv, other)

    def __ifloordiv__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.floordiv, other)

    def __imod__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.mod, other)

    def __ipow__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.pow, other)

    def __ilshift__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.lshift, other)

    def __irshift__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.rshift, other)

    def __iand__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.and_, other)

    def __ixor__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.xor, other)

    def __ior__(self, other: VT) -> 'WindowSet[KT, VT]':
        return self.apply(operator.or_, other)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: table={self.table}>'


class WindowWrapper(WindowWrapperT[KT, VT]):
    """Windowed table wrapper."""

    ValueType: ClassVar[Type[WindowSet[Any, Any]]] = WindowSet
    key_index: bool = False
    key_index_table: Optional[TableT[KT, int]] = None

    def __init__(
        self,
        table: TableT[KT, VT],
        *,
        relative_to: Optional[RelativeArg] = None,
        key_index: bool = False,
        key_index_table: Optional[TableT[KT, int]] = None
    ) -> None:
        self.table = table
        self.key_index = key_index
        self.key_index_table = key_index_table
        if self.key_index and self.key_index_table is None:
            self.key_index_table = self.table.clone(
                name=f'{self.table.name}-key_index',
                value_type=int,
                key_type=self.table.key_type,
                window=None
            )
        self._get_relative_timestamp = self._relative_handler(relative_to)

    def clone(self, relative_to: Optional[RelativeArg] = None) -> 'WindowWrapper[KT, VT]':
        """Clone this table using a new time-relativity configuration."""
        return type(self)(
            table=self.table,
            relative_to=relative_to or self._get_relative_timestamp,
            key_index=self.key_index,
            key_index_table=self.key_index_table
        )

    @property
    def name(self) -> str:
        """Return the name of this table."""
        return self.table.name

    def relative_to(self, ts: RelativeArg) -> 'WindowWrapper[KT, VT]':
        """Configure the time-relativity of this windowed table."""
        return self.clone(relative_to=ts)

    def relative_to_now(self) -> 'WindowWrapper[KT, VT]':
        """Configure table to be time-relative to the system clock."""
        return self.clone(relative_to=self.table._relative_now)

    def relative_to_field(self, field: FieldDescriptorT) -> 'WindowWrapper[KT, VT]':
        """Configure table to be time-relative to a field in the stream."""
        return self.clone(relative_to=self.table._relative_field(field))

    def relative_to_stream(self) -> 'WindowWrapper[KT, VT]':
        """Configure table to be time-relative to the stream."""
        return self.clone(relative_to=self.table._relative_event)

    def get_timestamp(self, event: Optional[EventT] = None) -> float:
        """Get timestamp from event."""
        event = event or current_event()
        get_relative_timestamp = self.get_relative_timestamp
        if get_relative_timestamp:
            timestamp = get_relative_timestamp(event)
            if isinstance(timestamp, datetime):
                return timestamp.timestamp()
            return timestamp
        if event is None:
            raise RuntimeError('Operation outside of stream iteration')
        return event.message.timestamp

    def on_recover(self, fun: RecoverCallback) -> None:
        """Call after table recovery."""
        return self.table.on_recover(fun)

    def __contains__(self, key: KT) -> bool:
        return self.table._windowed_contains(key, self.get_timestamp())

    def __getitem__(self, key: KT) -> WindowSet[KT, VT]:
        return self.ValueType(key, self.table, self, current_event())

    def __setitem__(self, key: KT, value: VT) -> None:
        if not isinstance(value, WindowSetT):
            table = cast(_Table, self.table)
            self.on_set_key(key, value)
            table._set_windowed(key, value, self.get_timestamp())

    def on_set_key(self, key: KT, value: VT) -> None:
        """Call when the value for a key in this table is set."""
        key_index_table = self.key_index_table
        if key_index_table is not None:
            if key not in key_index_table:
                key_index_table[key] = 1

    def on_del_key(self, key: KT) -> None:
        """Call when a key is deleted from this table."""
        key_index_table = self.key_index_table
        if key_index_table is not None:
            key_index_table.pop(key, None)

    def __delitem__(self, key: KT) -> None:
        self.on_del_key(key)
        cast(_Table, self.table)._del_windowed(key, self.get_timestamp())

    def __len__(self) -> int:
        if self.key_index_table is not None:
            return len(self.key_index_table)
        raise NotImplementedError(
            'Windowed table must use_index=True to support len()')

    def _relative_handler(
        self,
        relative_to: Optional[RelativeArg]
    ) -> Optional[RelativeHandler]:
        if relative_to is None:
            return None
        elif isinstance(relative_to, datetime):
            return self.table._relative_timestamp(relative_to.timestamp())
        elif isinstance(relative_to, float):
            return self.table._relative_timestamp(relative_to)
        elif isinstance(relative_to, FieldDescriptorT):
            return self.table._relative_field(relative_to)
        elif callable(relative_to):
            return relative_to
        raise ImproperlyConfigured(
            f'Relative cannot be type {type(relative_to)}')

    def __iter__(self) -> Iterator[KT]:
        return self._keys()

    def keys(self) -> WindowedKeysView[KT, VT]:
        """Return table keys view: iterate over keys found in this table."""
        return WindowedKeysView(self)

    def _keys(self) -> Iterator[KT]:
        key_index_table = self.key_index_table
        if key_index_table is not None:
            for key in key_index_table.keys():
                yield key
        else:
            raise NotImplementedError(
                'Windowed table must set use_index=True to support .keys/.items/.values'
            )

    def values(self, event: Optional[EventT] = None) -> WindowedValuesView[KT, VT]:
        """Return table values view: iterate over values in this table."""
        return WindowedValuesView(self, event or current_event())

    def items(self, event: Optional[EventT] = None) -> WindowedItemsView[KT, VT]:
        """Return table items view: iterate over ``(key, value)`` pairs."""
        return WindowedItemsView(self, event or current_event())

    def _items(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        timestamp = self.get_timestamp(event)
        for key in self._keys():
            try:
                yield key, table._windowed_timestamp(key, timestamp)
            except KeyError:
                pass

    def _items_now(self) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield key, table._windowed_now(key)
            except KeyError:
                pass

    def _items_current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        timestamp = table._relative_event(event)
        for key in self._keys():
            try:
                yield key, table._windowed_timestamp(key, timestamp)
