"""Wrappers for windowed tables."""
import operator
import typing
from datetime import datetime
from typing import Any, Callable, ClassVar, ItemsView, Iterator, KeysView, Optional, Tuple, Type, ValuesView, cast
from mode import Seconds
from mode.utils.typing import NoReturn
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import EventT, FieldDescriptorT
from faust.types.tables import KT, RecoverCallback, RelativeArg, RelativeHandler, TableT, VT, WindowSetT, WindowWrapperT, WindowedItemsViewT, WindowedKeysView, WindowedValuesViewT
from faust.utils.terminal.tables import dict_as_ansitable
if typing.TYPE_CHECKING:
    from .table import Table as _Table
else:
    class _Table:
        ...

__all__ = ['WindowSet', 'WindowWrapper', 'WindowedItemsView', 'WindowedKeysView', 'WindowedValuesView']

class WindowedKeysView(KeysView):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[KT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items(self.event):
            yield key

    def __len__(self) -> int:
        return len(self._mapping)

    def now(self) -> Iterator[KT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_now():
            yield key

    def current(self, event: Optional[EventT] = None) -> Iterator[KT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_current(event or self.event):
            yield key

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[KT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_delta(d, event or self.event):
            yield key

class WindowedItemsView(WindowedItemsViewT):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items(self.event)

    def now(self) -> Iterator[Tuple[KT, VT]]:
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_now()

    def current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_current(event or self.event)

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_delta(d, event or self.event)

class WindowedValuesView(WindowedValuesViewT):
    def __init__(self, mapping: WindowWrapperT, event: Optional[EventT] = None) -> None:
        self._mapping = mapping
        self.event = event

    def __iter__(self) -> Iterator[VT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items(self.event):
            yield value

    def now(self) -> Iterator[VT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_now():
            yield value

    def current(self, event: Optional[EventT] = None) -> Iterator[VT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_current(event or self.event):
            yield value

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[VT]:
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_delta(d, event or self.event):
            yield value

class WindowSet(WindowSetT[KT, VT]):
    def __init__(self, key: KT, table: _Table, wrapper: WindowWrapperT, event: Optional[EventT] = None) -> None:
        self.key = key
        self.table = cast(_Table, table)
        self.wrapper = wrapper
        self.event = event
        self.data = table

    def apply(self, op: Callable[[VT, VT], VT], value: VT, event: Optional[EventT] = None) -> 'WindowSet':
        table = cast(_Table, self.table)
        wrapper = cast(WindowWrapper, self.wrapper)
        timestamp = wrapper.get_timestamp(event or self.event)
        wrapper.on_set_key(self.key, value)
        table._apply_window_op(op, self.key, value, timestamp)
        return self

    def value(self, event: Optional[EventT] = None) -> VT:
        return cast(_Table, self.table)._windowed_timestamp(self.key, self.wrapper.get_timestamp(event or self.event))

    def now(self) -> VT:
        return cast(_Table, self.table)._windowed_now(self.key)

    def current(self, event: Optional[EventT] = None) -> VT:
        t = cast(_Table, self.table)
        return t._windowed_timestamp(self.key, t._relative_event(event or self.event))

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> VT:
        table = cast(_Table, self.table)
        return table._windowed_delta(self.key, d, event or self.event)

    def __unauthorized_dict_operation(self, operation: str) -> NoReturn:
        raise NotImplementedError(f'Accessing {operation} on a WindowSet is not implemented. Try using the underlying table directly')

    def keys(self) -> NoReturn:
        self.__unauthorized_dict_operation('keys')

    def items(self) -> NoReturn:
        self.__unauthorized_dict_operation('items')

    def values(self) -> NoReturn:
        self.__unauthorized_dict_operation('values')

    def __getitem__(self, w: Any) -> 'WindowSet':
        if isinstance(w, EventT):
            return type(self)(self.key, self.table, self.wrapper, w)
        return self.table[self.key, w]

    def __setitem__(self, w: Any, value: VT) -> None:
        if isinstance(w, EventT):
            raise NotImplementedError('Cannot set WindowSet key, when key is an event')
        self.table[self.key, w] = value
        self.wrapper.on_set_key(self.key, value)

    def __delitem__(self, w: Any) -> None:
        if isinstance(w, EventT):
            raise NotImplementedError('Cannot delete WindowSet key, when key is an event')
        del self.table[self.key, w]
        self.wrapper.on_del_key(self.key)

    def __iadd__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.add, other)

    def __isub__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.sub, other)

    def __imul__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.mul, other)

    def __itruediv__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.truediv, other)

    def __ifloordiv__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.floordiv, other)

    def __imod__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.mod, other)

    def __ipow__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.pow, other)

    def __ilshift__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.lshift, other)

    def __irshift__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.rshift, other)

    def __iand__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.and_, other)

    def __ixor__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.xor, other)

    def __ior__(self, other: VT) -> 'WindowSet':
        return self.apply(operator.or_, other)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: table={self.table}>'

class WindowWrapper(WindowWrapperT):
    ValueType = WindowSet
    key_index = False
    key_index_table = None

    def __init__(self, table: _Table, *, relative_to: Optional[RelativeArg] = None, key_index: bool = False, key_index_table: Optional[TableT] = None) -> None:
        self.table = table
        self.key_index = key_index
        self.key_index_table = key_index_table
        if self.key_index and self.key_index_table is None:
            self.key_index_table = self.table.clone(name=f'{self.table.name}-key_index', value_type=int, key_type=self.table.key_type, window=None)
        self._get_relative_timestamp = self._relative_handler(relative_to)

    def clone(self, relative_to: Optional[RelativeArg] = None) -> 'WindowWrapper':
        return type(self)(table=self.table, relative_to=relative_to or self._get_relative_timestamp, key_index=self.key_index, key_index_table=self.key_index_table)

    @property
    def name(self) -> str:
        return self.table.name

    def relative_to(self, ts: RelativeArg) -> 'WindowWrapper':
        return self.clone(relative_to=ts)

    def relative_to_now(self) -> 'WindowWrapper':
        return self.clone(relative_to=self.table._relative_now)

    def relative_to_field(self, field: FieldDescriptorT) -> 'WindowWrapper':
        return self.clone(relative_to=self.table._relative_field(field))

    def relative_to_stream(self) -> 'WindowWrapper':
        return self.clone(relative_to=self.table._relative_event)

    def get_timestamp(self, event: Optional[EventT] = None) -> float:
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

    def on_recover(self, fun: RecoverCallback) -> RecoverCallback:
        return self.table.on_recover(fun)

    def __contains__(self, key: KT) -> bool:
        return self.table._windowed_contains(key, self.get_timestamp())

    def __getitem__(self, key: KT) -> WindowSet:
        return self.ValueType(key, self.table, self, current_event())

    def __setitem__(self, key: KT, value: VT) -> None:
        if not isinstance(value, WindowSetT):
            table = cast(_Table, self.table)
            self.on_set_key(key, value)
            table._set_windowed(key, value, self.get_timestamp())

    def on_set_key(self, key: KT, value: VT) -> None:
        key_index_table = self.key_index_table
        if key_index_table is not None:
            if key not in key_index_table:
                key_index_table[key] = 1

    def on_del_key(self, key: KT) -> None:
        key_index_table = self.key_index_table
        if key_index_table is not None:
            key_index_table.pop(key, None)

    def __delitem__(self, key: KT) -> None:
        self.on_del_key(key)
        cast(_Table, self.table)._del_windowed(key, self.get_timestamp())

    def __len__(self) -> int:
        if self.key_index_table is not None:
            return len(self.key_index_table)
        raise NotImplementedError('Windowed table must use_index=True to support len()')

    def _relative_handler(self, relative_to: Optional[RelativeArg]) -> Optional[RelativeHandler]:
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
        raise ImproperlyConfigured(f'Relative cannot be type {type(relative_to)}')

    def __iter__(self) -> Iterator[KT]:
        return self._keys()

    def keys(self) -> WindowedKeysView:
        return WindowedKeysView(self)

    def _keys(self) -> Iterator[KT]:
        key_index_table = self.key_index_table
        if key_index_table is not None:
            for key in key_index_table.keys():
                yield key
        else:
            raise NotImplementedError('Windowed table must set use_index=True to support .keys/.items/.values')

    def values(self, event: Optional[EventT] = None) -> WindowedValuesViewT:
        return WindowedValuesView(self, event or current_event())

    def items(self, event: Optional[EventT] = None) -> WindowedItemsViewT:
        return WindowedItemsView(self, event or current_event())

    def _items(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        timestamp = self.get_timestamp(event)
        for key in self._keys():
            try:
                yield (key, table._windowed_timestamp(key, timestamp))
            except KeyError:
                pass

    def _items_now(self) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield (key, table._windowed_now(key))
            except KeyError:
                pass

    def _items_current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        timestamp = table._relative_event(event)
        for key in self._keys():
            try:
                yield (key, table._windowed_timestamp(key, timestamp))
            except KeyError:
                pass

    def _items_delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield (key, table._windowed_delta(key, d, event))
            except KeyError:
                pass

    def as_ansitable(self, title: str = '{table.name}', **kwargs: Any) -> str:
        return dict_as_ansitable(self, title=title.format(table=self.table), **kwargs)

    @property
    def get_relative_timestamp(self) -> Optional[RelativeHandler]:
        return self._get_relative_timestamp

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Optional[RelativeArg]) -> None:
        self._get_relative_timestamp = self._relative_handler(relative_to)