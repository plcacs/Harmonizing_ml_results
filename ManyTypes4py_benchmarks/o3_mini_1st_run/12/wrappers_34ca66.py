#!/usr/bin/env python3
"""Wrappers for windowed tables."""
import operator
import typing
from datetime import datetime
from typing import Any, Callable, ClassVar, Iterator, Optional, Tuple, Type, cast

from mode import Seconds
from mode.utils.typing import NoReturn
from faust.exceptions import ImproperlyConfigured
from faust.streams import current_event
from faust.types import EventT, FieldDescriptorT
from faust.types.tables import KT, RecoverCallback, RelativeArg, RelativeHandler, TableT, VT, WindowSetT, WindowWrapperT, WindowedItemsViewT, WindowedValuesViewT

from faust.utils.terminal.tables import dict_as_ansitable

if typing.TYPE_CHECKING:
    from .table import Table as _Table
else:

    class _Table:
        ...


__all__ = ['WindowSet', 'WindowWrapper', 'WindowedItemsView', 'WindowedKeysView', 'WindowedValuesView']


class WindowedKeysView(typing.KeysView, typing.Generic[KT]):
    """The object returned by ``windowed_table.keys()``."""

    def __init__(self, mapping: "WindowWrapper", event: Optional[EventT] = None) -> None:
        self._mapping: "WindowWrapper" = mapping
        self.event: Optional[EventT] = event

    def __iter__(self) -> Iterator[KT]:
        """Iterate over keys.

        The window is chosen based on the table's time-relativity setting.
        """
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items(self.event):
            yield key
    def __len__(self) -> int:
        return len(self._mapping)

    def now(self) -> Iterator[KT]:
        """Return all keys present in window closest to system time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_now():
            yield key

    def current(self, event: Optional[EventT] = None) -> Iterator[KT]:
        """Return all keys present in window closest to stream time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_current(event or self.event):
            yield key

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[KT]:
        """Return all keys present in window ±n seconds ago."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_delta(d, event or self.event):
            yield key


class WindowedItemsView(WindowedItemsViewT, typing.Generic[KT, VT]):
    """The object returned by ``windowed_table.items()``."""

    def __init__(self, mapping: "WindowWrapper", event: Optional[EventT] = None) -> None:
        self._mapping: "WindowWrapper" = mapping
        self.event: Optional[EventT] = event

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        """Iterate over items.

        The window is chosen based on the table's time-relativity setting.
        """
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items(self.event)

    def now(self) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window closest to system time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_now()

    def current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window closest to stream time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_current(event or self.event)

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        """Return all items present in window ±n seconds ago."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_delta(d, event or self.event)


class WindowedValuesView(WindowedValuesViewT, typing.Generic[VT]):
    """The object returned by ``windowed_table.values()``."""

    def __init__(self, mapping: "WindowWrapper", event: Optional[EventT] = None) -> None:
        self._mapping: "WindowWrapper" = mapping
        self.event: Optional[EventT] = event

    def __iter__(self) -> Iterator[VT]:
        """Iterate over values.

        The window is chosen based on the table's time-relativity setting.
        """
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items(self.event):
            yield value

    def now(self) -> Iterator[VT]:
        """Return all values present in window closest to system time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_now():
            yield value

    def current(self, event: Optional[EventT] = None) -> Iterator[VT]:
        """Return all values present in window closest to stream time."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_current(event or self.event):
            yield value

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[VT]:
        """Return all values present in window ±n seconds ago."""
        wrapper: WindowWrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_delta(d, event or self.event):
            yield value


class WindowSet(WindowSetT[KT, VT]):
    """Represents the windows available for a table key.

    ``Table[k]`` returns WindowSet since ``k`` can exist in multiple
    windows, and to retrieve an actual item we need a timestamp.

    The timestamp of the current event (if this is executing in a stream
    processor) can be used by accessing ``.current()``::

        Table[k].current()

    Similarly the most recent value can be accessed using ``.now()``::

        Table[k].now()

    From delta of the time of the current event::

        Table[k].delta(timedelta(hours=3))

    Or delta from time of other event::

        Table[k].delta(timedelta(hours=3), other_event)

    """

    def __init__(self, key: KT, table: _Table, wrapper: "WindowWrapper", event: Optional[EventT] = None) -> None:
        self.key: KT = key
        self.table: _Table = cast(_Table, table)
        self.wrapper: "WindowWrapper" = wrapper
        self.event: Optional[EventT] = event
        self.data: _Table = table

    def apply(self, op: Callable[[VT, Any], VT], value: Any, event: Optional[EventT] = None) -> "WindowSet":
        """Apply operation to all affected windows."""
        table: _Table = cast(_Table, self.table)
        wrapper: WindowWrapper = cast(WindowWrapper, self.wrapper)
        timestamp: float = wrapper.get_timestamp(event or self.event)
        wrapper.on_set_key(self.key, value)
        table._apply_window_op(op, self.key, value, timestamp)
        return self

    def value(self, event: Optional[EventT] = None) -> VT:
        """Return current value.

        The selected window depends on the current time-relativity
        setting used (:meth:`relative_to_now`, :meth:`relative_to_stream`,
        :meth:`relative_to_field`, etc.)
        """
        return cast(_Table, self.table)._windowed_timestamp(self.key, self.wrapper.get_timestamp(event or self.event))

    def now(self) -> VT:
        """Return current value, using the current system time."""
        return cast(_Table, self.table)._windowed_now(self.key)

    def current(self, event: Optional[EventT] = None) -> VT:
        """Return current value, using stream time-relativity."""
        t: _Table = cast(_Table, self.table)
        return t._windowed_timestamp(self.key, t._relative_event(event or self.event))

    def delta(self, d: Seconds, event: Optional[EventT] = None) -> VT:
        """Return value as it was ±n seconds ago."""
        table: _Table = cast(_Table, self.table)
        return table._windowed_delta(self.key, d, event or self.event)

    def __unauthorized_dict_operation(self, operation: str) -> NoReturn:
        raise NotImplementedError(f'Accessing {operation} on a WindowSet is not implemented. Try using the underlying table directly')

    def keys(self) -> None:
        self.__unauthorized_dict_operation('keys')

    def items(self) -> None:
        self.__unauthorized_dict_operation('items')

    def values(self) -> None:
        self.__unauthorized_dict_operation('values')

    def __getitem__(self, w: Any) -> Any:
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

    def __iadd__(self, other: Any) -> "WindowSet":
        return self.apply(operator.add, other)

    def __isub__(self, other: Any) -> "WindowSet":
        return self.apply(operator.sub, other)

    def __imul__(self, other: Any) -> "WindowSet":
        return self.apply(operator.mul, other)

    def __itruediv__(self, other: Any) -> "WindowSet":
        return self.apply(operator.truediv, other)

    def __ifloordiv__(self, other: Any) -> "WindowSet":
        return self.apply(operator.floordiv, other)

    def __imod__(self, other: Any) -> "WindowSet":
        return self.apply(operator.mod, other)

    def __ipow__(self, other: Any) -> "WindowSet":
        return self.apply(operator.pow, other)

    def __ilshift__(self, other: Any) -> "WindowSet":
        return self.apply(operator.lshift, other)

    def __irshift__(self, other: Any) -> "WindowSet":
        return self.apply(operator.rshift, other)

    def __iand__(self, other: Any) -> "WindowSet":
        return self.apply(operator.and_, other)

    def __ixor__(self, other: Any) -> "WindowSet":
        return self.apply(operator.xor, other)

    def __ior__(self, other: Any) -> "WindowSet":
        return self.apply(operator.or_, other)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: table={self.table}>'


class WindowWrapper(WindowWrapperT):
    """Windowed table wrapper.

    A windowed table does not return concrete values when keys are
    accessed, instead :class:`WindowSet` is returned so that
    the values can be further reduced to the wanted time period.
    """
    ValueType: ClassVar[Type[WindowSet]] = WindowSet
    key_index: bool = False
    key_index_table: Optional[_Table] = None

    def __init__(self, table: _Table, *, relative_to: Optional[Any] = None, key_index: bool = False, key_index_table: Optional[_Table] = None) -> None:
        self.table: _Table = table
        self.key_index: bool = key_index
        self.key_index_table: Optional[_Table] = key_index_table
        if self.key_index and self.key_index_table is None:
            self.key_index_table = self.table.clone(name=f'{self.table.name}-key_index', value_type=int, key_type=self.table.key_type, window=None)
        self._get_relative_timestamp: Optional[Callable[[EventT], float]] = self._relative_handler(relative_to)

    def clone(self, relative_to: Any) -> "WindowWrapper":
        """Clone this table using a new time-relativity configuration."""
        return type(self)(table=self.table, relative_to=relative_to or self._get_relative_timestamp, key_index=self.key_index, key_index_table=self.key_index_table)

    @property
    def name(self) -> str:
        """Return the name of this table."""
        return self.table.name

    def relative_to(self, ts: Any) -> "WindowWrapper":
        """Configure the time-relativity of this windowed table."""
        return self.clone(relative_to=ts)

    def relative_to_now(self) -> "WindowWrapper":
        """Configure table to be time-relative to the system clock."""
        return self.clone(relative_to=self.table._relative_now)

    def relative_to_field(self, field: FieldDescriptorT) -> "WindowWrapper":
        """Configure table to be time-relative to a field in the stream.

        This means the window will use the timestamp
        from the event currently being processed in the stream.

        Further it will not use the timestamp of the Kafka message,
        but a field in the value of the event.

        For example a model field:

        .. sourcecode:: python

            class Account(faust.Record):
                created: float

            table = app.Table('foo').hopping(
                ...,
            ).relative_to_field(Account.created)
        """
        return self.clone(relative_to=self.table._relative_field(field))

    def relative_to_stream(self) -> "WindowWrapper":
        """Configure table to be time-relative to the stream.

        This means the window will use the timestamp
        from the event currently being processed in the stream.
        """
        return self.clone(relative_to=self.table._relative_event)

    def get_timestamp(self, event: Optional[EventT] = None) -> float:
        """Get timestamp from event."""
        event = event or current_event()
        get_relative_timestamp: Optional[Callable[[EventT], float]] = self.get_relative_timestamp
        if get_relative_timestamp:
            timestamp = get_relative_timestamp(event)
            if isinstance(timestamp, datetime):
                return timestamp.timestamp()
            return timestamp
        if event is None:
            raise RuntimeError('Operation outside of stream iteration')
        return event.message.timestamp

    def on_recover(self, fun: Callable[..., Any]) -> RecoverCallback:
        """Call after table recovery."""
        return self.table.on_recover(fun)

    def __contains__(self, key: KT) -> bool:
        return self.table._windowed_contains(key, self.get_timestamp())

    def __getitem__(self, key: KT) -> WindowSet:
        return self.ValueType(key, self.table, self, current_event())

    def __setitem__(self, key: KT, value: VT) -> None:
        if not isinstance(value, WindowSet):
            table: _Table = cast(_Table, self.table)
            self.on_set_key(key, value)
            table._set_windowed(key, value, self.get_timestamp())

    def on_set_key(self, key: KT, value: VT) -> None:
        """Call when the value for a key in this table is set."""
        key_index_table: Optional[_Table] = self.key_index_table
        if key_index_table is not None:
            if key not in key_index_table:
                key_index_table[key] = 1  # type: ignore

    def on_del_key(self, key: KT) -> None:
        """Call when a key is deleted from this table."""
        key_index_table: Optional[_Table] = self.key_index_table
        if key_index_table is not None:
            key_index_table.pop(key, None)

    def __delitem__(self, key: KT) -> None:
        self.on_del_key(key)
        cast(_Table, self.table)._del_windowed(key, self.get_timestamp())

    def __len__(self) -> int:
        if self.key_index_table is not None:
            return len(self.key_index_table)
        raise NotImplementedError('Windowed table must use_index=True to support len()')

    def _relative_handler(self, relative_to: Any) -> Optional[Callable[[EventT], float]]:
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
        """Return table keys view: iterate over keys found in this table."""
        return WindowedKeysView(self)

    def _keys(self) -> Iterator[KT]:
        key_index_table: Optional[_Table] = self.key_index_table
        if key_index_table is not None:
            for key in key_index_table.keys():
                yield key
        else:
            raise NotImplementedError('Windowed table must set use_index=True to support .keys/.items/.values')

    def values(self, event: Optional[EventT] = None) -> WindowedValuesView:
        """Return table values view: iterate over values in this table."""
        return WindowedValuesView(self, event or current_event())

    def items(self, event: Optional[EventT] = None) -> WindowedItemsView:
        """Return table items view: iterate over ``(key, value)`` pairs."""
        return WindowedItemsView(self, event or current_event())

    def _items(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table: _Table = cast(_Table, self.table)
        timestamp: float = self.get_timestamp(event)
        for key in self._keys():
            try:
                yield (key, table._windowed_timestamp(key, timestamp))
            except KeyError:
                pass

    def _items_now(self) -> Iterator[Tuple[KT, VT]]:
        table: _Table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield (key, table._windowed_now(key))
            except KeyError:
                pass

    def _items_current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table: _Table = cast(_Table, self.table)
        timestamp: float = table._relative_event(event)
        for key in self._keys():
            try:
                yield (key, table._windowed_timestamp(key, timestamp))
            except KeyError:
                pass

    def _items_delta(self, d: Seconds, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]:
        table: _Table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield (key, table._windowed_delta(key, d, event))
            except KeyError:
                pass

    def as_ansitable(self, title: str = '{table.name}', **kwargs: Any) -> Any:
        """Draw table as a terminal ANSI table."""
        return dict_as_ansitable(self, title=title.format(table=self.table), **kwargs)

    @property
    def get_relative_timestamp(self) -> Optional[Callable[[EventT], float]]:
        """Return the current handler for extracting event timestamp."""
        return self._get_relative_timestamp

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Any) -> None:
        self._get_relative_timestamp = self._relative_handler(relative_to)
