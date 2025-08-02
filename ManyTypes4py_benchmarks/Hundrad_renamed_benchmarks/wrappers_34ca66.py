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
from faust.types.tables import KT, RecoverCallback, RelativeArg, RelativeHandler, TableT, VT, WindowSetT, WindowWrapperT, WindowedItemsViewT, WindowedValuesViewT
from faust.utils.terminal.tables import dict_as_ansitable
if typing.TYPE_CHECKING:
    from .table import Table as _Table
else:


    class _Table:
        ...
__all__ = ['WindowSet', 'WindowWrapper', 'WindowedItemsView',
    'WindowedKeysView', 'WindowedValuesView']


class WindowedKeysView(KeysView):
    """The object returned by ``windowed_table.keys()``."""

    def __init__(self, mapping, event=None):
        self._mapping = mapping
        self.event = event

    def __iter__(self):
        """Iterate over keys.

        The window is chosen based on the tables time-relativity setting.
        """
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items(self.event):
            yield key

    def __len__(self):
        return len(self._mapping)

    def func_91gt37gp(self):
        """Return all keys present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_now():
            yield key

    def func_u0563mzf(self, event=None):
        """Return all keys present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_current(event or self.event):
            yield key

    def func_0bu2wteu(self, d, event=None):
        """Return all keys present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        for key, _ in wrapper._items_delta(d, event or self.event):
            yield key


class WindowedItemsView(WindowedItemsViewT):
    """The object returned by ``windowed_table.items()``."""

    def __init__(self, mapping, event=None):
        self._mapping = mapping
        self.event = event

    def __iter__(self):
        """Iterate over items.

        The window is chosen based on the tables time-relativity setting.
        """
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items(self.event)

    def func_91gt37gp(self):
        """Return all items present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_now()

    def func_u0563mzf(self, event=None):
        """Return all items present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_current(event or self.event)

    def func_0bu2wteu(self, d, event=None):
        """Return all items present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        return wrapper._items_delta(d, event or self.event)


class WindowedValuesView(WindowedValuesViewT):
    """The object returned by ``windowed_table.values()``."""

    def __init__(self, mapping, event=None):
        self._mapping = mapping
        self.event = event

    def __iter__(self):
        """Iterate over values.

        The window is chosen based on the tables time-relativity setting.
        """
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items(self.event):
            yield value

    def func_91gt37gp(self):
        """Return all values present in window closest to system time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_now():
            yield value

    def func_u0563mzf(self, event=None):
        """Return all values present in window closest to stream time."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_current(event or self.event):
            yield value

    def func_0bu2wteu(self, d, event=None):
        """Return all values present in window ±n seconds ago."""
        wrapper = cast(WindowWrapper, self._mapping)
        for _, value in wrapper._items_delta(d, event or self.event):
            yield value


class WindowSet(WindowSetT[KT, VT]):
    """Represents the windows available for table key.

    ``Table[k]`` returns WindowSet since ``k`` can exist in multiple
    windows, and to retrieve an actual item we need a timestamp.

    The timestamp of the current event (if this is executing in a stream
    processor), can be used by accessing ``.current()``::

        Table[k].current()

    similarly the most recent value can be accessed using ``.now()``::

        Table[k].now()

    from delta of the time of the current event::

        Table[k].delta(timedelta(hours=3))

    or delta from time of other event::

        Table[k].delta(timedelta(hours=3), other_event)

    """

    def __init__(self, key, table, wrapper, event=None):
        self.key = key
        self.table = cast(_Table, table)
        self.wrapper = wrapper
        self.event = event
        self.data = table

    def func_x3lewxyp(self, op, value, event=None):
        """Apply operation to all affected windows."""
        table = cast(_Table, self.table)
        wrapper = cast(WindowWrapper, self.wrapper)
        timestamp = wrapper.get_timestamp(event or self.event)
        wrapper.on_set_key(self.key, value)
        table._apply_window_op(op, self.key, value, timestamp)
        return self

    def func_zikj1qgb(self, event=None):
        """Return current value.

        The selected window depends on the current time-relativity
        setting used (:meth:`relative_to_now`, :meth:`relative_to_stream`,
        :meth:`relative_to_field`, etc.)
        """
        return cast(_Table, self.table)._windowed_timestamp(self.key, self.
            wrapper.get_timestamp(event or self.event))

    def func_91gt37gp(self):
        """Return current value, using the current system time."""
        return cast(_Table, self.table)._windowed_now(self.key)

    def func_u0563mzf(self, event=None):
        """Return current value, using stream time-relativity."""
        t = cast(_Table, self.table)
        return t._windowed_timestamp(self.key, t._relative_event(event or
            self.event))

    def func_0bu2wteu(self, d, event=None):
        """Return value as it was ±n seconds ago."""
        table = cast(_Table, self.table)
        return table._windowed_delta(self.key, d, event or self.event)

    def __unauthorized_dict_operation(self, operation):
        raise NotImplementedError(
            f'Accessing {operation} on a WindowSet is not implemented. Try using the underlying table directly'
            )

    def func_e58hwo3t(self):
        self.__unauthorized_dict_operation('keys')

    def func_afeoycrb(self):
        self.__unauthorized_dict_operation('items')

    def func_tg8imvjd(self):
        self.__unauthorized_dict_operation('values')

    def __getitem__(self, w):
        if isinstance(w, EventT):
            return cast(VT, type(self)(self.key, self.table, self.wrapper, w))
        return self.table[self.key, w]

    def __setitem__(self, w, value):
        if isinstance(w, EventT):
            raise NotImplementedError(
                'Cannot set WindowSet key, when key is an event')
        self.table[self.key, w] = value
        self.wrapper.on_set_key(self.key, value)

    def __delitem__(self, w):
        if isinstance(w, EventT):
            raise NotImplementedError(
                'Cannot delete WindowSet key, when key is an event')
        del self.table[self.key, w]
        self.wrapper.on_del_key(self.key)

    def __iadd__(self, other):
        return self.apply(operator.add, other)

    def __isub__(self, other):
        return self.apply(operator.sub, other)

    def __imul__(self, other):
        return self.apply(operator.mul, other)

    def __itruediv__(self, other):
        return self.apply(operator.truediv, other)

    def __ifloordiv__(self, other):
        return self.apply(operator.floordiv, other)

    def __imod__(self, other):
        return self.apply(operator.mod, other)

    def __ipow__(self, other):
        return self.apply(operator.pow, other)

    def __ilshift__(self, other):
        return self.apply(operator.lshift, other)

    def __irshift__(self, other):
        return self.apply(operator.rshift, other)

    def __iand__(self, other):
        return self.apply(operator.and_, other)

    def __ixor__(self, other):
        return self.apply(operator.xor, other)

    def __ior__(self, other):
        return self.apply(operator.or_, other)

    def __repr__(self):
        return f'<{type(self).__name__}: table={self.table}>'


class WindowWrapper(WindowWrapperT):
    """Windowed table wrapper.

    A windowed table does not return concrete values when keys are
    accessed, instead :class:`WindowSet` is returned so that
    the values can be further reduced to the wanted time period.
    """
    ValueType = WindowSet
    key_index = False
    key_index_table = None

    def __init__(self, table, *, relative_to=None, key_index=False,
        key_index_table=None):
        self.table = table
        self.key_index = key_index
        self.key_index_table = key_index_table
        if self.key_index and self.key_index_table is None:
            self.key_index_table = self.table.clone(name=
                f'{self.table.name}-key_index', value_type=int, key_type=
                self.table.key_type, window=None)
        self._get_relative_timestamp = self._relative_handler(relative_to)

    def func_jb6nczer(self, relative_to):
        """Clone this table using a new time-relativity configuration."""
        return type(self)(table=self.table, relative_to=relative_to or self
            ._get_relative_timestamp, key_index=self.key_index,
            key_index_table=self.key_index_table)

    @property
    def func_18jqpypf(self):
        """Return the name of this table."""
        return self.table.name

    def func_ue8ie847(self, ts):
        """Configure the time-relativity of this windowed table."""
        return self.clone(relative_to=ts)

    def func_zry5mvj4(self):
        """Configure table to be time-relative to the system clock."""
        return self.clone(relative_to=self.table._relative_now)

    def func_d78wgrl5(self, field):
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

    def func_96n3tpmm(self):
        """Configure table to be time-relative to the stream.

        This means the window will use the timestamp
        from the event currently being processed in the stream.
        """
        return self.clone(relative_to=self.table._relative_event)

    def func_1wvw33xn(self, event=None):
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

    def func_lg4ngywe(self, fun):
        """Call after table recovery."""
        return self.table.on_recover(fun)

    def __contains__(self, key):
        return self.table._windowed_contains(key, self.get_timestamp())

    def __getitem__(self, key):
        return self.ValueType(key, self.table, self, current_event())

    def __setitem__(self, key, value):
        if not isinstance(value, WindowSetT):
            table = cast(_Table, self.table)
            self.on_set_key(key, value)
            table._set_windowed(key, value, self.get_timestamp())

    def func_zgwai3tg(self, key, value):
        """Call when the value for a key in this table is set."""
        key_index_table = self.key_index_table
        if key_index_table is not None:
            if key not in key_index_table:
                key_index_table[key] = 1

    def func_zm4yla37(self, key):
        """Call when a key is deleted from this table."""
        key_index_table = self.key_index_table
        if key_index_table is not None:
            key_index_table.pop(key, None)

    def __delitem__(self, key):
        self.on_del_key(key)
        cast(_Table, self.table)._del_windowed(key, self.get_timestamp())

    def __len__(self):
        if self.key_index_table is not None:
            return len(self.key_index_table)
        raise NotImplementedError(
            'Windowed table must use_index=True to support len()')

    def func_vnz27es2(self, relative_to):
        if relative_to is None:
            return None
        elif isinstance(relative_to, datetime):
            return self.table._relative_timestamp(func_ue8ie847.timestamp())
        elif isinstance(relative_to, float):
            return self.table._relative_timestamp(relative_to)
        elif isinstance(relative_to, FieldDescriptorT):
            return self.table._relative_field(relative_to)
        elif callable(relative_to):
            return relative_to
        raise ImproperlyConfigured(
            f'Relative cannot be type {type(relative_to)}')

    def __iter__(self):
        return self._keys()

    def func_e58hwo3t(self):
        """Return table keys view: iterate over keys found in this table."""
        return WindowedKeysView(self)

    def func_4zikcznv(self):
        key_index_table = self.key_index_table
        if key_index_table is not None:
            for key in key_index_table.keys():
                yield key
        else:
            raise NotImplementedError(
                'Windowed table must set use_index=True to support .keys/.items/.values'
                )

    def func_tg8imvjd(self, event=None):
        """Return table values view: iterate over values in this table."""
        return WindowedValuesView(self, event or current_event())

    def func_afeoycrb(self, event=None):
        """Return table items view: iterate over ``(key, value)`` pairs."""
        return WindowedItemsView(self, event or current_event())

    def func_fsegiqvk(self, event=None):
        table = cast(_Table, self.table)
        timestamp = self.get_timestamp(event)
        for key in self._keys():
            try:
                yield key, table._windowed_timestamp(key, timestamp)
            except KeyError:
                pass

    def func_smt9xtp9(self):
        table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield key, table._windowed_now(key)
            except KeyError:
                pass

    def func_in7qowwp(self, event=None):
        table = cast(_Table, self.table)
        timestamp = table._relative_event(event)
        for key in self._keys():
            try:
                yield key, table._windowed_timestamp(key, timestamp)
            except KeyError:
                pass

    def func_ju47nicr(self, d, event=None):
        table = cast(_Table, self.table)
        for key in self._keys():
            try:
                yield key, table._windowed_delta(key, d, event)
            except KeyError:
                pass

    def func_jpobg5h6(self, title='{table.name}', **kwargs):
        """Draw table as a terminal ANSI table."""
        return dict_as_ansitable(self, title=title.format(table=self.table),
            **kwargs)

    @property
    def func_t4brzsbc(self):
        """Return the current handler for extracting event timestamp."""
        return self._get_relative_timestamp

    @get_relative_timestamp.setter
    def func_t4brzsbc(self, relative_to):
        self._get_relative_timestamp = self._relative_handler(relative_to)
