"""Base class for table storage drivers."""
import abc
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union, cast, Dict, ClassVar
from mode import Service
from yarl import URL
from faust.types import AppT, CodecArg, CollectionT, EventT, ModelArg, StoreT, TP
from faust.types.stores import KT, VT
__all__ = ['Store', 'SerializedStore']

class Store(StoreT[KT, VT], Service):
    """Base class for table storage drivers."""

    url: URL
    app: AppT
    table: CollectionT
    table_name: str
    key_type: Optional[ModelArg]
    value_type: Optional[ModelArg]
    key_serializer: Optional[CodecArg]
    value_serializer: Optional[CodecArg]
    options: Optional[Dict[str, Any]]

    def __init__(self, 
                 url: Union[str, URL], 
                 app: AppT, 
                 table: CollectionT, 
                 *, 
                 table_name: str = '', 
                 key_type: Optional[ModelArg] = None, 
                 value_type: Optional[ModelArg] = None, 
                 key_serializer: Optional[CodecArg] = None, 
                 value_serializer: Optional[CodecArg] = None, 
                 options: Optional[Dict[str, Any]] = None, 
                 **kwargs: Any) -> None:
        Service.__init__(self, **kwargs)
        self.url = URL(url)
        self.app = app
        self.table = table
        self.table_name = table_name or self.table.name
        self.key_type = key_type
        self.value_type = value_type
        self.key_serializer = key_serializer
        self.value_serializer = value_serializer
        self.options = options

    def __hash__(self) -> int:
        return object.__hash__(self)

    def persisted_offset(self, tp: TP) -> Optional[int]:
        """Return the persisted offset for this topic and partition."""
        raise NotImplementedError('In-memory store only, does not persist.')

    def set_persisted_offset(self, tp: TP, offset: int) -> None:
        """Set the persisted offset for this topic and partition."""
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        """Return :const:`True` if we have a copy of standby from elsewhere."""
        return True

    async def on_rebalance(self, 
                           table: CollectionT, 
                           assigned: Set[TP], 
                           revoked: Set[TP], 
                           newly_assigned: Set[TP]) -> None:
        """Handle rebalancing of the cluster."""
        ...

    async def on_recovery_completed(self, 
                                    active_tps: Set[TP], 
                                    standby_tps: Set[TP]) -> None:
        """Signal that table recovery completed."""
        ...

    def _encode_key(self, key: KT) -> bytes:
        key_bytes = self.app.serializers.dumps_key(self.key_type, key, serializer=self.key_serializer)
        if key_bytes is None:
            raise TypeError('Table key cannot be None')
        return key_bytes

    def _encode_value(self, value: VT) -> Optional[bytes]:
        return self.app.serializers.dumps_value(self.value_type, value, serializer=self.value_serializer)

    def _decode_key(self, key: bytes) -> KT:
        return cast(KT, self.app.serializers.loads_key(self.key_type, key, serializer=self.key_serializer))

    def _decode_value(self, value: bytes) -> VT:
        return self.app.serializers.loads_value(self.value_type, value, serializer=self.value_serializer)

    def _repr_info(self) -> str:
        return f'table_name={self.table_name} url={self.url}'

    @property
    def label(self) -> str:
        """Return short description of this store."""
        return f'{type(self).__name__}: {self.url}'

class _SerializedStoreKeysView(KeysView[KT]):

    def __init__(self, store: 'SerializedStore[KT, VT]') -> None:
        self._mapping = store

    def __iter__(self) -> Iterator[KT]:
        yield from self._mapping._keys_decoded()

class _SerializedStoreValuesView(ValuesView[VT]):

    def __init__(self, store: 'SerializedStore[KT, VT]') -> None:
        self._mapping = store

    def __iter__(self) -> Iterator[VT]:
        yield from self._mapping._values_decoded()

class _SerializedStoreItemsView(ItemsView[KT, VT]):

    def __init__(self, store: 'SerializedStore[KT, VT]') -> None:
        self._mapping = store

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        yield from self._mapping._items_decoded()

class SerializedStore(Store[KT, VT]):
    """Base class for table storage drivers requiring serialization."""

    @abc.abstractmethod
    def _get(self, key: bytes) -> Optional[bytes]:
        ...

    @abc.abstractmethod
    def _set(self, key: bytes, value: bytes) -> None:
        ...

    @abc.abstractmethod
    def _del(self, key: bytes) -> None:
        ...

    @abc.abstractmethod
    def _iterkeys(self) -> Iterator[bytes]:
        ...

    @abc.abstractmethod
    def _itervalues(self) -> Iterator[bytes]:
        ...

    @abc.abstractmethod
    def _iteritems(self) -> Iterator[Tuple[bytes, bytes]]:
        ...

    @abc.abstractmethod
    def _size(self) -> int:
        ...

    @abc.abstractmethod
    def _contains(self, key: bytes) -> bool:
        ...

    @abc.abstractmethod
    def _clear(self) -> None:
        ...

    def apply_changelog_batch(self, 
                              batch: Iterable[EventT], 
                              to_key: Callable[[Any], KT], 
                              to_value: Callable[[Any], VT]) -> None:
        """Apply batch of events from changelog topic to this store."""
        for event in batch:
            key = event.message.key
            if key is None:
                raise TypeError(f'Changelog entry is missing key: {event.message}')
            value = event.message.value
            if value is None:
                self._del(key)
            else:
                self._set(key, value)

    def __getitem__(self, key: KT) -> VT:
        value = self._get(self._encode_key(key))
        if value is None:
            raise KeyError(key)
        return self._decode_value(value)

    def __setitem__(self, key: KT, value: VT) -> None:
        return self._set(self._encode_key(key), self._encode_value(value))

    def __delitem__(self, key: KT) -> None:
        return self._del(self._encode_key(key))

    def __iter__(self) -> Iterator[KT]:
        yield from self._keys_decoded()

    def __len__(self) -> int:
        return self._size()

    def __contains__(self, key: object) -> bool:
        return self._contains(self._encode_key(cast(KT, key)))

    def keys(self) -> KeysView[KT]:
        """Return view of keys in the K/V store."""
        return _SerializedStoreKeysView(self)

    def _keys_decoded(self) -> Iterator[KT]:
        for key in self._iterkeys():
            yield self._decode_key(key)

    def values(self) -> ValuesView[VT]:
        """Return view of values in the K/V store."""
        return _SerializedStoreValuesView(self)

    def _values_decoded(self) -> Iterator[VT]:
        for value in self._itervalues():
            yield self._decode_value(value)

    def items(self) -> ItemsView[KT, VT]:
        """Return view of items in the K/V store as (key, value) pairs."""
        return _SerializedStoreItemsView(self)

    def _items_decoded(self) -> Iterator[Tuple[KT, VT]]:
        for key, value in self._iteritems():
            yield (self._decode_key(key), self._decode_value(value))

    def clear(self) -> None:
        """Clear all data from this K/V store."""
        self._clear()
