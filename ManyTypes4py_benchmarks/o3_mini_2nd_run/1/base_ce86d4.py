from collections.abc import ItemsView, KeysView, ValuesView, Iterator, Iterable
import abc
from typing import Any, Callable, Mapping, Optional, Tuple, Union, cast, TypeVar, Generic
from mode import Service
from yarl import URL
from faust.types import AppT, CodecArg, CollectionT, EventT, ModelArg, StoreT, TP
from faust.types.stores import KT, VT

__all__ = ['Store', 'SerializedStore']

class Store(StoreT[KT, VT], Service):
    def __init__(
        self,
        url: Union[str, URL],
        app: AppT,
        table: Any,
        *,
        table_name: str = '',
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        options: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        Service.__init__(self, **kwargs)
        self.url: URL = URL(url) if isinstance(url, str) else url
        self.app: AppT = app
        self.table: Any = table
        self.table_name: str = table_name or self.table.name
        self.key_type: Optional[Any] = key_type
        self.value_type: Optional[Any] = value_type
        self.key_serializer: Optional[CodecArg] = key_serializer
        self.value_serializer: Optional[CodecArg] = value_serializer
        self.options: Optional[Mapping[str, Any]] = options

    def __hash__(self) -> int:
        return object.__hash__(self)

    def persisted_offset(self, tp: TP) -> Any:
        raise NotImplementedError('In-memory store only, does not persist.')

    def set_persisted_offset(self, tp: TP, offset: Any) -> None:
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        return True

    async def on_rebalance(
        self, table: Any, assigned: Iterable[TP], revoked: Iterable[TP], newly_assigned: Iterable[TP]
    ) -> None:
        ...

    async def on_recovery_completed(
        self, active_tps: Iterable[TP], standby_tps: Iterable[TP]
    ) -> None:
        ...

    def _encode_key(self, key: KT) -> bytes:
        key_bytes: Optional[bytes] = self.app.serializers.dumps_key(self.key_type, key, serializer=self.key_serializer)
        if key_bytes is None:
            raise TypeError('Table key cannot be None')
        return key_bytes

    def _encode_value(self, value: VT) -> bytes:
        return self.app.serializers.dumps_value(self.value_type, value, serializer=self.value_serializer)

    def _decode_key(self, key: bytes) -> KT:
        return cast(KT, self.app.serializers.loads_key(self.key_type, key, serializer=self.key_serializer))

    def _decode_value(self, value: bytes) -> VT:
        return self.app.serializers.loads_value(self.value_type, value, serializer=self.value_serializer)

    def _repr_info(self) -> str:
        return f'table_name={self.table_name} url={self.url}'

    @property
    def label(self) -> str:
        return f'{type(self).__name__}: {self.url}'

class _SerializedStoreKeysView(KeysView, Generic[KT]):
    def __init__(self, store: "SerializedStore[KT, Any]") -> None:
        self._mapping: "SerializedStore[KT, Any]" = store

    def __iter__(self) -> Iterator[KT]:
        yield from self._mapping._keys_decoded()

class _SerializedStoreValuesView(ValuesView, Generic[VT]):
    def __init__(self, store: "SerializedStore[Any, VT]") -> None:
        self._mapping: "SerializedStore[Any, VT]" = store

    def __iter__(self) -> Iterator[VT]:
        yield from self._mapping._values_decoded()

class _SerializedStoreItemsView(ItemsView, Generic[KT, VT]):
    def __init__(self, store: "SerializedStore[KT, VT]") -> None:
        self._mapping: "SerializedStore[KT, VT]" = store

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        yield from self._mapping._items_decoded()

class SerializedStore(Store[KT, VT], abc.ABC):
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

    def apply_changelog_batch(
        self,
        batch: Iterable[EventT],
        to_key: Callable[[bytes], KT],
        to_value: Callable[[bytes], VT],
    ) -> None:
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
        value: Optional[bytes] = self._get(self._encode_key(key))
        if value is None:
            raise KeyError(key)
        return self._decode_value(value)

    def __setitem__(self, key: KT, value: VT) -> None:
        self._set(self._encode_key(key), self._encode_value(value))

    def __delitem__(self, key: KT) -> None:
        self._del(self._encode_key(key))

    def __iter__(self) -> Iterator[KT]:
        yield from self._keys_decoded()

    def __len__(self) -> int:
        return self._size()

    def __contains__(self, key: object) -> bool:
        return self._contains(self._encode_key(cast(KT, key)))

    def keys(self) -> KeysView[KT]:
        return _SerializedStoreKeysView(self)

    def _keys_decoded(self) -> Iterator[KT]:
        for key in self._iterkeys():
            yield self._decode_key(key)

    def values(self) -> ValuesView[VT]:
        return _SerializedStoreValuesView(self)

    def _values_decoded(self) -> Iterator[VT]:
        for value in self._itervalues():
            yield self._decode_value(value)

    def items(self) -> ItemsView[KT, VT]:
        return _SerializedStoreItemsView(self)

    def _items_decoded(self) -> Iterator[Tuple[KT, VT]]:
        for key, value in self._iteritems():
            yield (self._decode_key(key), self._decode_value(value))

    def clear(self) -> None:
        self._clear()