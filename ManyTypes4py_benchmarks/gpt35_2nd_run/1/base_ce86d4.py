from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union
from yarl import URL
from faust.types import AppT, CodecArg, CollectionT, EventT, ModelArg, StoreT, TP
from faust.types.stores import KT, VT

class Store(StoreT[KT, VT], Service):
    def __init__(self, url: str, app: AppT, table: Any, *, table_name: str = '', key_type: Optional[type] = None, value_type: Optional[type] = None, key_serializer: Optional[Callable] = None, value_serializer: Optional[Callable] = None, options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    def persisted_offset(self, tp: TP) -> Any:
    def set_persisted_offset(self, tp: TP, offset: Any) -> None:
    async def need_active_standby_for(self, tp: TP) -> bool:
    async def on_rebalance(self, table: Any, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
    async def on_recovery_completed(self, active_tps: Set[TP], standby_tps: Set[TP]) -> None:
    def _encode_key(self, key: KT) -> bytes:
    def _encode_value(self, value: VT) -> bytes:
    def _decode_key(self, key: bytes) -> KT:
    def _decode_value(self, value: bytes) -> VT:
    def _repr_info(self) -> str:
    @property
    def label(self) -> str:

class SerializedStore(Store[KT, VT]):
    def _get(self, key: KT) -> Any:
    def _set(self, key: KT, value: VT) -> None:
    def _del(self, key: KT) -> None:
    def _iterkeys(self) -> Iterator[KT]:
    def _itervalues(self) -> Iterator[VT]:
    def _iteritems(self) -> Iterator[Tuple[KT, VT]]:
    def _size(self) -> int:
    def _contains(self, key: KT) -> bool:
    def _clear(self) -> None:
    def apply_changelog_batch(self, batch: Iterable[EventT], to_key: Callable[[EventT], KT], to_value: Callable[[EventT], VT]) -> None:
    def __getitem__(self, key: Any) -> VT:
    def __setitem__(self, key: Any, value: Any) -> None:
    def __delitem__(self, key: Any) -> None:
    def __len__(self) -> int:
    def keys(self) -> Iterable[KT]:
    def values(self) -> Iterable[VT]:
    def items(self) -> Iterable[Tuple[KT, VT]]:
    def clear(self) -> None:
