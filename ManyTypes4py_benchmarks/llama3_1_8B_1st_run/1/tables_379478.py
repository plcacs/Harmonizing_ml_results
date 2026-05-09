import abc
import typing
from datetime import datetime
from typing import Any, Awaitable, Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping, Optional, Set, Tuple, TypeVar, Union, ValuesView
from mode import Seconds, ServiceT
from mode.utils.collections import FastUserDict, ManagedUserDict
from yarl import URL
from .codecs import CodecArg
from .events import EventT
from .stores import StoreT
from .streams import JoinableT
from .topics import TopicT
from .tuples import FutureMessage, TP
from .windows import WindowT
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import FieldDescriptorT as _FieldDescriptorT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
else:

    class _AppT:
        ...

    class _FieldDescriptorT:
        ...

    class _ModelArg:
        ...

    class _SchemaT:
        ...
__all__ = ['RecoverCallback', 'RelativeArg', 'CollectionT', 'TableT', 'GlobalTableT', 'TableManagerT', 'WindowCloseCallback', 'WindowSetT', 'WindowedItemsViewT', 'WindowedValuesViewT', 'WindowWrapperT', 'ChangelogEventCallback', 'CollectionTps']
RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]
KT = TypeVar('KT')
VT = TypeVar('VT')

class CollectionT(ServiceT, JoinableT):
    is_global: bool

    def __init__(self, app: _AppT, *, name: Optional[str] = None, default: Optional[Any] = None, store: StoreT = None, schema: Optional[_SchemaT] = None, key_type: Optional[type] = None, value_type: Optional[type] = None, partitions: Optional[int] = None, window: WindowT = None, changelog_topic: TopicT = None, help: Optional[str] = None, on_recover: Optional[RecoverCallback] = None, on_changelog_event: Optional[ChangelogEventCallback] = None, recovery_buffer_size: int = 1000, standby_buffer_size: Optional[int] = None, extra_topic_configs: Optional[Mapping[str, Any]] = None, options: Optional[Mapping[str, Any]] = None, use_partitioner: bool = False, on_window_close: Optional[WindowCloseCallback] = None, **kwargs: Any) -> None:
        ...

    def clone(self, **kwargs: Any) -> 'CollectionT':
        ...

    @property
    def changelog_topic(self) -> TopicT:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None:
        ...

    def _changelog_topic_name(self) -> str:
        ...

    def apply_changelog_batch(self, batch: Iterable[FutureMessage]) -> None:
        ...

    def persisted_offset(self, tp: TP) -> int:
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    def reset_state(self) -> None:
        ...

    def send_changelog(self, partition: int, key: KT, value: VT, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None) -> None:
        ...

    def partition_for_key(self, key: KT) -> int:
        ...

    async def on_window_close(self, key: KT, value: VT) -> None:
        ...

    async def on_rebalance(self, assigned: Iterable[KT], revoked: Iterable[KT], newly_assigned: Iterable[KT]) -> None:
        ...

    async def on_changelog_event(self, event: EventT) -> None:
        ...

    def on_recover(self, fun: RecoverCallback) -> RecoverCallback:
        ...

    async def on_recovery_completed(self, active_tps: Mapping[KT, int], standby_tps: Mapping[KT, int]) -> None:
        ...

    async def call_recover_callbacks(self) -> None:
        ...

    def using_window(self, window: WindowT, *, key_index: bool = False) -> 'CollectionT':
        ...

    def hopping(self, size: int, step: int, expires: Optional[datetime] = None, key_index: bool = False) -> 'CollectionT':
        ...

    def tumbling(self, size: int, expires: Optional[datetime] = None, key_index: bool = False) -> 'CollectionT':
        ...

    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    def _relative_now(self, event: Optional[EventT] = None) -> datetime:
        ...

    def _relative_event(self, event: Optional[EventT] = None) -> datetime:
        ...

    def _relative_field(self, field: FieldDescriptorT) -> datetime:
        ...

    def _relative_timestamp(self, timestamp: datetime) -> datetime:
        ...

    def _windowed_contains(self, key: KT, timestamp: datetime) -> bool:
        ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):

    def __init__(self, app: _AppT, **kwargs: Any) -> None:
        ...

    def add(self, table: CollectionT) -> None:
        ...

    def persist_offset_on_commit(self, store: StoreT, tp: TP, offset: int) -> None:
        ...

    def on_commit(self, offsets: Mapping[KT, int]) -> None:
        ...

    async def on_rebalance(self, assigned: Iterable[str], revoked: Iterable[str], newly_assigned: Iterable[str]) -> None:
        ...

    def on_partitions_revoked(self, revoked: Iterable[str]) -> None:
        ...

    def on_rebalance_start(self) -> None:
        ...

    async def wait_until_tables_registered(self) -> None:
        ...

    async def wait_until_recovery_completed(self) -> None:
        ...

    @property
    def changelog_topics(self) -> Mapping[str, TopicT]:
        ...

class WindowSetT(FastUserDict[KT, VT]):

    def __init__(self, key: KT, table: TableT, wrapper: WindowWrapperT, event: Optional[EventT] = None) -> None:
        ...

    def apply(self, op: str, value: VT, event: Optional[EventT] = None) -> None:
        ...

    def value(self, event: Optional[EventT] = None) -> VT:
        ...

    def current(self, event: Optional[EventT] = None) -> VT:
        ...

    def now(self) -> datetime:
        ...

    def delta(self, d: datetime, event: Optional[EventT] = None) -> VT:
        ...

    def __iadd__(self, other: VT) -> 'WindowSetT':
        ...

    def __isub__(self, other: VT) -> 'WindowSetT':
        ...

    def __imul__(self, other: VT) -> 'WindowSetT':
        ...

    def __itruediv__(self, other: VT) -> 'WindowSetT':
        ...

    def __ifloordiv__(self, other: VT) -> 'WindowSetT':
        ...

    def __imod__(self, other: VT) -> 'WindowSetT':
        ...

    def __ipow__(self, other: VT) -> 'WindowSetT':
        ...

    def __ilshift__(self, other: VT) -> 'WindowSetT':
        ...

    def __irshift__(self, other: VT) -> 'WindowSetT':
        ...

    def __iand__(self, other: VT) -> 'WindowSetT':
        ...

    def __ixor__(self, other: VT) -> 'WindowSetT':
        ...

    def __ior__(self, other: VT) -> 'WindowSetT':
        ...

class WindowedItemsViewT(ItemsView):

    def __init__(self, mapping: Mapping[KT, VT], event: Optional[EventT] = None) -> None:
        ...

    def __iter__(self) -> Iterator[KT]:
        ...

    def now(self) -> datetime:
        ...

    def current(self, event: Optional[EventT] = None) -> KT:
        ...

    def delta(self, d: datetime, event: Optional[EventT] = None) -> KT:
        ...

class WindowedValuesViewT(ValuesView):

    def __init__(self, mapping: Mapping[KT, VT], event: Optional[EventT] = None) -> None:
        ...

    def __iter__(self) -> Iterator[VT]:
        ...

    def now(self) -> datetime:
        ...

    def current(self, event: Optional[EventT] = None) -> VT:
        ...

    def delta(self, d: datetime, event: Optional[EventT] = None) -> VT:
        ...

class WindowWrapperT(MutableMapping):

    def __init__(self, table: TableT, *, relative_to: Optional[WindowT] = None, key_index: bool = False, key_index_table: Optional[TableT] = None) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    def clone(self, relative_to: WindowT) -> 'WindowWrapperT':
        ...

    def relative_to_now(self) -> datetime:
        ...

    def relative_to_field(self, field: FieldDescriptorT) -> datetime:
        ...

    def relative_to_stream(self) -> datetime:
        ...

    def get_timestamp(self, event: Optional[EventT] = None) -> datetime:
        ...

    def __getitem__(self, key: KT) -> VT:
        ...

    def keys(self) -> KeysView[KT]:
        ...

    def on_set_key(self, key: KT, value: VT) -> None:
        ...

    def on_del_key(self, key: KT) -> None:
        ...

    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    @property
    def get_relative_timestamp(self) -> Callable[[], datetime]:
        ...
