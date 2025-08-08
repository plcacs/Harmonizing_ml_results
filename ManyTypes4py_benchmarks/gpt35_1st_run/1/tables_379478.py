from typing import Any, Awaitable, Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping, Optional, Set, Tuple, TypeVar, Union, ValuesView
from mode import Seconds, ServiceT
from mode.utils.collections import FastUserDict, ManagedUserDict
from yarl import URL

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]
RelativeArg = Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]
KT = TypeVar('KT')
VT = TypeVar('VT')

class CollectionT(ServiceT, JoinableT):
    is_global: bool = False

    def __init__(self, app: '_AppT', *, name: Optional[str] = None, default: Any = None, store: Any = None, schema: Any = None, key_type: Any = None, value_type: Any = None, partitions: Any = None, window: Any = None, changelog_topic: Any = None, help: Any = None, on_recover: Any = None, on_changelog_event: Any = None, recovery_buffer_size: int = 1000, standby_buffer_size: Optional[int] = None, extra_topic_configs: Any = None, options: Any = None, use_partitioner: bool = False, on_window_close: Any = None, **kwargs: Any):
        ...

    def clone(self, **kwargs: Any) -> Any:
        ...

    @property
    def changelog_topic(self) -> Any:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic: Any) -> None:
        ...

    def _changelog_topic_name(self) -> Any:
        ...

    def apply_changelog_batch(self, batch: Any) -> None:
        ...

    def persisted_offset(self, tp: Any) -> Any:
        ...

    async def need_active_standby_for(self, tp: Any) -> Any:
        ...

    def reset_state(self) -> None:
        ...

    def send_changelog(self, partition: Any, key: Any, value: Any, key_serializer: Any = None, value_serializer: Any = None) -> None:
        ...

    def partition_for_key(self, key: Any) -> Any:
        ...

    async def on_window_close(self, key: Any, value: Any) -> Any:
        ...

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> Any:
        ...

    async def on_changelog_event(self, event: Any) -> Any:
        ...

    def on_recover(self, fun: Any) -> Any:
        ...

    async def on_recovery_completed(self, active_tps: Any, standby_tps: Any) -> Any:
        ...

    async def call_recover_callbacks(self) -> Any:
        ...

    def using_window(self, window: Any, *, key_index: bool = False) -> Any:
        ...

    def hopping(self, size: Any, step: Any, expires: Any = None, key_index: bool = False) -> Any:
        ...

    def tumbling(self, size: Any, expires: Any = None, key_index: bool = False) -> Any:
        ...

    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    def _relative_now(self, event: Any = None) -> Any:
        ...

    def _relative_event(self, event: Any = None) -> Any:
        ...

    def _relative_field(self, field: Any) -> Any:
        ...

    def _relative_timestamp(self, timestamp: Any) -> Any:
        ...

    def _windowed_contains(self, key: Any, timestamp: Any) -> Any:
        ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):

    def __init__(self, app: Any, **kwargs: Any) -> None:
        ...

    def add(self, table: Any) -> None:
        ...

    def persist_offset_on_commit(self, store: Any, tp: Any, offset: Any) -> None:
        ...

    def on_commit(self, offsets: Any) -> None:
        ...

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> Any:
        ...

    def on_partitions_revoked(self, revoked: Any) -> None:
        ...

    def on_rebalance_start(self) -> None:
        ...

    async def wait_until_tables_registered(self) -> Any:
        ...

    async def wait_until_recovery_completed(self) -> Any:
        ...

    @property
    def changelog_topics(self) -> Any:
        ...

class WindowSetT(FastUserDict[KT, VT]):

    def __init__(self, key: Any, table: Any, wrapper: Any, event: Any = None) -> None:
        ...

    def apply(self, op: Any, value: Any, event: Any = None) -> None:
        ...

    def value(self, event: Any = None) -> Any:
        ...

    def current(self, event: Any = None) -> Any:
        ...

    def now(self) -> Any:
        ...

    def delta(self, d: Any, event: Any = None) -> Any:
        ...

    def __iadd__(self, other: Any) -> Any:
        ...

    def __isub__(self, other: Any) -> Any:
        ...

    def __imul__(self, other: Any) -> Any:
        ...

    def __itruediv__(self, other: Any) -> Any:
        ...

    def __ifloordiv__(self, other: Any) -> Any:
        ...

    def __imod__(self, other: Any) -> Any:
        ...

    def __ipow__(self, other: Any) -> Any:
        ...

    def __ilshift__(self, other: Any) -> Any:
        ...

    def __irshift__(self, other: Any) -> Any:
        ...

    def __iand__(self, other: Any) -> Any:
        ...

    def __ixor__(self, other: Any) -> Any:
        ...

    def __ior__(self, other: Any) -> Any:
        ...

class WindowedItemsViewT(ItemsView):

    def __init__(self, mapping: Any, event: Any = None) -> None:
        ...

    def __iter__(self) -> Any:
        ...

    def now(self) -> Any:
        ...

    def current(self, event: Any = None) -> Any:
        ...

    def delta(self, d: Any, event: Any = None) -> Any:
        ...

class WindowedValuesViewT(ValuesView):

    def __init__(self, mapping: Any, event: Any = None) -> None:
        ...

    def __iter__(self) -> Any:
        ...

    def now(self) -> Any:
        ...

    def current(self, event: Any = None) -> Any:
        ...

    def delta(self, d: Any, event: Any = None) -> Any:
        ...

class WindowWrapperT(MutableMapping):

    def __init__(self, table: Any, *, relative_to: Any = None, key_index: bool = False, key_index_table: Any = None) -> None:
        ...

    @property
    def name(self) -> Any:
        ...

    def clone(self, relative_to: Any) -> Any:
        ...

    def relative_to_now(self) -> Any:
        ...

    def relative_to_field(self, field: Any) -> Any:
        ...

    def relative_to_stream(self) -> Any:
        ...

    def get_timestamp(self, event: Any = None) -> Any:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def keys(self) -> Any:
        ...

    def on_set_key(self, key: Any, value: Any) -> None:
        ...

    def on_del_key(self, key: Any) -> None:
        ...

    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    @property
    def get_relative_timestamp(self) -> Any:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Any) -> None:
        ...
