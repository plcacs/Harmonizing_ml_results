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

    def persisted_offset(self, tp: Any) -> None:
        ...

    async def need_active_standby_for(self, tp: Any) -> None:
        ...

    def reset_state(self) -> None:
        ...

    def send_changelog(self, partition: Any, key: Any, value: Any, key_serializer: Any = None, value_serializer: Any = None) -> None:
        ...

    def partition_for_key(self, key: Any) -> None:
        ...

    async def on_window_close(self, key: Any, value: Any) -> None:
        ...

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> None:
        ...

    async def on_changelog_event(self, event: Any) -> None:
        ...

    def on_recover(self, fun: Any) -> None:
        ...

    async def on_recovery_completed(self, active_tps: Any, standby_tps: Any) -> None:
        ...

    async def call_recover_callbacks(self) -> None:
        ...

    def using_window(self, window: Any, *, key_index: bool = False) -> None:
        ...

    def hopping(self, size: Any, step: Any, expires: Any = None, key_index: bool = False) -> None:
        ...

    def tumbling(self, size: Any, expires: Any = None, key_index: bool = False) -> None:
        ...

    def as_ansitable(self, **kwargs: Any) -> None:
        ...

    def _relative_now(self, event: Any = None) -> None:
        ...

    def _relative_event(self, event: Any = None) -> None:
        ...

    def _relative_field(self, field: Any) -> None:
        ...

    def _relative_timestamp(self, timestamp: Any) -> None:
        ...

    def _windowed_contains(self, key: Any, timestamp: Any) -> None:
        ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):

    def __init__(self, app: '_AppT', **kwargs: Any) -> None:
        ...

    def add(self, table: Any) -> None:
        ...

    def persist_offset_on_commit(self, store: Any, tp: Any, offset: Any) -> None:
        ...

    def on_commit(self, offsets: Any) -> None:
        ...

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> None:
        ...

    def on_partitions_revoked(self, revoked: Any) -> None:
        ...

    def on_rebalance_start(self) -> None:
        ...

    async def wait_until_tables_registered(self) -> None:
        ...

    async def wait_until_recovery_completed(self) -> None:
        ...

    @property
    def changelog_topics(self) -> Any:
        ...

class WindowSetT(FastUserDict[KT, VT]):

    def __init__(self, key: Any, table: Any, wrapper: Any, event: Any = None) -> None:
        ...

    def apply(self, op: Any, value: Any, event: Any = None) -> None:
        ...

    def value(self, event: Any = None) -> None:
        ...

    def current(self, event: Any = None) -> None:
        ...

    def now(self) -> None:
        ...

    def delta(self, d: Any, event: Any = None) -> None:
        ...

    def __iadd__(self, other: Any) -> None:
        ...

    def __isub__(self, other: Any) -> None:
        ...

    def __imul__(self, other: Any) -> None:
        ...

    def __itruediv__(self, other: Any) -> None:
        ...

    def __ifloordiv__(self, other: Any) -> None:
        ...

    def __imod__(self, other: Any) -> None:
        ...

    def __ipow__(self, other: Any) -> None:
        ...

    def __ilshift__(self, other: Any) -> None:
        ...

    def __irshift__(self, other: Any) -> None:
        ...

    def __iand__(self, other: Any) -> None:
        ...

    def __ixor__(self, other: Any) -> None:
        ...

    def __ior__(self, other: Any) -> None:
        ...

class WindowedItemsViewT(ItemsView):

    def __init__(self, mapping: Any, event: Any = None) -> None:
        ...

    def __iter__(self) -> None:
        ...

    def now(self) -> None:
        ...

    def current(self, event: Any = None) -> None:
        ...

    def delta(self, d: Any, event: Any = None) -> None:
        ...

class WindowedValuesViewT(ValuesView):

    def __init__(self, mapping: Any, event: Any = None) -> None:
        ...

    def __iter__(self) -> None:
        ...

    def now(self) -> None:
        ...

    def current(self, event: Any = None) -> None:
        ...

    def delta(self, d: Any, event: Any = None) -> None:
        ...

class WindowWrapperT(MutableMapping):

    def __init__(self, table: Any, *, relative_to: Any = None, key_index: bool = False, key_index_table: Any = None) -> None:
        ...

    @property
    def name(self) -> Any:
        ...

    def clone(self, relative_to: Any) -> None:
        ...

    def relative_to_now(self) -> None:
        ...

    def relative_to_field(self, field: Any) -> None:
        ...

    def relative_to_stream(self) -> None:
        ...

    def get_timestamp(self, event: Any = None) -> None:
        ...

    def __getitem__(self, key: Any) -> None:
        ...

    def keys(self) -> None:
        ...

    def on_set_key(self, key: Any, value: Any) -> None:
        ...

    def on_del_key(self, key: Any) -> None:
        ...

    def as_ansitable(self, **kwargs: Any) -> None:
        ...

    @property
    def get_relative_timestamp(self) -> Any:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Any) -> None:
        ...
