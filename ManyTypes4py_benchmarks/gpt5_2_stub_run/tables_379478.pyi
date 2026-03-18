import abc
from datetime import datetime
from typing import Any, Awaitable, Callable, ItemsView, Iterator, KeysView, MutableMapping, Optional, Set, TypeVar, Union, ValuesView
from mode import ServiceT
from mode.utils.collections import FastUserDict, ManagedUserDict
from .events import EventT
from .streams import JoinableT
from .tuples import TP
from .models import FieldDescriptorT as _FieldDescriptorT

__all__: list[str] = ...

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]
KT = TypeVar('KT')
VT = TypeVar('VT')


class CollectionT(ServiceT, JoinableT):
    is_global: bool = ...

    @abc.abstractmethod
    def __init__(
        self,
        app: Any,
        *,
        name: Any = ...,
        default: Any = ...,
        store: Any = ...,
        schema: Any = ...,
        key_type: Any = ...,
        value_type: Any = ...,
        partitions: Any = ...,
        window: Any = ...,
        changelog_topic: Any = ...,
        help: Any = ...,
        on_recover: Any = ...,
        on_changelog_event: Any = ...,
        recovery_buffer_size: Any = ...,
        standby_buffer_size: Any = ...,
        extra_topic_configs: Any = ...,
        options: Any = ...,
        use_partitioner: Any = ...,
        on_window_close: Any = ...,
        **kwargs: Any
    ) -> None: ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> Any: ...

    @property
    @abc.abstractmethod
    def changelog_topic(self) -> Any: ...

    @changelog_topic.setter
    def changelog_topic(self, topic: Any) -> None: ...

    @abc.abstractmethod
    def _changelog_topic_name(self) -> Any: ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch: Any) -> Any: ...

    @abc.abstractmethod
    def persisted_offset(self, tp: Any) -> Any: ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp: Any) -> Any: ...

    @abc.abstractmethod
    def reset_state(self) -> Any: ...

    @abc.abstractmethod
    def send_changelog(
        self,
        partition: Any,
        key: Any,
        value: Any,
        key_serializer: Any = ...,
        value_serializer: Any = ...
    ) -> Any: ...

    @abc.abstractmethod
    def partition_for_key(self, key: Any) -> Any: ...

    @abc.abstractmethod
    async def on_window_close(self, key: Any, value: Any) -> Any: ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> Any: ...

    @abc.abstractmethod
    async def on_changelog_event(self, event: Any) -> Any: ...

    @abc.abstractmethod
    def on_recover(self, fun: Any) -> Any: ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps: Any, standby_tps: Any) -> Any: ...

    @abc.abstractmethod
    async def call_recover_callbacks(self) -> Any: ...

    @abc.abstractmethod
    def using_window(self, window: Any, *, key_index: Any = ...) -> Any: ...

    @abc.abstractmethod
    def hopping(self, size: Any, step: Any, expires: Any = ..., key_index: Any = ...) -> Any: ...

    @abc.abstractmethod
    def tumbling(self, size: Any, expires: Any = ..., key_index: Any = ...) -> Any: ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def _relative_now(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def _relative_event(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def _relative_field(self, field: Any) -> Any: ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp: Any) -> Any: ...

    @abc.abstractmethod
    def _windowed_contains(self, key: Any, timestamp: Any) -> Any: ...


class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...


class GlobalTableT(TableT):
    ...


class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    @abc.abstractmethod
    def __init__(self, app: Any, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def add(self, table: Any) -> Any: ...

    @abc.abstractmethod
    def persist_offset_on_commit(self, store: Any, tp: Any, offset: Any) -> Any: ...

    @abc.abstractmethod
    def on_commit(self, offsets: Any) -> Any: ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> Any: ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked: Any) -> Any: ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> Any: ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self) -> Any: ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self) -> Any: ...

    @property
    @abc.abstractmethod
    def changelog_topics(self) -> Any: ...


class WindowSetT(FastUserDict[KT, VT]):
    @abc.abstractmethod
    def __init__(self, key: Any, table: Any, wrapper: Any, event: Any = ...) -> None: ...

    @abc.abstractmethod
    def apply(self, op: Any, value: Any, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def value(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def current(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def now(self) -> Any: ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def __iadd__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __isub__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __imul__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __itruediv__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __ifloordiv__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __imod__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __ipow__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __ilshift__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __irshift__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __iand__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __ixor__(self, other: Any) -> Any: ...

    @abc.abstractmethod
    def __ior__(self, other: Any) -> Any: ...


class WindowedItemsViewT(ItemsView[Any, Any]):
    @abc.abstractmethod
    def __init__(self, mapping: Any, event: Any = ...) -> None: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]: ...

    @abc.abstractmethod
    def now(self) -> Any: ...

    @abc.abstractmethod
    def current(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Any = ...) -> Any: ...


class WindowedValuesViewT(ValuesView[Any]):
    @abc.abstractmethod
    def __init__(self, mapping: Any, event: Any = ...) -> None: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]: ...

    @abc.abstractmethod
    def now(self) -> Any: ...

    @abc.abstractmethod
    def current(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Any = ...) -> Any: ...


class WindowWrapperT(MutableMapping[Any, Any]):
    @abc.abstractmethod
    def __init__(
        self,
        table: Any,
        *,
        relative_to: Any = ...,
        key_index: Any = ...,
        key_index_table: Any = ...
    ) -> None: ...

    @property
    @abc.abstractmethod
    def name(self) -> Any: ...

    @abc.abstractmethod
    def clone(self, relative_to: Any) -> Any: ...

    @abc.abstractmethod
    def relative_to_now(self) -> Any: ...

    @abc.abstractmethod
    def relative_to_field(self, field: Any) -> Any: ...

    @abc.abstractmethod
    def relative_to_stream(self) -> Any: ...

    @abc.abstractmethod
    def get_timestamp(self, event: Any = ...) -> Any: ...

    @abc.abstractmethod
    def __getitem__(self, key: Any) -> Any: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[Any]: ...

    @abc.abstractmethod
    def on_set_key(self, key: Any, value: Any) -> Any: ...

    @abc.abstractmethod
    def on_del_key(self, key: Any) -> Any: ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> Any: ...

    @property
    def get_relative_timestamp(self) -> Any: ...
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Any) -> None: ...