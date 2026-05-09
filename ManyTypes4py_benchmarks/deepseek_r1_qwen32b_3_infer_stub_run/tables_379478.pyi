from __future__ import annotations
import abc
import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    ValuesView,
)
from datetime import datetime
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

KT = TypeVar('KT')
VT = TypeVar('VT')

class CollectionT(ServiceT, JoinableT):
    is_global: bool

    def __init__(
        self,
        app: _AppT,
        *,
        name: Optional[str] = ...,
        default: Any = ...,
        store: Optional[StoreT] = ...,
        schema: Optional[_SchemaT] = ...,
        key_type: Any = ...,
        value_type: Any = ...,
        partitions: Optional[int] = ...,
        window: Optional[WindowT] = ...,
        changelog_topic: Optional[TopicT] = ...,
        help: Optional[str] = ...,
        on_recover: Optional[RecoverCallback] = ...,
        on_changelog_event: Optional[ChangelogEventCallback] = ...,
        recovery_buffer_size: int = ...,
        standby_buffer_size: Optional[int] = ...,
        extra_topic_configs: Optional[dict] = ...,
        options: Optional[dict] = ...,
        use_partitioner: bool = ...,
        on_window_close: Optional[WindowCloseCallback] = ...,
        **kwargs: Any,
    ) -> None: ...

    def clone(self, **kwargs: Any) -> CollectionT: ...

    @property
    def changelog_topic(self) -> TopicT: ...
    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None: ...

    def _changelog_topic_name(self) -> str: ...

    def apply_changelog_batch(self, batch: Iterable[FutureMessage]) -> None: ...

    def persisted_offset(self, tp: TP) -> Optional[int]: ...

    async def need_active_standby_for(self, tp: TP) -> bool: ...

    def reset_state(self) -> None: ...

    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: CodecArg = ...,
        value_serializer: CodecArg = ...,
    ) -> None: ...

    def partition_for_key(self, key: Any) -> Optional[int]: ...

    async def on_window_close(self, key: KT, value: VT) -> None: ...

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP],
    ) -> None: ...

    async def on_changelog_event(self, event: EventT) -> None: ...

    def on_recover(self, fun: RecoverCallback) -> None: ...

    async def on_recovery_completed(
        self,
        active_tps: Set[TP],
        standby_tps: Set[TP],
    ) -> None: ...

    async def call_recover_callbacks(self) -> None: ...

    def using_window(
        self,
        window: WindowT,
        *,
        key_index: bool = ...,
    ) -> WindowWrapperT: ...

    def hopping(
        self,
        size: Seconds,
        step: Seconds,
        expires: Optional[Seconds] = ...,
        key_index: bool = ...,
    ) -> WindowWrapperT: ...

    def tumbling(
        self,
        size: Seconds,
        expires: Optional[Seconds] = ...,
        key_index: bool = ...,
    ) -> WindowWrapperT: ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    def _relative_now(self, event: Optional[EventT] = ...) -> datetime: ...

    def _relative_event(self, event: Optional[EventT] = ...) -> datetime: ...

    def _relative_field(self, field: _FieldDescriptorT) -> datetime: ...

    def _relative_timestamp(self, timestamp: Union[float, datetime]) -> datetime: ...

    def _windowed_contains(self, key: KT, timestamp: datetime) -> bool: ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    def __init__(self, app: _AppT, **kwargs: Any) -> None: ...

    def add(self, table: CollectionT) -> None: ...

    def persist_offset_on_commit(self, store: StoreT, tp: TP, offset: int) -> None: ...

    def on_commit(self, offsets: MutableMapping[TP, int]) -> None: ...

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP],
    ) -> None: ...

    def on_partitions_revoked(self, revoked: Set[TP]) -> None: ...

    def on_rebalance_start(self) -> None: ...

    async def wait_until_tables_registered(self) -> None: ...

    async def wait_until_recovery_completed(self) -> None: ...

    @property
    def changelog_topics(self) -> MutableMapping[str, TopicT]: ...

class WindowSetT(FastUserDict[KT, VT]):
    def __init__(
        self,
        key: KT,
        table: TableT,
        wrapper: WindowWrapperT,
        event: Optional[EventT] = ...,
    ) -> None: ...

    def apply(self, op: Callable[[VT, Any], VT], value: Any, event: Optional[EventT] = ...) -> None: ...

    def value(self, event: Optional[EventT] = ...) -> VT: ...

    def current(self, event: Optional[EventT] = ...) -> VT: ...

    def now(self) -> datetime: ...

    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> datetime: ...

    def __iadd__(self, other: Any) -> WindowSetT: ...
    def __isub__(self, other: Any) -> WindowSetT: ...
    def __imul__(self, other: Any) -> WindowSetT: ...
    def __itruediv__(self, other: Any) -> WindowSetT: ...
    def __ifloordiv__(self, other: Any) -> WindowSetT: ...
    def __imod__(self, other: Any) -> WindowSetT: ...
    def __ipow__(self, other: Any) -> WindowSetT: ...
    def __ilshift__(self, other: Any) -> WindowSetT: ...
    def __irshift__(self, other: Any) -> WindowSetT: ...
    def __iand__(self, other: Any) -> WindowSetT: ...
    def __ixor__(self, other: Any) -> WindowSetT: ...
    def __ior__(self, other: Any) -> WindowSetT: ...

class WindowedItemsViewT(ItemsView):
    def __init__(self, mapping: MutableMapping[KT, VT], event: Optional[EventT] = ...) -> None: ...

    def __iter__(self) -> Iterator[Tuple[KT, VT]]: ...

    def now(self) -> datetime: ...

    def current(self, event: Optional[EventT] = ...) -> datetime: ...

    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> datetime: ...

class WindowedValuesViewT(ValuesView):
    def __init__(self, mapping: MutableMapping[KT, VT], event: Optional[EventT] = ...) -> None: ...

    def __iter__(self) -> Iterator[VT]: ...

    def now(self) -> datetime: ...

    def current(self, event: Optional[EventT] = ...) -> datetime: ...

    def delta(self, d: Seconds, event: Optional[EventT] = ...) -> datetime: ...

class WindowWrapperT(MutableMapping):
    def __init__(
        self,
        table: TableT,
        *,
        relative_to: Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]] = ...,
        key_index: bool = ...,
        key_index_table: Optional[TableT] = ...,
    ) -> None: ...

    @property
    def name(self) -> str: ...

    def clone(self, relative_to: Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]] = ...) -> WindowWrapperT: ...

    def relative_to_now(self) -> None: ...

    def relative_to_field(self, field: _FieldDescriptorT) -> None: ...

    def relative_to_stream(self) -> None: ...

    def get_timestamp(self, event: Optional[EventT] = ...) -> datetime: ...

    def __getitem__(self, key: KT) -> VT: ...

    def keys(self) -> KeysView[KT]: ...

    def on_set_key(self, key: KT, value: VT) -> None: ...

    def on_del_key(self, key: KT) -> None: ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    @property
    def get_relative_timestamp(self) -> Callable[[Optional[EventT]], datetime]: ...
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Union[_FieldDescriptorT, RelativeHandler, datetime, float]) -> None: ...