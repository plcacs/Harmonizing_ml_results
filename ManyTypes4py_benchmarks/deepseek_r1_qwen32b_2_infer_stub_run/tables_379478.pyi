from __future__ import annotations
import abc
import typing
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
)
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
    is_global: ClassVar[bool] = False
    app: _AppT
    name: str
    default: Any
    store: StoreT
    schema: _SchemaT
    key_type: type
    value_type: type
    partitions: Optional[int]
    window: Optional[WindowT]
    changelog_topic: TopicT
    recovery_buffer_size: int
    standby_buffer_size: Optional[int]
    extra_topic_configs: Dict[str, Any]
    options: Dict[str, Any]
    use_partitioner: bool
    on_window_close: Optional[WindowCloseCallback]

    def __init__(
        self,
        app: _AppT,
        *,
        name: Optional[str] = None,
        default: Optional[Any] = None,
        store: Optional[StoreT] = None,
        schema: Optional[_SchemaT] = None,
        key_type: Optional[type] = None,
        value_type: Optional[type] = None,
        partitions: Optional[int] = None,
        window: Optional[WindowT] = None,
        changelog_topic: Optional[TopicT] = None,
        help: Optional[str] = None,
        on_recover: Optional[RecoverCallback] = None,
        on_changelog_event: Optional[ChangelogEventCallback] = None,
        recovery_buffer_size: int = 1000,
        standby_buffer_size: Optional[int] = None,
        extra_topic_configs: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        use_partitioner: bool = False,
        on_window_close: Optional[WindowCloseCallback] = None,
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
        key: KT,
        value: VT,
        key_serializer: CodecArg = None,
        value_serializer: CodecArg = None,
    ) -> None: ...

    def partition_for_key(self, key: KT) -> Optional[int]: ...

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
        key_index: bool = False,
    ) -> WindowWrapperT[KT, VT]: ...

    def hopping(
        self,
        size: Seconds,
        step: Seconds,
        expires: Optional[Seconds] = None,
        key_index: bool = False,
    ) -> WindowWrapperT[KT, VT]: ...

    def tumbling(
        self,
        size: Seconds,
        expires: Optional[Seconds] = None,
        key_index: bool = False,
    ) -> WindowWrapperT[KT, VT]: ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    def _relative_now(self, event: Optional[EventT] = None) -> float: ...

    def _relative_event(self, event: Optional[EventT] = None) -> float: ...

    def _relative_field(self, field: _FieldDescriptorT) -> float: ...

    def _relative_timestamp(self, timestamp: Union[float, datetime]) -> float: ...

    def _windowed_contains(self, key: KT, timestamp: float) -> bool: ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    app: _AppT

    def __init__(self, app: _AppT, **kwargs: Any) -> None: ...

    def add(self, table: CollectionT) -> None: ...

    def persist_offset_on_commit(
        self,
        store: StoreT,
        tp: TP,
        offset: int,
    ) -> None: ...

    def on_commit(self, offsets: Dict[TP, int]) -> None: ...

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
    def changelog_topics(self) -> Set[TopicT]: ...

class WindowSetT(FastUserDict[KT, VT]):
    def __init__(
        self,
        key: KT,
        table: TableT,
        wrapper: WindowWrapperT[KT, VT],
        event: Optional[EventT] = None,
    ) -> None: ...

    def apply(
        self,
        op: Callable[[VT, VT], VT],
        value: VT,
        event: Optional[EventT] = None,
    ) -> None: ...

    def value(self, event: Optional[EventT] = None) -> VT: ...

    def current(self, event: Optional[EventT] = None) -> VT: ...

    def now(self) -> float: ...

    def delta(
        self,
        d: Seconds,
        event: Optional[EventT] = None,
    ) -> float: ...

    def __iadd__(self, other: VT) -> WindowSetT[KT, VT]: ...
    __isub__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __imul__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __itruediv__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __ifloordiv__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __imod__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __ipow__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __ilshift__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __irshift__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __iand__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __ixor__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]
    __ior__: Callable[[WindowSetT[KT, VT], VT], WindowSetT[KT, VT]]

class WindowedItemsViewT(ItemsView):
    def __init__(
        self,
        mapping: MutableMapping[KT, VT],
        event: Optional[EventT] = None,
    ) -> None: ...

    def __iter__(self) -> Iterator[Tuple[KT, VT]]: ...

    def now(self) -> float: ...

    def current(self, event: Optional[EventT] = None) -> float: ...

    def delta(
        self,
        d: Seconds,
        event: Optional[EventT] = None,
    ) -> float: ...

class WindowedValuesViewT(ValuesView):
    def __init__(
        self,
        mapping: MutableMapping[KT, VT],
        event: Optional[EventT] = None,
    ) -> None: ...

    def __iter__(self) -> Iterator[VT]: ...

    def now(self) -> float: ...

    def current(self, event: Optional[EventT] = None) -> float: ...

    def delta(
        self,
        d: Seconds,
        event: Optional[EventT] = None,
    ) -> float: ...

class WindowWrapperT(MutableMapping[KT, VT]):
    def __init__(
        self,
        table: TableT,
        *,
        relative_to: Optional[RelativeArg] = None,
        key_index: bool = False,
        key_index_table: Optional[TableT] = None,
    ) -> None: ...

    @property
    def name(self) -> str: ...

    def clone(self, relative_to: RelativeArg) -> WindowWrapperT[KT, VT]: ...

    def relative_to_now(self) -> WindowWrapperT[KT, VT]: ...

    def relative_to_field(self, field: _FieldDescriptorT) -> WindowWrapperT[KT, VT]: ...

    def relative_to_stream(self) -> WindowWrapperT[KT, VT]: ...

    def get_timestamp(self, event: Optional[EventT] = None) -> float: ...

    def __getitem__(self, key: KT) -> VT: ...

    def keys(self) -> KeysView[KT]: ...

    def on_set_key(self, key: KT, value: VT) -> None: ...

    def on_del_key(self, key: KT) -> None: ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    @property
    def get_relative_timestamp(self) -> RelativeArg: ...
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: RelativeArg) -> None: ...