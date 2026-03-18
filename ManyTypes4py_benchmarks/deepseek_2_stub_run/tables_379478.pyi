```python
import abc
import typing
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
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

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import FieldDescriptorT as _FieldDescriptorT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
else:
    _AppT = Any
    _FieldDescriptorT = Any
    _ModelArg = Any
    _SchemaT = Any

__all__: Tuple[str, ...] = ...

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union[_FieldDescriptorT, RelativeHandler, datetime, float]]
CollectionTps = MutableMapping["CollectionT", Set[TP]]
KT = TypeVar("KT")
VT = TypeVar("VT")

class CollectionT(ServiceT, JoinableT):
    is_global: bool = ...

    @abc.abstractmethod
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
        extra_topic_configs: Optional[Mapping[str, Any]] = ...,
        options: Optional[Mapping[str, Any]] = ...,
        use_partitioner: bool = ...,
        on_window_close: Optional[WindowCloseCallback] = ...,
        **kwargs: Any,
    ) -> None: ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> "CollectionT": ...

    @property
    @abc.abstractmethod
    def changelog_topic(self) -> Optional[TopicT]: ...

    @changelog_topic.setter
    def changelog_topic(self, topic: Optional[TopicT]) -> None: ...

    @abc.abstractmethod
    def _changelog_topic_name(self) -> str: ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch: Any) -> None: ...

    @abc.abstractmethod
    def persisted_offset(self, tp: TP) -> Optional[int]: ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp: TP) -> bool: ...

    @abc.abstractmethod
    def reset_state(self) -> None: ...

    @abc.abstractmethod
    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
    ) -> FutureMessage: ...

    @abc.abstractmethod
    def partition_for_key(self, key: Any) -> int: ...

    @abc.abstractmethod
    async def on_window_close(self, key: Any, value: Any) -> None: ...

    @abc.abstractmethod
    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP],
    ) -> None: ...

    @abc.abstractmethod
    async def on_changelog_event(self, event: EventT) -> None: ...

    @abc.abstractmethod
    def on_recover(self, fun: RecoverCallback) -> None: ...

    @abc.abstractmethod
    async def on_recovery_completed(
        self, active_tps: Set[TP], standby_tps: Set[TP]
    ) -> None: ...

    @abc.abstractmethod
    async def call_recover_callbacks(self) -> None: ...

    @abc.abstractmethod
    def using_window(
        self, window: WindowT, *, key_index: bool = ...
    ) -> "CollectionT": ...

    @abc.abstractmethod
    def hopping(
        self,
        size: Seconds,
        step: Seconds,
        expires: Optional[Seconds] = ...,
        key_index: bool = ...,
    ) -> "CollectionT": ...

    @abc.abstractmethod
    def tumbling(
        self,
        size: Seconds,
        expires: Optional[Seconds] = ...,
        key_index: bool = ...,
    ) -> "CollectionT": ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> str: ...

    @abc.abstractmethod
    def _relative_now(self, event: Optional[EventT] = ...) -> float: ...

    @abc.abstractmethod
    def _relative_event(self, event: Optional[EventT] = ...) -> float: ...

    @abc.abstractmethod
    def _relative_field(self, field: _FieldDescriptorT) -> float: ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp: float) -> float: ...

    @abc.abstractmethod
    def _windowed_contains(self, key: Any, timestamp: float) -> bool: ...

class TableT(CollectionT, ManagedUserDict[KT, VT]): ...

class GlobalTableT(TableT): ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    @abc.abstractmethod
    def __init__(self, app: _AppT, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def add(self, table: CollectionT) -> None: ...

    @abc.abstractmethod
    def persist_offset_on_commit(
        self, store: StoreT, tp: TP, offset: int
    ) -> None: ...

    @abc.abstractmethod
    def on_commit(self, offsets: Mapping[TP, int]) -> None: ...

    @abc.abstractmethod
    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP],
    ) -> None: ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked: Set[TP]) -> None: ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None: ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self) -> None: ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self) -> None: ...

    @property
    @abc.abstractmethod
    def changelog_topics(self) -> Mapping[str, TopicT]: ...

class WindowSetT(FastUserDict[KT, VT]):
    @abc.abstractmethod
    def __init__(
        self,
        key: KT,
        table: "TableT[KT, VT]",
        wrapper: "WindowWrapperT[KT, VT]",
        event: Optional[EventT] = ...,
    ) -> None: ...

    @abc.abstractmethod
    def apply(self, op: Callable[[VT, VT], VT], value: VT, event: Optional[EventT] = ...) -> VT: ...

    @abc.abstractmethod
    def value(self, event: Optional[EventT] = ...) -> VT: ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = ...) -> VT: ...

    @abc.abstractmethod
    def now(self) -> VT: ...

    @abc.abstractmethod
    def delta(self, d: VT, event: Optional[EventT] = ...) -> VT: ...

    @abc.abstractmethod
    def __iadd__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __isub__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __imul__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __itruediv__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __ifloordiv__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __imod__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __ipow__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __ilshift__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __irshift__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __iand__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __ixor__(self, other: VT) -> "WindowSetT[KT, VT]": ...

    @abc.abstractmethod
    def __ior__(self, other: VT) -> "WindowSetT[KT, VT]": ...

class WindowedItemsViewT(ItemsView[KT, VT]):
    @abc.abstractmethod
    def __init__(
        self, mapping: Mapping[KT, VT], event: Optional[EventT] = ...
    ) -> None: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[KT, VT]]: ...

    @abc.abstractmethod
    def now(self) -> ItemsView[KT, VT]: ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = ...) -> ItemsView[KT, VT]: ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Optional[EventT] = ...) -> ItemsView[KT, VT]: ...

class WindowedValuesViewT(ValuesView[VT]):
    @abc.abstractmethod
    def __init__(
        self, mapping: Mapping[Any, VT], event: Optional[EventT] = ...
    ) -> None: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[VT]: ...

    @abc.abstractmethod
    def now(self) -> ValuesView[VT]: ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = ...) -> ValuesView[VT]: ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Optional[EventT] = ...) -> ValuesView[VT]: ...

class WindowWrapperT(MutableMapping[KT, VT]):
    @abc.abstractmethod
    def __init__(
        self,
        table: "TableT[KT, VT]",
        *,
        relative_to: Optional[RelativeArg] = ...,
        key_index: bool = ...,
        key_index_table: Optional["TableT[Any, Any]"] = ...,
    ) -> None: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def clone(self, relative_to: RelativeArg) -> "WindowWrapperT[KT, VT]": ...

    @abc.abstractmethod
    def relative_to_now(self) -> "WindowWrapperT[KT, VT]": ...

    @abc.abstractmethod
    def relative_to_field(self, field: _FieldDescriptorT) -> "WindowWrapperT[KT, VT]": ...

    @abc.abstractmethod
    def relative_to_stream(self) -> "WindowWrapperT[KT, VT]": ...

    @abc.abstractmethod
    def get_timestamp(self, event: Optional[EventT] = ...) -> float: ...

    @abc.abstractmethod
    def __getitem__(self, key: KT) -> WindowSetT[KT, VT]: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[KT]: ...

    @abc.abstractmethod
    def on_set_key(self, key: KT, value: VT) -> None: ...

    @abc.abstractmethod
    def on_del_key(self, key: KT) -> None: ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> str: ...

    @property
    def get_relative_timestamp(self) -> Callable[[Optional[EventT]], float]: ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: RelativeArg) -> None: ...
```