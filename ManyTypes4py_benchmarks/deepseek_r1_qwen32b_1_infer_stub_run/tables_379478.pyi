import abc
import datetime
import typing
from datetime import datetime as datetime_type
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
    class _AppT: ...
    class _FieldDescriptorT: ...
    class _ModelArg: ...
    class _SchemaT: ...

__all__ = [
    'RecoverCallback',
    'RelativeArg',
    'CollectionT',
    'TableT',
    'GlobalTableT',
    'TableManagerT',
    'WindowCloseCallback',
    'WindowSetT',
    'WindowedItemsViewT',
    'WindowedValuesViewT',
    'WindowWrapperT',
    'ChangelogEventCallback',
    'CollectionTps',
]

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime_type]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union[_FieldDescriptorT, RelativeHandler, datetime_type, float]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]

KT = TypeVar('KT')
VT = TypeVar('VT')

class CollectionT(ServiceT, JoinableT):
    is_global: bool

    def __init__(
        self,
        app: _AppT,
        *,
        name: Optional[str] = None,
        default: Optional[Any] = None,
        store: Optional[StoreT] = None,
        schema: Optional[_SchemaT] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        partitions: Optional[int] = None,
        window: Optional[WindowT] = None,
        changelog_topic: Optional[TopicT] = None,
        help: Optional[str] = None,
        on_recover: Optional[RecoverCallback] = None,
        on_changelog_event: Optional[ChangelogEventCallback] = None,
        recovery_buffer_size: int = 1000,
        standby_buffer_size: Optional[int] = None,
        extra_topic_configs: Optional[dict] = None,
        options: Optional[dict] = None,
        use_partitioner: bool = False,
        on_window_close: Optional[WindowCloseCallback] = None,
        **kwargs: Any
    ) -> None: ...

    def clone(self, **kwargs: Any) -> 'CollectionT': ...

    @property
    def changelog_topic(self) -> TopicT: ...
    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None: ...

    def _changelog_topic_name(self) -> str: ...

    def apply_changelog_batch(self, batch: Any) -> None: ...

    def persisted_offset(self, tp: TP) -> Optional[int]: ...

    async def need_active_standby_for(self, tp: TP) -> bool: ...

    def reset_state(self) -> None: ...

    def send_changelog(
        self,
        partition: int,
        key: KT,
        value: VT,
        key_serializer: CodecArg = None,
        value_serializer: CodecArg = None
    ) -> None: ...

    def partition_for_key(self, key: KT) -> int: ...

    async def on_window_close(self, key: KT, value: VT) -> None: ...

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP]
    ) -> None: ...

    async def on_changelog_event(self, event: EventT) -> None: ...

    def on_recover(self, fun: RecoverCallback) -> None: ...

    async def on_recovery_completed(
        self,
        active_tps: Set[TP],
        standby_tps: Set[TP]
    ) -> None: ...

    async def call_recover_callbacks(self) -> None: ...

    def using_window(self, window: WindowT, *, key_index: bool = False) -> 'WindowWrapperT': ...

    def hopping(
        self,
        size: Seconds,
        step: Seconds,
        expires: Optional[Seconds] = None,
        key_index: bool = False
    ) -> 'WindowWrapperT': ...

    def tumbling(
        self,
        size: Seconds,
        expires: Optional[Seconds] = None,
        key_index: bool = False
    ) -> 'WindowWrapperT': ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    def _relative_now(self, event: Optional[EventT] = None) -> float: ...

    def _relative_event(self, event: Optional[EventT] = None) -> float: ...

    def _relative_field(self, field: _FieldDescriptorT) -> float: ...

    def _relative_timestamp(self, timestamp: Union[float, datetime_type]) -> float: ...

    def _windowed_contains(self, key: KT, timestamp: float) -> bool: ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    def __init__(self, app: _AppT, **kwargs: Any) -> None: ...

    def add(self, table: CollectionT) -> None: ...

    def persist_offset_on_commit(self, store: StoreT, tp: TP, offset: int) -> None: ...

    def on_commit(self, offsets: Mapping[TP, int]) -> None: ...

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP]
    ) -> None: ...

    def on_partitions_revoked(self, revoked: Set[TP]) -> None: ...

    def on_rebalance_start(self) -> None: ...

    async def wait_until_tables_registered(self) -> None: ...

    async def wait_until_recovery_completed(self) -> None: ...

    @property
    def changelog_topics(self) -> Set[TopicT]: ...

class WindowSetT(FastUserDict[KT, VT]):
    def __init__(self, key: KT, table: CollectionT, wrapper: 'WindowWrapperT', event: Optional[EventT] = None) -> None: ...

    def apply(self, op: Any, value: VT, event: Optional[EventT] = None) -> None: ...

    def value(self, event: Optional[EventT] = None) -> VT: ...

    def current(self, event: Optional[EventT] = None) -> VT: ...

    def now(self) -> VT: ...

    def delta(self, d: Union[float, datetime_type], event: Optional[EventT] = None) -> VT: ...

    def __iadd__(self, other: Any) -> 'WindowSetT': ...
    def __isub__(self, other: Any) -> 'WindowSetT': ...
    def __imul__(self, other: Any) -> 'WindowSetT': ...
    def __itruediv__(self, other: Any) -> 'WindowSetT': ...
    def __ifloordiv__(self, other: Any) -> 'WindowSetT': ...
    def __imod__(self, other: Any) -> 'WindowSetT': ...
    def __ipow__(self, other: Any) -> 'WindowSetT': ...
    def __ilshift__(self, other: Any) -> 'WindowSetT': ...
    def __irshift__(self, other: Any) -> 'WindowSetT': ...
    def __iand__(self, other: Any) -> 'WindowSetT': ...
    def __ixor__(self, other: Any) -> 'WindowSetT': ...
    def __ior__(self, other: Any) -> 'WindowSetT': ...

class WindowedItemsViewT(ItemsView):
    def __init__(self, mapping: MutableMapping[KT, VT], event: Optional[EventT] = None) -> None: ...

    def __iter__(self) -> Iterator[Tuple[KT, VT]]: ...

    def now(self) -> Iterator[Tuple[KT, VT]]: ...

    def current(self, event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]: ...

    def delta(self, d: Union[float, datetime_type], event: Optional[EventT] = None) -> Iterator[Tuple[KT, VT]]: ...

class WindowedValuesViewT(ValuesView):
    def __init__(self, mapping: MutableMapping[KT, VT], event: Optional[EventT] = None) -> None: ...

    def __iter__(self) -> Iterator[VT]: ...

    def now(self) -> Iterator[VT]: ...

    def current(self, event: Optional[EventT] = None) -> Iterator[VT]: ...

    def delta(self, d: Union[float, datetime_type], event: Optional[EventT] = None) -> Iterator[VT]: ...

class WindowWrapperT(MutableMapping):
    def __init__(
        self,
        table: CollectionT,
        *,
        relative_to: Optional[Union[RelativeHandler, _FieldDescriptorT]] = None,
        key_index: bool = False,
        key_index_table: Optional[TableT] = None
    ) -> None: ...

    @property
    def name(self) -> str: ...

    def clone(self, relative_to: Union[RelativeHandler, _FieldDescriptorT]) -> 'WindowWrapperT': ...

    def relative_to_now(self) -> 'WindowWrapperT': ...

    def relative_to_field(self, field: _FieldDescriptorT) -> 'WindowWrapperT': ...

    def relative_to_stream(self) -> 'WindowWrapperT': ...

    def get_timestamp(self, event: Optional[EventT] = None) -> float: ...

    def __getitem__(self, key: KT) -> VT: ...

    def keys(self) -> Iterable[KT]: ...

    def on_set_key(self, key: KT, value: VT) -> None: ...

    def on_del_key(self, key: KT) -> None: ...

    def as_ansitable(self, **kwargs: Any) -> str: ...

    @property
    def get_relative_timestamp(self) -> Callable[[Optional[EventT]], float]: ...
    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Union[RelativeHandler, _FieldDescriptorT]) -> None: ...