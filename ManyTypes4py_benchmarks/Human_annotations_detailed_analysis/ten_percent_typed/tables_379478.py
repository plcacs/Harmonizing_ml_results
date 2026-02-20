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
    class _AppT: ...  # noqa
    class _FieldDescriptorT: ...  # noqa
    class _ModelArg: ...  # noqa
    class _SchemaT: ...   # noqa

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

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union[
    _FieldDescriptorT,
    RelativeHandler,
    datetime,
    float,
]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]

KT = TypeVar('KT')
VT = TypeVar('VT')


class CollectionT(ServiceT, JoinableT):
    app: _AppT
    name: str
    default: Any  # noqa: E704
    schema: Optional[_SchemaT]
    key_type: Optional[_ModelArg]
    value_type: Optional[_ModelArg]
    partitions: Optional[int]
    window: Optional[WindowT]
    help: str
    recovery_buffer_size: int
    standby_buffer_size: int
    options: Optional[Mapping[str, Any]]
    last_closed_window: float
    use_partitioner: bool

    is_global: bool = False

    @abc.abstractmethod
    def __init__(self,
                 app,
                 *,
                 name = None,
                 default: Callable[[], Any] = None,
                 store = None,
                 schema = None,
                 key_type = None,
                 value_type = None,
                 partitions: int = None,
                 window = None,
                 changelog_topic = None,
                 help = None,
                 on_recover = None,
                 on_changelog_event = None,
                 recovery_buffer_size = 1000,
                 standby_buffer_size = None,
                 extra_topic_configs: Mapping[str, Any] = None,
                 options = None,
                 use_partitioner = False,
                 on_window_close = None,
                 **kwargs) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs) -> Any:
        ...

    @property
    @abc.abstractmethod
    def changelog_topic(self) -> TopicT:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None:
        ...

    @abc.abstractmethod
    def _changelog_topic_name(self) -> str:
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch) -> None:
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp) -> Optional[int]:
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp) -> bool:
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        ...

    @abc.abstractmethod
    def send_changelog(self,
                       partition: Optional[int],
                       key,
                       value,
                       key_serializer = None,
                       value_serializer = None) -> FutureMessage:
        ...

    @abc.abstractmethod
    def partition_for_key(self, key) -> Optional[int]:
        ...

    @abc.abstractmethod
    async def on_window_close(self, key, value) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self,
                           assigned,
                           revoked,
                           newly_assigned) -> None:
        ...

    @abc.abstractmethod
    async def on_changelog_event(self, event) -> None:
        ...

    @abc.abstractmethod
    def on_recover(self, fun) -> RecoverCallback:
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self,
                                    active_tps,
                                    standby_tps) -> None:
        ...

    @abc.abstractmethod
    async def call_recover_callbacks(self) -> None:
        ...

    @abc.abstractmethod
    def using_window(self, window, *,
                     key_index = False) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def hopping(self, size, step,
                expires = None,
                key_index = False) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def tumbling(self, size,
                 expires = None,
                 key_index = False) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs) -> str:
        ...

    @abc.abstractmethod
    def _relative_now(self, event = None) -> float:
        ...

    @abc.abstractmethod
    def _relative_event(self, event = None) -> float:
        ...

    @abc.abstractmethod
    def _relative_field(self, field) -> RelativeHandler:
        ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp) -> RelativeHandler:
        ...

    @abc.abstractmethod
    def _windowed_contains(self, key, timestamp) -> bool:
        ...


class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...


class GlobalTableT(TableT):
    ...


class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):
    app: _AppT

    actives_ready: bool
    standbys_ready: bool

    @abc.abstractmethod
    def __init__(self, app, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def add(self, table) -> CollectionT:
        ...

    @abc.abstractmethod
    def persist_offset_on_commit(self,
                                 store,
                                 tp,
                                 offset) -> None:
        ...

    @abc.abstractmethod
    def on_commit(self, offsets) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self,
                           assigned,
                           revoked,
                           newly_assigned) -> None:
        ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def changelog_topics(self) -> Set[str]:
        ...


class WindowSetT(FastUserDict[KT, VT]):
    key: Any
    table: TableT
    event: Optional[EventT]

    @abc.abstractmethod
    def __init__(self,
                 key,
                 table,
                 wrapper,
                 event = None) -> None:
        ...

    @abc.abstractmethod
    def apply(self,
              op,
              value,
              event = None) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def value(self, event = None) -> VT:
        ...

    @abc.abstractmethod
    def current(self, event = None) -> VT:
        ...

    @abc.abstractmethod
    def now(self) -> VT:
        ...

    @abc.abstractmethod
    def delta(self, d, event = None) -> VT:
        ...

    @abc.abstractmethod
    def __iadd__(self, other: VT) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __isub__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __imul__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __itruediv__(self, other: VT) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __ifloordiv__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __imod__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __ipow__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __ilshift__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __irshift__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __iand__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __ixor__(self, other) -> 'WindowSetT':
        ...

    @abc.abstractmethod
    def __ior__(self, other: VT) -> 'WindowSetT':
        ...


class WindowedItemsViewT(ItemsView):

    @abc.abstractmethod
    def __init__(self,
                 mapping,
                 event = None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        ...

    @abc.abstractmethod
    def now(self) -> Iterator[Tuple[Any, Any]]:
        ...

    @abc.abstractmethod
    def current(self, event = None) -> Iterator[Tuple[Any, Any]]:
        ...

    @abc.abstractmethod
    def delta(self,
              d,
              event: EventT = None) -> Iterator[Tuple[Any, Any]]:
        ...


class WindowedValuesViewT(ValuesView):

    @abc.abstractmethod
    def __init__(self,
                 mapping,
                 event = None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def now(self) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def current(self, event = None) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def delta(self, d, event = None) -> Iterator[Any]:
        ...


class WindowWrapperT(MutableMapping):
    table: TableT

    @abc.abstractmethod
    def __init__(self, table, *,
                 relative_to = None,
                 key_index = False,
                 key_index_table = None) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def clone(self, relative_to) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def relative_to_now(self) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def relative_to_field(self, field) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def relative_to_stream(self) -> 'WindowWrapperT':
        ...

    @abc.abstractmethod
    def get_timestamp(self, event = None) -> float:
        ...

    @abc.abstractmethod
    def __getitem__(self, key) -> WindowSetT:
        ...

    @abc.abstractmethod
    def keys(self) -> KeysView:
        ...

    @abc.abstractmethod
    def on_set_key(self, key: Any, value) -> None:
        ...

    @abc.abstractmethod
    def on_del_key(self, key) -> None:
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs) -> str:
        ...

    @property
    def get_relative_timestamp(self) -> Optional[RelativeHandler]:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to) -> None:
        ...
