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
    app: _AppT
    name: str
    default: Any
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
    def __init__(self, app, *, name: str=None, default: Callable[[], Any]=None, store: Union[str, URL]=None, schema: _SchemaT=None, key_type: _ModelArg=None, value_type: _ModelArg=None, partitions: int=None, window: WindowT=None, changelog_topic: TopicT=None, help: str=None, on_recover: RecoverCallback=None, on_changelog_event: ChangelogEventCallback=None, recovery_buffer_size: int=1000, standby_buffer_size: int=None, extra_topic_configs: Mapping[str, Any]=None, options: Mapping[str, Any]=None, use_partitioner: bool=False, on_window_close: WindowCloseCallback=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any):
        ...

    @property
    @abc.abstractmethod
    def changelog_topic(self):
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic):
        ...

    @abc.abstractmethod
    def _changelog_topic_name(self):
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch):
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp):
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    @abc.abstractmethod
    def reset_state(self):
        ...

    @abc.abstractmethod
    def send_changelog(self, partition, key, value, key_serializer=None, value_serializer=None):
        ...

    @abc.abstractmethod
    def partition_for_key(self, key):
        ...

    @abc.abstractmethod
    async def on_window_close(self, key: Any, value: Any) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def on_changelog_event(self, event: EventT) -> None:
        ...

    @abc.abstractmethod
    def on_recover(self, fun):
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps: Set[TP], standby_tps: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    async def call_recover_callbacks(self) -> None:
        ...

    @abc.abstractmethod
    def using_window(self, window, *, key_index: bool=False):
        ...

    @abc.abstractmethod
    def hopping(self, size, step, expires=None, key_index=False):
        ...

    @abc.abstractmethod
    def tumbling(self, size, expires=None, key_index=False):
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any):
        ...

    @abc.abstractmethod
    def _relative_now(self, event=None):
        ...

    @abc.abstractmethod
    def _relative_event(self, event=None):
        ...

    @abc.abstractmethod
    def _relative_field(self, field):
        ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp):
        ...

    @abc.abstractmethod
    def _windowed_contains(self, key, timestamp):
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
    def __init__(self, app, **kwargs: Any):
        ...

    @abc.abstractmethod
    def add(self, table):
        ...

    @abc.abstractmethod
    def persist_offset_on_commit(self, store, tp, offset):
        ...

    @abc.abstractmethod
    def on_commit(self, offsets):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked):
        ...

    @abc.abstractmethod
    def on_rebalance_start(self):
        ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def changelog_topics(self):
        ...

class WindowSetT(FastUserDict[KT, VT]):
    key: Any
    table: TableT
    event: Optional[EventT]

    @abc.abstractmethod
    def __init__(self, key, table, wrapper, event=None):
        ...

    @abc.abstractmethod
    def apply(self, op, value, event=None):
        ...

    @abc.abstractmethod
    def value(self, event=None):
        ...

    @abc.abstractmethod
    def current(self, event=None):
        ...

    @abc.abstractmethod
    def now(self):
        ...

    @abc.abstractmethod
    def delta(self, d, event=None):
        ...

    @abc.abstractmethod
    def __iadd__(self, other):
        ...

    @abc.abstractmethod
    def __isub__(self, other):
        ...

    @abc.abstractmethod
    def __imul__(self, other):
        ...

    @abc.abstractmethod
    def __itruediv__(self, other):
        ...

    @abc.abstractmethod
    def __ifloordiv__(self, other):
        ...

    @abc.abstractmethod
    def __imod__(self, other):
        ...

    @abc.abstractmethod
    def __ipow__(self, other):
        ...

    @abc.abstractmethod
    def __ilshift__(self, other):
        ...

    @abc.abstractmethod
    def __irshift__(self, other):
        ...

    @abc.abstractmethod
    def __iand__(self, other):
        ...

    @abc.abstractmethod
    def __ixor__(self, other):
        ...

    @abc.abstractmethod
    def __ior__(self, other):
        ...

class WindowedItemsViewT(ItemsView):

    @abc.abstractmethod
    def __init__(self, mapping, event=None):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def now(self):
        ...

    @abc.abstractmethod
    def current(self, event=None):
        ...

    @abc.abstractmethod
    def delta(self, d, event=None):
        ...

class WindowedValuesViewT(ValuesView):

    @abc.abstractmethod
    def __init__(self, mapping, event=None):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def now(self):
        ...

    @abc.abstractmethod
    def current(self, event=None):
        ...

    @abc.abstractmethod
    def delta(self, d, event=None):
        ...

class WindowWrapperT(MutableMapping):
    table: TableT

    @abc.abstractmethod
    def __init__(self, table, *, relative_to: RelativeArg=None, key_index: bool=False, key_index_table: TableT=None):
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @abc.abstractmethod
    def clone(self, relative_to):
        ...

    @abc.abstractmethod
    def relative_to_now(self):
        ...

    @abc.abstractmethod
    def relative_to_field(self, field):
        ...

    @abc.abstractmethod
    def relative_to_stream(self):
        ...

    @abc.abstractmethod
    def get_timestamp(self, event=None):
        ...

    @abc.abstractmethod
    def __getitem__(self, key):
        ...

    @abc.abstractmethod
    def keys(self):
        ...

    @abc.abstractmethod
    def on_set_key(self, key, value):
        ...

    @abc.abstractmethod
    def on_del_key(self, key):
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any):
        ...

    @property
    def get_relative_timestamp(self):
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to):
        ...