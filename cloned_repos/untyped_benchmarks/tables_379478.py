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
    is_global = False

    @abc.abstractmethod
    def __init__(self, app, *, name=None, default=None, store=None, schema=None, key_type=None, value_type=None, partitions=None, window=None, changelog_topic=None, help=None, on_recover=None, on_changelog_event=None, recovery_buffer_size=1000, standby_buffer_size=None, extra_topic_configs=None, options=None, use_partitioner=False, on_window_close=None, **kwargs):
        ...

    @abc.abstractmethod
    def clone(self, **kwargs):
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
    async def need_active_standby_for(self, tp):
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
    async def on_window_close(self, key, value):
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    async def on_changelog_event(self, event):
        ...

    @abc.abstractmethod
    def on_recover(self, fun):
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps, standby_tps):
        ...

    @abc.abstractmethod
    async def call_recover_callbacks(self):
        ...

    @abc.abstractmethod
    def using_window(self, window, *, key_index=False):
        ...

    @abc.abstractmethod
    def hopping(self, size, step, expires=None, key_index=False):
        ...

    @abc.abstractmethod
    def tumbling(self, size, expires=None, key_index=False):
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs):
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

    @abc.abstractmethod
    def __init__(self, app, **kwargs):
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
    async def on_rebalance(self, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked):
        ...

    @abc.abstractmethod
    def on_rebalance_start(self):
        ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self):
        ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self):
        ...

    @property
    @abc.abstractmethod
    def changelog_topics(self):
        ...

class WindowSetT(FastUserDict[KT, VT]):

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

    @abc.abstractmethod
    def __init__(self, table, *, relative_to=None, key_index=False, key_index_table=None):
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
    def as_ansitable(self, **kwargs):
        ...

    @property
    def get_relative_timestamp(self):
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to):
        ...