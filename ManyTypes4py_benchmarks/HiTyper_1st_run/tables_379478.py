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
    def __init__(self, app, *, name=None, default=None, store=None, schema=None, key_type=None, value_type=None, partitions=None, window=None, changelog_topic=None, help=None, on_recover=None, on_changelog_event=None, recovery_buffer_size=1000, standby_buffer_size=None, extra_topic_configs=None, options=None, use_partitioner=False, on_window_close=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs) -> None:
        ...

    @property
    @abc.abstractmethod
    def changelog_topic(self) -> None:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic) -> None:
        ...

    @abc.abstractmethod
    def _changelog_topic_name(self) -> None:
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch: Union[list[list[str]], tuple]) -> None:
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp: Union[tuples.TP, typing.Type, None]) -> None:
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp):
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        ...

    @abc.abstractmethod
    def send_changelog(self, partition: Union[bool, str], key: Union[bool, str], value: Union[bool, str], key_serializer: Union[None, bool, str]=None, value_serializer: Union[None, bool, str]=None) -> None:
        ...

    @abc.abstractmethod
    def partition_for_key(self, key: Union[list, str, bytes]) -> None:
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
    def on_recover(self, fun: typing.Callable) -> None:
        ...

    @abc.abstractmethod
    async def on_recovery_completed(self, active_tps, standby_tps):
        ...

    @abc.abstractmethod
    async def call_recover_callbacks(self):
        ...

    @abc.abstractmethod
    def using_window(self, window: Union[int, str], *, key_index: bool=False) -> None:
        ...

    @abc.abstractmethod
    def hopping(self, size: Union[int, None], step: Union[int, None], expires: Union[None, int]=None, key_index: bool=False) -> None:
        ...

    @abc.abstractmethod
    def tumbling(self, size: Union[int, None, transfer.models.ChannelID], expires: Union[None, int, transfer.models.ChannelID]=None, key_index: bool=False) -> None:
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def _relative_now(self, event: Union[None, events.Event, tonga.models.records.command.command.BaseCommand, events.events_base.EventType]=None) -> None:
        ...

    @abc.abstractmethod
    def _relative_event(self, event: Union[None, dict[str, typing.Any], list[aw_core.models.Event], tonga.models.records.command.command.BaseCommand]=None) -> None:
        ...

    @abc.abstractmethod
    def _relative_field(self, field: Union[str, models.FieldDescriptorT, list[str]]) -> None:
        ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp: Union[float, str, int]) -> None:
        ...

    @abc.abstractmethod
    def _windowed_contains(self, key: Union[bytes, float, int, str], timestamp: Union[bytes, float, int, str]) -> None:
        ...

class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...

class GlobalTableT(TableT):
    ...

class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):

    @abc.abstractmethod
    def __init__(self, app, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def add(self, table: Union[str, set[faustypes.TP]]) -> None:
        ...

    @abc.abstractmethod
    def persist_offset_on_commit(self, store: Union[int, typing.Callable], tp: Union[int, typing.Callable], offset: Union[int, typing.Callable]) -> None:
        ...

    @abc.abstractmethod
    def on_commit(self, offsets: Union[typing.MutableMapping, int]) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(self, assigned, revoked, newly_assigned):
        ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked: Union[set[faustypes.TP], typing.Callable]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self):
        ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self):
        ...

    @property
    @abc.abstractmethod
    def changelog_topics(self) -> None:
        ...

class WindowSetT(FastUserDict[KT, VT]):

    @abc.abstractmethod
    def __init__(self, key, table, wrapper, event: Union[None, events.EventT, str, list[dict[str, typing.Any]]]=None) -> None:
        ...

    @abc.abstractmethod
    def apply(self, op: Any, value: Any, event: None=None) -> None:
        ...

    @abc.abstractmethod
    def value(self, event: Union[None, typing.Callable, dict]=None) -> None:
        ...

    @abc.abstractmethod
    def current(self, event: Union[None, str, list, watchdog.events.FileSystemEvent]=None) -> None:
        ...

    @abc.abstractmethod
    def now(self) -> None:
        ...

    @abc.abstractmethod
    def delta(self, d: Union[int, str, float], event: Union[None, int, str, float]=None) -> None:
        ...

    @abc.abstractmethod
    def __iadd__(self, other: Union[list[str], SupportsFloat, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __isub__(self, other: Union[typing.AbstractSet, SupportsFloat, list[str]]) -> None:
        ...

    @abc.abstractmethod
    def __imul__(self, other: Union[SupportsFloat, list[str], datetime.timedelta]) -> None:
        ...

    @abc.abstractmethod
    def __itruediv__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __ifloordiv__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __imod__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __ipow__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __ilshift__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __irshift__(self, other: Union[datetime.timedelta, typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __iand__(self, other: Union[SupportsFloat, list[str], typing.AbstractSet]) -> None:
        ...

    @abc.abstractmethod
    def __ixor__(self, other: Union[typing.AbstractSet, typing.Iterable[T], datetime.timedelta]) -> None:
        ...

    @abc.abstractmethod
    def __ior__(self, other: typing.Iterable[T]) -> None:
        ...

class WindowedItemsViewT(ItemsView):

    @abc.abstractmethod
    def __init__(self, mapping: Union[events.EventT, str, list[dict[str, typing.Any]]], event: Union[None, events.EventT, str, list[dict[str, typing.Any]]]=None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> None:
        ...

    @abc.abstractmethod
    def now(self) -> None:
        ...

    @abc.abstractmethod
    def current(self, event: Union[None, str, list, watchdog.events.FileSystemEvent]=None) -> None:
        ...

    @abc.abstractmethod
    def delta(self, d: Union[int, str, float], event: Union[None, int, str, float]=None) -> None:
        ...

class WindowedValuesViewT(ValuesView):

    @abc.abstractmethod
    def __init__(self, mapping: Union[events.EventT, str, list[dict[str, typing.Any]]], event: Union[None, events.EventT, str, list[dict[str, typing.Any]]]=None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> None:
        ...

    @abc.abstractmethod
    def now(self) -> None:
        ...

    @abc.abstractmethod
    def current(self, event: Union[None, str, list, watchdog.events.FileSystemEvent]=None) -> None:
        ...

    @abc.abstractmethod
    def delta(self, d: Union[int, str, float], event: Union[None, int, str, float]=None) -> None:
        ...

class WindowWrapperT(MutableMapping):

    @abc.abstractmethod
    def __init__(self, table, *, relative_to=None, key_index=False, key_index_table=None) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, relative_to: Union[int, list, str]) -> None:
        ...

    @abc.abstractmethod
    def relative_to_now(self) -> None:
        ...

    @abc.abstractmethod
    def relative_to_field(self, field: Union[str, list[str], models.FieldDescriptorT]) -> None:
        ...

    @abc.abstractmethod
    def relative_to_stream(self) -> None:
        ...

    @abc.abstractmethod
    def get_timestamp(self, event: Union[None, list[aw_core.models.Event], dict]=None) -> None:
        ...

    @abc.abstractmethod
    def __getitem__(self, key: Union[int, slice, str]) -> None:
        ...

    @abc.abstractmethod
    def keys(self) -> None:
        ...

    @abc.abstractmethod
    def on_set_key(self, key: Union[bytes, VT, KT], value: Union[bytes, VT, KT]) -> None:
        ...

    @abc.abstractmethod
    def on_del_key(self, key: Union[KT, bytes, str]) -> None:
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs) -> None:
        ...

    @property
    def get_relative_timestamp(self) -> None:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to) -> None:
        ...