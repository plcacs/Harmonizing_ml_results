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

__all__ = [
    'RecoverCallback', 'RelativeArg', 'CollectionT', 'TableT', 'GlobalTableT',
    'TableManagerT', 'WindowCloseCallback', 'WindowSetT', 'WindowedItemsViewT',
    'WindowedValuesViewT', 'WindowWrapperT', 'ChangelogEventCallback', 'CollectionTps'
]

RelativeHandler = Callable[[Optional[EventT]], Union[float, datetime]]
RecoverCallback = Callable[[], Awaitable[None]]
ChangelogEventCallback = Callable[[EventT], Awaitable[None]]
WindowCloseCallback = Callable[[Any, Any], Union[None, Awaitable[None]]]
RelativeArg = Optional[Union["_FieldDescriptorT", RelativeHandler, datetime, float]]
CollectionTps = MutableMapping['CollectionT', Set[TP]]
KT = TypeVar('KT')
VT = TypeVar('VT')


class CollectionT(ServiceT, JoinableT):
    is_global: bool = False

    @abc.abstractmethod
    def __init__(
        self,
        app: "_AppT",
        *,
        name: Optional[str] = None,
        default: Any = None,
        store: Optional[StoreT] = None,
        schema: Optional["_SchemaT"] = None,
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
        extra_topic_configs: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
        use_partitioner: bool = False,
        on_window_close: Optional[WindowCloseCallback] = None,
        **kwargs: Any
    ) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> "CollectionT":
        ...

    @property
    @abc.abstractmethod
    def changelog_topic(self) -> Optional[TopicT]:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None:
        ...

    @abc.abstractmethod
    def _changelog_topic_name(self) -> str:
        ...

    @abc.abstractmethod
    def apply_changelog_batch(self, batch: Iterable[Any]) -> None:
        ...

    @abc.abstractmethod
    def persisted_offset(self, tp: TP) -> Optional[int]:
        ...

    @abc.abstractmethod
    async def need_active_standby_for(self, tp: TP) -> None:
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        ...

    @abc.abstractmethod
    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: Optional[Callable[[Any], bytes]] = None,
        value_serializer: Optional[Callable[[Any], bytes]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def partition_for_key(self, key: Any) -> int:
        ...

    @abc.abstractmethod
    async def on_window_close(self, key: Any, value: Any) -> Union[None, Awaitable[None]]:
        ...

    @abc.abstractmethod
    async def on_rebalance(
        self,
        assigned: Iterable[TP],
        revoked: Iterable[TP],
        newly_assigned: Iterable[TP]
    ) -> None:
        ...

    @abc.abstractmethod
    async def on_changelog_event(self, event: EventT) -> None:
        ...

    @abc.abstractmethod
    def on_recover(self, fun: RecoverCallback) -> None:
        ...

    @abc.abstractmethod
    async def on_recovery_completed(
        self,
        active_tps: CollectionTps,
        standby_tps: CollectionTps
    ) -> None:
        ...

    @abc.abstractmethod
    async def call_recover_callbacks(self) -> None:
        ...

    @abc.abstractmethod
    def using_window(self, window: WindowT, *, key_index: bool = False) -> "CollectionT":
        ...

    @abc.abstractmethod
    def hopping(self, size: Seconds, step: Seconds, expires: Optional[Seconds] = None, key_index: bool = False) -> "CollectionT":
        ...

    @abc.abstractmethod
    def tumbling(self, size: Seconds, expires: Optional[Seconds] = None, key_index: bool = False) -> "CollectionT":
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    @abc.abstractmethod
    def _relative_now(self, event: Optional[EventT] = None) -> Union[float, datetime]:
        ...

    @abc.abstractmethod
    def _relative_event(self, event: Optional[EventT] = None) -> Union[float, datetime]:
        ...

    @abc.abstractmethod
    def _relative_field(self, field: "_FieldDescriptorT") -> RelativeArg:
        ...

    @abc.abstractmethod
    def _relative_timestamp(self, timestamp: Union[float, datetime]) -> Union[float, datetime]:
        ...

    @abc.abstractmethod
    def _windowed_contains(self, key: Any, timestamp: datetime) -> bool:
        ...


class TableT(CollectionT, ManagedUserDict[KT, VT]):
    ...


class GlobalTableT(TableT):
    ...


class TableManagerT(ServiceT, FastUserDict[str, CollectionT]):

    @abc.abstractmethod
    def __init__(self, app: "_AppT", **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def add(self, table: CollectionT) -> None:
        ...

    @abc.abstractmethod
    def persist_offset_on_commit(self, store: StoreT, tp: TP, offset: int) -> None:
        ...

    @abc.abstractmethod
    def on_commit(self, offsets: Mapping[TP, int]) -> None:
        ...

    @abc.abstractmethod
    async def on_rebalance(
        self,
        assigned: Iterable[TP],
        revoked: Iterable[TP],
        newly_assigned: Iterable[TP]
    ) -> None:
        ...

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked: Iterable[TP]) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_tables_registered(self) -> None:
        ...

    @abc.abstractmethod
    async def wait_until_recovery_completed(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def changelog_topics(self) -> Mapping[str, TopicT]:
        ...


class WindowSetT(FastUserDict[KT, VT]):

    @abc.abstractmethod
    def __init__(self, key: KT, table: CollectionT, wrapper: "WindowWrapperT", event: Optional[EventT] = None) -> None:
        ...

    @abc.abstractmethod
    def apply(self, op: str, value: Any, event: Optional[EventT] = None) -> None:
        ...

    @abc.abstractmethod
    def value(self, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def now(self) -> Any:
        ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def __iadd__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __isub__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __imul__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __itruediv__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __ifloordiv__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __imod__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __ipow__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __ilshift__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __irshift__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __iand__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __ixor__(self, other: Any) -> "WindowSetT":
        ...

    @abc.abstractmethod
    def __ior__(self, other: Any) -> "WindowSetT":
        ...


class WindowedItemsViewT(ItemsView):

    @abc.abstractmethod
    def __init__(self, mapping: Mapping[Any, Any], event: Optional[EventT] = None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        ...

    @abc.abstractmethod
    def now(self) -> Any:
        ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Optional[EventT] = None) -> Any:
        ...


class WindowedValuesViewT(ValuesView):

    @abc.abstractmethod
    def __init__(self, mapping: Mapping[Any, Any], event: Optional[EventT] = None) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abc.abstractmethod
    def now(self) -> Any:
        ...

    @abc.abstractmethod
    def current(self, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def delta(self, d: Any, event: Optional[EventT] = None) -> Any:
        ...


class WindowWrapperT(MutableMapping):

    @abc.abstractmethod
    def __init__(
        self,
        table: CollectionT,
        *,
        relative_to: Optional[Union["_FieldDescriptorT", RelativeHandler, datetime, float]] = None,
        key_index: bool = False,
        key_index_table: Optional[Any] = None
    ) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def clone(self, relative_to: Union["_FieldDescriptorT", RelativeHandler, datetime, float]) -> "WindowWrapperT":
        ...

    @abc.abstractmethod
    def relative_to_now(self) -> Any:
        ...

    @abc.abstractmethod
    def relative_to_field(self, field: "_FieldDescriptorT") -> Any:
        ...

    @abc.abstractmethod
    def relative_to_stream(self) -> Any:
        ...

    @abc.abstractmethod
    def get_timestamp(self, event: Optional[EventT] = None) -> Any:
        ...

    @abc.abstractmethod
    def __getitem__(self, key: Any) -> Any:
        ...

    @abc.abstractmethod
    def keys(self) -> KeysView[Any]:
        ...

    @abc.abstractmethod
    def on_set_key(self, key: Any, value: Any) -> None:
        ...

    @abc.abstractmethod
    def on_del_key(self, key: Any) -> None:
        ...

    @abc.abstractmethod
    def as_ansitable(self, **kwargs: Any) -> Any:
        ...

    @property
    def get_relative_timestamp(self) -> Any:
        ...

    @get_relative_timestamp.setter
    def get_relative_timestamp(self, relative_to: Any) -> None:
        ...