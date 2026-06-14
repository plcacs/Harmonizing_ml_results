import abc
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    Union,
)

from mode import Service
from yarl import URL

from faust.types import (
    AppT,
    CodecArg,
    EventT,
    FieldDescriptorT,
    FutureMessage,
    JoinT,
    RecordMetadata,
    SchemaT,
    TP,
    TopicT,
)
from faust.types.models import ModelArg, ModelT
from faust.types.stores import StoreT
from faust.types.streams import JoinableT, StreamT
from faust.types.tables import (
    ChangelogEventCallback,
    CollectionT,
    RecoverCallback,
    RelativeHandler,
    WindowCloseCallback,
)
from faust.types.windows import WindowRange, WindowT

__all__ = ['Collection']

TABLE_CLEANING: str
E_SOURCE_PARTITIONS_MISMATCH: str

class Collection(Service, CollectionT):
    _data: Optional[StoreT]
    _changelog_compacting: bool
    _changelog_deleting: Optional[bool]

    app: AppT
    name: str
    default: Optional[Callable[[], Any]]
    _store: Optional[URL]
    schema: Optional[SchemaT]
    key_type: Optional[ModelArg]
    value_type: Optional[ModelArg]
    partitions: Optional[int]
    window: Optional[WindowT]
    _changelog_topic: Optional[TopicT]
    extra_topic_configs: Mapping[str, Any]
    help: str
    _on_changelog_event: Optional[ChangelogEventCallback]
    recovery_buffer_size: int
    standby_buffer_size: int
    use_partitioner: bool
    _on_window_close: Optional[WindowCloseCallback]
    last_closed_window: float
    is_global: bool
    options: Optional[Mapping[str, Any]]
    key_serializer: Optional[CodecArg]
    value_serializer: Optional[CodecArg]
    _partition_timestamp_keys: defaultdict[Any, set]
    _partition_timestamps: defaultdict[Any, list]
    _partition_latest_timestamp: defaultdict[Any, int]
    _recover_callbacks: MutableSet[RecoverCallback]
    _sensor_on_get: Any
    _sensor_on_set: Any
    _sensor_on_del: Any

    @abc.abstractmethod
    def _has_key(self, key: Any) -> bool: ...
    @abc.abstractmethod
    def _get_key(self, key: Any) -> Any: ...
    @abc.abstractmethod
    def _set_key(self, key: Any, value: Any) -> None: ...
    @abc.abstractmethod
    def _del_key(self, key: Any) -> None: ...

    def __init__(
        self,
        app: AppT,
        *,
        name: Optional[str] = ...,
        default: Optional[Callable[[], Any]] = ...,
        store: Optional[Union[str, URL]] = ...,
        schema: Optional[SchemaT] = ...,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        partitions: Optional[int] = ...,
        window: Optional[WindowT] = ...,
        changelog_topic: Optional[TopicT] = ...,
        help: Optional[str] = ...,
        on_recover: Optional[RecoverCallback] = ...,
        on_changelog_event: Optional[ChangelogEventCallback] = ...,
        recovery_buffer_size: int = ...,
        standby_buffer_size: Optional[int] = ...,
        extra_topic_configs: Optional[Mapping[str, Any]] = ...,
        recover_callbacks: Optional[Iterable[RecoverCallback]] = ...,
        options: Optional[Mapping[str, Any]] = ...,
        use_partitioner: bool = ...,
        on_window_close: Optional[WindowCloseCallback] = ...,
        is_global: bool = ...,
        **kwargs: Any,
    ) -> None: ...

    def _serializer_from_type(self, typ: Optional[ModelArg]) -> Optional[CodecArg]: ...
    def __hash__(self) -> int: ...
    def _new_store(self) -> StoreT: ...
    def _new_store_by_url(self, url: Union[str, URL]) -> StoreT: ...

    @property
    def data(self) -> StoreT: ...

    async def on_start(self) -> None: ...
    def on_recover(self, fun: RecoverCallback) -> RecoverCallback: ...
    def info(self) -> Mapping[str, Any]: ...
    def persisted_offset(self, tp: TP) -> Optional[int]: ...
    async def need_active_standby_for(self, tp: TP) -> bool: ...
    def reset_state(self) -> None: ...

    def send_changelog(
        self,
        partition: Optional[int],
        key: Any,
        value: Any,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
    ) -> FutureMessage: ...

    def _send_changelog(
        self,
        event: Optional[EventT],
        key: Any,
        value: Any,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
    ) -> None: ...

    def partition_for_key(self, key: Any) -> Optional[int]: ...
    def _verify_source_topic_partitions(self, source_topic: str) -> None: ...
    def _on_changelog_sent(self, fut: Any) -> None: ...

    async def _clean_data(self) -> None: ...
    async def _del_old_keys(self) -> None: ...
    async def on_window_close(self, key: Any, value: Any) -> None: ...
    def _should_expire_keys(self) -> bool: ...
    def _maybe_set_key_ttl(self, key: Any, partition: int) -> None: ...
    def _maybe_del_key_ttl(self, key: Any, partition: int) -> None: ...
    def _changelog_topic_name(self) -> str: ...

    def join(self, *fields: FieldDescriptorT) -> StreamT: ...
    def left_join(self, *fields: FieldDescriptorT) -> StreamT: ...
    def inner_join(self, *fields: FieldDescriptorT) -> StreamT: ...
    def outer_join(self, *fields: FieldDescriptorT) -> StreamT: ...
    def _join(self, join_strategy: JoinT) -> StreamT: ...

    def clone(self, **kwargs: Any) -> Any: ...
    def combine(self, *nodes: JoinableT, **kwargs: Any) -> StreamT: ...
    def contribute_to_stream(self, active: StreamT) -> None: ...
    async def remove_from_stream(self, stream: StreamT) -> None: ...

    def _new_changelog_topic(
        self,
        *,
        retention: Optional[float] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
    ) -> TopicT: ...

    def __copy__(self) -> Any: ...
    def __and__(self, other: Any) -> StreamT: ...

    def _apply_window_op(
        self, op: Callable, key: Any, value: Any, timestamp: float
    ) -> None: ...
    def _set_windowed(self, key: Any, value: Any, timestamp: float) -> None: ...
    def _del_windowed(self, key: Any, timestamp: float) -> None: ...
    def _window_ranges(self, timestamp: float) -> Iterator[WindowRange]: ...

    def _relative_now(self, event: Optional[EventT] = ...) -> float: ...
    def _relative_event(self, event: Optional[EventT] = ...) -> float: ...
    def _relative_field(self, field: FieldDescriptorT) -> RelativeHandler: ...
    def _relative_timestamp(self, timestamp: float) -> RelativeHandler: ...

    def _windowed_now(self, key: Any) -> Any: ...
    def _windowed_timestamp(self, key: Any, timestamp: float) -> Any: ...
    def _windowed_contains(self, key: Any, timestamp: float) -> bool: ...
    def _windowed_delta(self, key: Any, d: float, event: Optional[EventT] = ...) -> Any: ...

    async def on_rebalance(
        self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]
    ) -> None: ...
    async def on_recovery_completed(
        self, active_tps: Set[TP], standby_tps: Set[TP]
    ) -> None: ...
    async def call_recover_callbacks(self) -> None: ...
    async def on_changelog_event(self, event: EventT) -> None: ...

    @property
    def label(self) -> str: ...
    @property
    def shortlabel(self) -> str: ...

    @property
    def changelog_topic(self) -> TopicT: ...
    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None: ...

    @property
    def changelog_topic_name(self) -> str: ...

    def apply_changelog_batch(self, batch: Iterable[EventT]) -> None: ...
    def _to_key(self, k: Any) -> Any: ...
    def _to_value(self, v: Any) -> Any: ...
    def _human_channel(self) -> str: ...
    def _repr_info(self) -> str: ...