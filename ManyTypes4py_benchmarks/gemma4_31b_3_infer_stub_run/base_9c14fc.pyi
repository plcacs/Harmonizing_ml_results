import abc
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator, List, Mapping, MutableMapping, MutableSet, Optional, Set, Tuple, Union, cast
from yarl import URL
from faust.types import AppT, TopicT, TP
from faust.types.tables import CollectionT
from faust.types.windows import WindowT
from mode import Service

TABLE_CLEANING: str
E_SOURCE_PARTITIONS_MISMATCH: str

class Collection(Service, CollectionT):
    """Base class for changelog-backed data structures stored in Kafka."""
    _data: Any
    _changelog_compacting: bool
    _changelog_deleting: Optional[bool]

    app: AppT
    name: str
    default: Any
    schema: Any
    key_type: Any
    value_type: Any
    partitions: Optional[int]
    window: Optional[WindowT]
    extra_topic_configs: Mapping[str, Any]
    help: str
    recovery_buffer_size: int
    standby_buffer_size: int
    use_partitioner: bool
    last_closed_window: float
    is_global: bool
    options: Any
    key_serializer: str
    value_serializer: str
    _partition_timestamp_keys: MutableMapping[Tuple[int, float], Set[Any]]
    _partition_timestamps: MutableMapping[int, List[float]]
    _partition_latest_timestamp: MutableMapping[int, int]
    _recover_callbacks: Set[Callable[[], Any]]

    def __init__(
        self,
        app: AppT,
        *,
        name: Optional[str] = None,
        default: Any = None,
        store: Optional[str] = None,
        schema: Any = None,
        key_type: Any = None,
        value_type: Any = None,
        partitions: Optional[int] = None,
        window: Optional[WindowT] = None,
        changelog_topic: Optional[TopicT] = None,
        help: Optional[str] = None,
        on_recover: Optional[Callable[[Callable], Any]] = None,
        on_changelog_event: Optional[Callable[[Any], Any]] = None,
        recovery_buffer_size: int = 1000,
        standby_buffer_size: Optional[int] = None,
        extra_topic_configs: Optional[Mapping[str, Any]] = None,
        recover_callbacks: Optional[Iterable[Callable[[], Any]]] = None,
        options: Any = None,
        use_partitioner: bool = False,
        on_window_close: Optional[Callable[[Any, Any], Any]] = None,
        is_global: bool = False,
        **kwargs: Any,
    ) -> None: ...

    @abc.abstractmethod
    def _has_key(self, key: Any) -> bool: ...

    @abc.abstractmethod
    def _get_key(self, key: Any) -> Any: ...

    @abc.abstractmethod
    def _set_key(self, key: Any, value: Any) -> None: ...

    @abc.abstractmethod
    def _del_key(self, key: Any) -> None: ...

    def _serializer_from_type(self, typ: Any) -> str: ...

    def __hash__(self) -> int: ...

    def _new_store(self) -> Any: ...

    def _new_store_by_url(self, url: str) -> Any: ...

    @property
    def data(self) -> Any: ...

    async def on_start(self) -> None: ...

    def on_recover(self, fun: Callable[[], Any]) -> Callable[[], Any]: ...

    def info(self) -> Mapping[str, Any]: ...

    def persisted_offset(self, tp: TP) -> int: ...

    async def need_active_standby_for(self, tp: TP) -> bool: ...

    def reset_state(self) -> None: ...

    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: Optional[str] = None,
        value_serializer: Optional[str] = None,
    ) -> Any: ...

    def _send_changelog(
        self,
        event: Any,
        key: Any,
        value: Any,
        key_serializer: Optional[str] = None,
        value_serializer: Optional[str] = None,
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

    def join(self, *fields: Any) -> Any: ...

    def left_join(self, *fields: Any) -> Any: ...

    def inner_join(self, *fields: Any) -> Any: ...

    def outer_join(self, *fields: Any) -> Any: ...

    def _join(self, join_strategy: Any) -> Any: ...

    def clone(self, **kwargs: Any) -> 'Collection': ...

    def combine(self, *nodes: Any, **kwargs: Any) -> Any: ...

    def contribute_to_stream(self, active: Any) -> None: ...

    async def remove_from_stream(self, stream: Any) -> None: ...

    def _new_changelog_topic(
        self,
        *,
        retention: Optional[float] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> TopicT: ...

    def __copy__(self) -> 'Collection': ...

    def __and__(self, other: Any) -> Any: ...

    def _apply_window_op(self, op: Callable[[Any, Any], Any], key: Any, value: Any, timestamp: float) -> None: ...

    def _set_windowed(self, key: Any, value: Any, timestamp: float) -> None: ...

    def _del_windowed(self, key: Any, timestamp: float) -> None: ...

    def _window_ranges(self, timestamp: float) -> Iterator[Tuple[float, float]]: ...

    def _relative_now(self, event: Any = None) -> float: ...

    def _relative_event(self, event: Any = None) -> float: ...

    def _relative_field(self, field: Any) -> Callable[[Optional[Any]], Any]: ...

    def _relative_timestamp(self, timestamp: float) -> Callable[[Optional[Any]], float]: ...

    def _windowed_now(self, key: Any) -> Any: ...

    def _windowed_timestamp(self, key: Any, timestamp: float) -> Any: ...

    def _windowed_contains(self, key: Any, timestamp: float) -> bool: ...

    def _windowed_delta(self, key: Any, d: float, event: Any = None) -> Any: ...

    async def on_rebalance(self, assigned: Any, revoked: Any, newly_assigned: Any) -> None: ...

    async def on_recovery_completed(self, active_tps: Any, standby_tps: Any) -> None: ...

    async def call_recover_callbacks(self) -> None: ...

    async def on_changelog_event(self, event: Any) -> None: ...

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

    def apply_changelog_batch(self, batch: Any) -> None: ...

    def _to_key(self, k: Any) -> Any: ...

    def _to_value(self, v: Any) -> Any: ...

    def _human_channel(self) -> str: ...

    def _repr_info(self) -> str: ...