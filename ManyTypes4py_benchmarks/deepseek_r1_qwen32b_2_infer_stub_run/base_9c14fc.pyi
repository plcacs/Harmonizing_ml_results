"""Base class Collection for Table and future data structures."""
import abc
import time
from contextlib import suppress
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
from heapq import heappop, heappush
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Set,
    Tuple,
    Union,
    Type,
    Dict,
    cast,
    no_type_check,
)
from mode import Seconds, Service
from yarl import URL
from faust import stores, joins
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
from faust.types.models import ModelT
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


class Collection(Service, CollectionT):
    """Base class for changelog-backed data structures stored in Kafka."""
    _data: Optional[StoreT]
    _changelog_compacting: bool
    _changelog_deleting: Optional[Any]
    _partition_timestamp_keys: defaultdict[Any, MutableSet[Any]]
    _partition_timestamps: defaultdict[Any, List[Any]]
    _partition_latest_timestamp: defaultdict[Any, int]
    _recover_callbacks: Set[Callable]

    def __init__(self, app: AppT, *, name: Optional[str] = None, default: Any = None, store: Optional[URL] = None, schema: Optional[Any] = None, key_type: Optional[Type] = None, value_type: Optional[Type] = None, partitions: Optional[int] = None, window: Optional[WindowT] = None, changelog_topic: Optional[TopicT] = None, help: Optional[str] = None, on_recover: Optional[Callable] = None, on_changelog_event: Optional[ChangelogEventCallback] = None, recovery_buffer_size: int = 1000, standby_buffer_size: Optional[int] = None, extra_topic_configs: Optional[Dict[str, Any]] = None, recover_callbacks: Optional[List[Callable]] = None, options: Any = None, use_partitioner: bool = False, on_window_close: Optional[WindowCloseCallback] = None, is_global: bool = False, **kwargs: Any) -> None:
        ...

    def _serializer_from_type(self, typ: Any) -> str:
        ...

    def __hash__(self) -> int:
        ...

    def _new_store(self) -> StoreT:
        ...

    def _new_store_by_url(self, url: URL) -> StoreT:
        ...

    @property
    def data(self) -> StoreT:
        ...

    async def on_start(self) -> None:
        ...

    def on_recover(self, fun: Callable) -> Callable:
        ...

    def info(self) -> Dict[str, Any]:
        ...

    def persisted_offset(self, tp: TP) -> int:
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    def reset_state(self) -> None:
        ...

    def send_changelog(self, partition: int, key: Any, value: Any, key_serializer: Optional[str] = None, value_serializer: Optional[str] = None) -> FutureMessage:
        ...

    def _send_changelog(self, event: Optional[EventT], key: Any, value: Any, key_serializer: Optional[str] = None, value_serializer: Optional[str] = None) -> None:
        ...

    def partition_for_key(self, key: Any) -> Optional[int]:
        ...

    @lru_cache()
    def _verify_source_topic_partitions(self, source_topic: str) -> None:
        ...

    def _on_changelog_sent(self, fut: FutureMessage) -> None:
        ...

    @Service.task
    @Service.transitions_to(TABLE_CLEANING)
    async def _clean_data(self) -> None:
        ...

    async def _del_old_keys(self) -> None:
        ...

    async def on_window_close(self, key: Any, value: Any) -> None:
        ...

    def _should_expire_keys(self) -> bool:
        ...

    def _maybe_set_key_ttl(self, key: Any, partition: int) -> None:
        ...

    def _maybe_del_key_ttl(self, key: Any, partition: int) -> None:
        ...

    def _changelog_topic_name(self) -> str:
        ...

    def join(self, *fields: FieldDescriptorT) -> JoinT:
        ...

    def left_join(self, *fields: FieldDescriptorT) -> JoinT:
        ...

    def inner_join(self, *fields: FieldDescriptorT) -> JoinT:
        ...

    def outer_join(self, *fields: FieldDescriptorT) -> JoinT:
        ...

    def _join(self, join_strategy: JoinT) -> None:
        ...

    def clone(self, **kwargs: Any) -> CollectionT:
        ...

    def combine(self, *nodes: Union[CollectionT, StreamT], **kwargs: Any) -> None:
        ...

    def contribute_to_stream(self, active: bool) -> None:
        ...

    async def remove_from_stream(self, stream: StreamT) -> None:
        ...

    def _new_changelog_topic(self, *, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None) -> TopicT:
        ...

    def __copy__(self) -> CollectionT:
        ...

    def __and__(self, other: CollectionT) -> CollectionT:
        ...

    def _apply_window_op(self, op: Callable, key: Any, value: Any, timestamp: float) -> None:
        ...

    def _set_windowed(self, key: Any, value: Any, timestamp: float) -> None:
        ...

    def _del_windowed(self, key: Any, timestamp: float) -> None:
        ...

    def _window_ranges(self, timestamp: float) -> Iterable[WindowRange]:
        ...

    def _relative_now(self, event: Optional[EventT] = None) -> float:
        ...

    def _relative_event(self, event: Optional[EventT] = None) -> float:
        ...

    def _relative_field(self, field: FieldDescriptorT) -> RelativeHandler:
        ...

    def _relative_timestamp(self, timestamp: float) -> RelativeHandler:
        ...

    def _windowed_now(self, key: Any) -> Any:
        ...

    def _windowed_timestamp(self, key: Any, timestamp: float) -> Any:
        ...

    def _windowed_contains(self, key: Any, timestamp: float) -> bool:
        ...

    def _windowed_delta(self, key: Any, d: float, event: Optional[EventT] = None) -> Any:
        ...

    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    async def on_recovery_completed(self, active_tps: Set[TP], standby_tps: Set[TP]) -> None:
        ...

    async def call_recover_callbacks(self) -> None:
        ...

    async def on_changelog_event(self, event: EventT) -> None:
        ...

    @property
    def label(self) -> str:
        ...

    @property
    def shortlabel(self) -> str:
        ...

    @property
    def changelog_topic(self) -> TopicT:
        ...

    @changelog_topic.setter
    def changelog_topic(self, topic: TopicT) -> None:
        ...

    @property
    def changelog_topic_name(self) -> str:
        ...

    def apply_changelog_batch(self, batch: Iterable[EventT]) -> None:
        ...

    def _to_key(self, k: Any) -> Any:
        ...

    def _to_value(self, v: Any) -> Any:
        ...

    def _human_channel(self) -> str:
        ...

    def _repr_info(self) -> str:
        ...