"""Base class Collection for Table and future data structures."""

from abc import ABC
from datetime import datetime
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
    cast,
    no_type_check,
)
from faust.types import (
    AppT,
    CodecArg,
    EventT,
    FieldDescriptorT,
    FutureMessage,
    JoinT,
    ModelArg,
    ModelT,
    RecordMetadata,
    SchemaT,
    TP,
    TopicT,
)
from faust.types.models import ModelArg, ModelT
from faust.types.stores import StoreT
from faust.types.tables import (
    ChangelogEventCallback,
    CollectionT,
    RecoverCallback,
    RelativeHandler,
    WindowCloseCallback,
)
from faust.types.windows import WindowRange, WindowT
from yarl import URL

__all__ = ['Collection']

TABLE_CLEANING = 'CLEANING'

class Collection(Service, CollectionT, ABC):
    """Base class for changelog-backed data structures stored in Kafka."""
    _data: Optional[StoreT] = None
    _changelog_compacting: bool = True
    _changelog_deleting: Optional[bool] = None

    @abc.abstractmethod
    def _has_key(self, key: Any) -> bool:
        ...

    @abc.abstractmethod
    def _get_key(self, key: Any) -> Any:
        ...

    @abc.abstractmethod
    def _set_key(self, key: Any, value: Any) -> None:
        ...

    @abc.abstractmethod
    def _del_key(self, key: Any) -> None:
        ...

    def __init__(
        self,
        app: AppT,
        *,
        name: Optional[str] = None,
        default: Any = None,
        store: Optional[URL] = None,
        schema: Optional[SchemaT] = None,
        key_type: ModelArg = None,
        value_type: ModelArg = None,
        partitions: Optional[int] = None,
        window: Optional[WindowT] = None,
        changelog_topic: Optional[TopicT] = None,
        help: Optional[str] = None,
        on_recover: Optional[Callable] = None,
        on_changelog_event: Optional[ChangelogEventCallback] = None,
        recovery_buffer_size: Optional[int] = 1000,
        standby_buffer_size: Optional[int] = None,
        extra_topic_configs: Optional[Mapping[str, Any]] = None,
        recover_callbacks: Optional[List[RecoverCallback]] = None,
        options: Any = None,
        use_partitioner: bool = False,
        on_window_close: Optional[WindowCloseCallback] = None,
        is_global: bool = False,
        **kwargs: Any,
    ) -> None:
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

    def info(self) -> Mapping[str, Any]:
        ...

    def persisted_offset(self, tp: TP) -> Optional[int]:
        ...

    async def need_active_standby_for(self, tp: TP) -> bool:
        ...

    def reset_state(self) -> None:
        ...

    def send_changelog(
        self,
        partition: int,
        key: Any,
        value: Any,
        key_serializer: CodecArg = None,
        value_serializer: CodecArg = None,
    ) -> FutureMessage:
        ...

    def _send_changelog(
        self,
        event: Any,
        key: Any,
        value: Any,
        key_serializer: CodecArg = None,
        value_serializer: CodecArg = None,
    ) -> None:
        ...

    def partition_for_key(self, key: Any) -> Optional[int]:
        ...

    @lru_cache()
    def _verify_source_topic_partitions(self, source_topic: str) -> None:
        ...

    def _on_changelog_sent(self, fut: Any) -> None:
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

    def _join(self, join_strategy: Any) -> Any:
        ...

    def clone(self, **kwargs: Any) -> Any:
        ...

    def combine(self, *nodes: Any, **kwargs: Any) -> Any:
        ...

    def contribute_to_stream(self, active: Any) -> None:
        ...

    async def remove_from_stream(self, stream: Any) -> None:
        ...

    def _new_changelog_topic(
        self,
        *,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
    ) -> TopicT:
        ...

    def __copy__(self) -> Any:
        ...

    def __and__(self, other: Any) -> Any:
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

    def _relative_field(self, field: Any) -> RelativeHandler:
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

    async def on_rebalance(
        self,
        assigned: Set[TP],
        revoked: Set[TP],
        newly_assigned: Set[TP],
    ) -> None:
        ...

    async def on_recovery_completed(
        self,
        active_tps: MutableSet[TP],
        standby_tps: MutableSet[TP],
    ) -> None:
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