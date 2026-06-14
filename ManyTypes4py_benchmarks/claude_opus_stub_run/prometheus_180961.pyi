import typing
from typing import Any, Dict, Optional, cast

from faust import web as _web
from faust.types import (
    AppT,
    CollectionT,
    EventT,
    Message,
    PendingMessage,
    RecordMetadata,
    StreamT,
    TP,
)
from faust.types.assignor import PartitionAssignorT
from faust.types.transports import ConsumerT, ProducerT

from .monitor import Monitor, TPOffsetMapping

__all__ = ["PrometheusMonitor"]

class PrometheusMonitor(Monitor):
    ERROR: str
    COMPLETED: str
    KEYS_RETRIEVED: str
    KEYS_UPDATED: str
    KEYS_DELETED: str

    app: AppT
    pattern: str

    messages_received: Any
    active_messages: Any
    messages_received_per_topics: Any
    messages_received_per_topics_partition: Any
    events_runtime_latency: Any
    total_events: Any
    total_active_events: Any
    total_events_per_stream: Any
    table_operations: Any
    topic_messages_sent: Any
    total_sent_messages: Any
    producer_send_latency: Any
    total_error_messages_sent: Any
    producer_error_send_latency: Any
    assignment_operations: Any
    assign_latency: Any
    total_rebalances: Any
    total_rebalances_recovering: Any
    revalance_done_consumer_latency: Any
    revalance_done_latency: Any
    count_metrics_by_name: Any
    http_status_codes: Any
    http_latency: Any
    topic_partition_end_offset: Any
    topic_partition_offset_commited: Any
    consumer_commit_latency: Any

    def __init__(self, app: AppT, pattern: str = "/metrics", **kwargs: Any) -> None: ...
    def _initialize_metrics(self) -> None: ...
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Dict[str, Any]: ...
    def _stream_label(self, stream: StreamT) -> str: ...
    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_table_get(self, table: CollectionT, key: Any) -> None: ...
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None: ...
    def on_table_del(self, table: CollectionT, key: Any) -> None: ...
    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None: ...
    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: PendingMessage,
        keysize: int,
        valsize: int,
    ) -> Any: ...
    def on_send_completed(
        self, producer: ProducerT, state: Any, metadata: RecordMetadata
    ) -> None: ...
    def on_send_error(
        self, producer: ProducerT, exc: BaseException, state: Any
    ) -> None: ...
    def on_assignment_error(
        self, assignor: PartitionAssignorT, state: Dict[str, Any], exc: BaseException
    ) -> None: ...
    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Dict[str, Any]
    ) -> None: ...
    def on_rebalance_start(self, app: AppT) -> Dict[str, Any]: ...
    def on_rebalance_return(self, app: AppT, state: Dict[str, Any]) -> None: ...
    def on_rebalance_end(self, app: AppT, state: Dict[str, Any]) -> None: ...
    def count(self, metric_name: str, count: int = 1) -> None: ...
    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None: ...
    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...
    def on_web_request_end(
        self,
        app: AppT,
        request: _web.Request,
        response: Optional[_web.Response],
        state: Dict[str, Any],
        *,
        view: Optional[_web.View] = None,
    ) -> None: ...
    def expose_metrics(self) -> None: ...