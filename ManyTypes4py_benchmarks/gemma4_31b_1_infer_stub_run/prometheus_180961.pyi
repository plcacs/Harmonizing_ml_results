"""Monitor using Promethus."""
import typing
from typing import Any, Optional, Union
from faust.types import AppT, CollectionT, EventT, Message, PendingMessage, RecordMetadata, StreamT, TP
from faust.types.transports import ConsumerT, ProducerT
from faust.types.assignor import PartitionAssignorT
from aiohttp.web import Response
from prometheus_client import Counter, Gauge, Histogram
from .monitor import Monitor, TPOffsetMapping

__all__ = ['PrometheusMonitor']

class PrometheusMonitor(Monitor):
    """
    Prometheus Faust Sensor.

    This sensor, records statistics using prometheus_client and expose
    them using the aiohttp server running under /metrics by default

    Usage:
        import faust
        from faust.sensors.prometheus import PrometheusMonitor

        app = faust.App('example', broker='kafka://')
        app.monitor = PrometheusMonitor(app, pattern='/metrics')
    """
    ERROR: str
    COMPLETED: str
    KEYS_RETRIEVED: str
    KEYS_UPDATED: str
    KEYS_DELETED: str

    messages_received: Counter
    active_messages: Gauge
    messages_received_per_topics: Counter
    messages_received_per_topics_partition: Gauge
    events_runtime_latency: Histogram
    total_events: Counter
    total_active_events: Gauge
    total_events_per_stream: Counter
    table_operations: Counter
    topic_messages_sent: Counter
    total_sent_messages: Counter
    producer_send_latency: Histogram
    total_error_messages_sent: Counter
    producer_error_send_latency: Histogram
    assignment_operations: Counter
    assign_latency: Histogram
    total_rebalances: Gauge
    total_rebalances_recovering: Gauge
    revalance_done_consumer_latency: Histogram
    revalance_done_latency: Histogram
    count_metrics_by_name: Gauge
    http_status_codes: Counter
    http_latency: Histogram
    topic_partition_end_offset: Gauge
    topic_partition_offset_commited: Gauge
    consumer_commit_latency: Histogram

    def __init__(self, app: AppT, pattern: str = '/metrics', **kwargs: Any) -> None: ...

    def _initialize_metrics(self) -> None: ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Any: ...

    def _stream_label(self, stream: StreamT) -> str: ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Any] = None) -> None: ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_table_get(self, table: CollectionT, key: Any) -> None: ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None: ...

    def on_table_del(self, table: CollectionT, key: Any) -> None: ...

    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None: ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: Any, keysize: int, valsize: int) -> Any: ...

    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None: ...

    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None: ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: typing.Dict[str, Any], exc: Exception) -> None: ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: typing.Dict[str, Any]) -> None: ...

    def on_rebalance_start(self, app: AppT) -> Any: ...

    def on_rebalance_return(self, app: AppT, state: typing.Dict[str, Any]) -> None: ...

    def on_rebalance_end(self, app: AppT, state: typing.Dict[str, Any]) -> None: ...

    def count(self, metric_name: str, count: Union[int, float] = 1) -> None: ...

    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None: ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...

    def on_web_request_end(self, app: AppT, request: Any, response: Response, state: typing.Dict[str, Any], *, view: Optional[Any] = None) -> None: ...

    def expose_metrics(self) -> None: ...