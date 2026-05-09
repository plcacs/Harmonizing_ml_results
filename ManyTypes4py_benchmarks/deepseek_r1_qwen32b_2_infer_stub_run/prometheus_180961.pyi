"""Monitor using Promethus."""
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
from prometheus_client import Counter, Gauge, Histogram
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
from aiohttp.web import Response
from faust.web import Request

__all__: List[str] = ['PrometheusMonitor']

class PrometheusMonitor:
    """
    Prometheus Faust Sensor.
    """
    ERROR: str
    COMPLETED: str
    KEYS_RETRIEVED: str
    KEYS_UPDATED: str
    KEYS_DELETED: str

    def __init__(self, app: AppT, pattern: str = '/metrics', **kwargs: Any) -> None:
        ...
    
    def _initialize_metrics(self) -> None:
        ...

    messages_received: Counter
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

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        ...
    
    def on_stream_event_in(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
    ) -> Dict[str, Any]:
        ...
    
    def _stream_label(self, stream: StreamT) -> str:
        ...
    
    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...
    
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        ...
    
    def on_table_get(self, table: CollectionT, key: Any) -> None:
        ...
    
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        ...
    
    def on_table_del(self, table: CollectionT, key: Any) -> None:
        ...
    
    def on_commit_completed(self, consumer: ConsumerT, state: Dict[str, Any]) -> None:
        ...
    
    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: PendingMessage,
        keysize: int,
        valsize: int,
    ) -> None:
        ...
    
    def on_send_completed(
        self,
        producer: ProducerT,
        state: Dict[str, Any],
        metadata: RecordMetadata,
    ) -> None:
        ...
    
    def on_send_error(
        self,
        producer: ProducerT,
        exc: BaseException,
        state: Dict[str, Any],
    ) -> None:
        ...
    
    def on_assignment_error(
        self,
        assignor: PartitionAssignorT,
        state: Dict[str, Any],
        exc: BaseException,
    ) -> None:
        ...
    
    def on_assignment_completed(
        self,
        assignor: PartitionAssignorT,
        state: Dict[str, Any],
    ) -> None:
        ...
    
    def on_rebalance_start(self, app: AppT) -> Dict[str, Any]:
        ...
    
    def on_rebalance_return(
        self,
        app: AppT,
        state: Dict[str, Any],
    ) -> None:
        ...
    
    def on_rebalance_end(
        self,
        app: AppT,
        state: Dict[str, Any],
    ) -> None:
        ...
    
    def count(self, metric_name: str, count: int = 1) -> None:
        ...
    
    def on_tp_commit(self, tp_offsets: Dict[TP, int]) -> None:
        ...
    
    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
        ...
    
    def on_web_request_end(
        self,
        app: AppT,
        request: Request,
        response: Response,
        state: Dict[str, Any],
        view: Any = None,
    ) -> None:
        ...
    
    def expose_metrics(self) -> None:
        ...

    async def metrics_handler(self, request: Request) -> Response:
        ...