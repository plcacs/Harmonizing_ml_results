from faust import web
from aiohttp.web import Response
from faust.types import AppT, MonitorT, TP
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY

class PrometheusMonitor(Monitor):
    def __init__(self, app: AppT, pattern: str = '/metrics', **kwargs: Any) -> None:
    def _initialize_metrics(self) -> None:
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Any:
    def _stream_label(self, stream: StreamT) -> str:
    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Any = None) -> None:
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
    def on_table_get(self, table: CollectionT, key: Any) -> None:
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
    def on_table_del(self, table: CollectionT, key: Any) -> None:
    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None:
    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> Any:
    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None:
    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None:
    def on_assignment_error(self, assignor: PartitionAssignorT, state: Any, exc: Exception) -> None:
    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Any) -> None:
    def on_rebalance_start(self, app: AppT) -> Any:
    def on_rebalance_return(self, app: AppT, state: Any) -> None:
    def on_rebalance_end(self, app: AppT, state: Any) -> None:
    def count(self, metric_name: str, count: int = 1) -> None:
    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None:
    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
    def on_web_request_end(self, app: AppT, request: Any, response: Any, state: Any, *, view: Any = None) -> None:
    def expose_metrics(self) -> None:
