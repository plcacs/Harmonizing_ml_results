    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Any) -> None:
    def gauge(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    def increment(self, metric: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    def incr(self, metric: str, count: int = 1) -> None:
    def decrement(self, metric: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> Any:
    def decr(self, metric: str, count: float = 1.0) -> None:
    def timing(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    def timed(self, metric: Optional[str] = None, labels: Optional[Dict[str, str]] = None, use_ms: Optional[bool] = None) -> Any:
    def histogram(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    def _encode_labels(self, labels: Optional[Dict[str, str]]) -> Optional[List[str]]:
    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Any) -> None:
    def _new_datadog_stats_client(self) -> DatadogStatsClient:
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Any:
    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Any] = None) -> None:
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
    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[str, Any], *, view: Optional[str] = None) -> None:
    def _format_label(self, tp: Optional[TP] = None, stream: Optional[StreamT] = None, table: Optional[CollectionT] = None) -> Dict[str, str]:
    def _format_tp_label(self, tp: TP) -> Dict[str, str]:
    def _format_stream_label(self, stream: StreamT) -> Dict[str, str]:
    def _stream_label(self, stream: StreamT) -> str:
    def _format_table_label(self, table: CollectionT) -> Dict[str, str]:
    @cached_property
    def client(self) -> DatadogStatsClient:
