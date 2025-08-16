from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Pattern, Tuple, cast, Counter, Deque

class TableState(KeywordReduce):
    table: CollectionT = cast(CollectionT, None)
    keys_retrieved: int = 0
    keys_updated: int = 0
    keys_deleted: int = 0

    def __init__(self, table: CollectionT, *, keys_retrieved: int = 0, keys_updated: int = 0, keys_deleted: int = 0) -> None:
        self.table = table
        self.keys_retrieved = keys_retrieved
        self.keys_updated = keys_updated
        self.keys_deleted = keys_deleted

    def asdict(self) -> Dict[str, int]:
        return {'keys_retrieved': self.keys_retrieved, 'keys_updated': self.keys_updated, 'keys_deleted': self.keys_deleted}

class Monitor(Sensor, KeywordReduce):
    max_avg_history: int = MAX_AVG_HISTORY
    max_commit_latency_history: int = MAX_COMMIT_LATENCY_HISTORY
    max_send_latency_history: int = MAX_SEND_LATENCY_HISTORY
    max_assignment_latency_history: int = MAX_ASSIGNMENT_LATENCY_HISTORY
    tables: MutableMapping[str, TableState] = cast(MutableMapping[str, TableState], None)
    messages_active: int = 0
    messages_received_total: int = 0
    messages_received_by_topic: Counter[str] = cast(Counter[str], None)
    messages_s: int = 0
    messages_sent: int = 0
    messages_sent_by_topic: Counter[str] = cast(Counter[str], None)
    events_active: int = 0
    events_total: int = 0
    events_s: int = 0
    events_by_stream: Counter[str] = cast(Counter[str], None)
    events_by_task: Counter[str] = cast(Counter[str], None)
    events_runtime_avg: float = 0.0
    events_runtime: Deque[float] = cast(Deque[float], None)
    commit_latency: Deque[float] = cast(Deque[float], None)
    send_latency: Deque[float] = cast(Deque[float], None)
    assignment_latency: Deque[float] = cast(Deque[float], None)
    topic_buffer_full: Counter[TP] = cast(Counter[TP], None)
    metric_counts: Counter[str] = cast(Counter[str], None)
    tp_committed_offsets: TPOffsetMapping = cast(TPOffsetMapping, None)
    tp_read_offsets: TPOffsetMapping = cast(TPOffsetMapping, None)
    tp_end_offsets: TPOffsetMapping = cast(TPOffsetMapping, None)
    send_errors: int = 0
    assignments_completed: int = 0
    assignments_failed: int = 0
    rebalances: int = 0
    rebalance_return_latency: Deque[float] = cast(Deque[float], None)
    rebalance_end_latency: Deque[float] = cast(Deque[float], None)
    rebalance_return_avg: float = 0.0
    rebalance_end_avg: float = 0.0
    http_response_codes: Counter[HTTPStatus] = cast(Counter[HTTPStatus], None)
    http_response_latency: Deque[float] = cast(Deque[float], None)
    http_response_latency_avg: float = 0.0
    stream_inbound_time: Dict[TP, float] = cast(Dict[TP, float], None)

    def __init__(self, *, max_avg_history: Optional[int] = None, max_commit_latency_history: Optional[int] = None, max_send_latency_history: Optional[int] = None, max_assignment_latency_history: Optional[int] = None, messages_sent: int = 0, tables: Optional[MutableMapping[str, TableState]] = None, messages_active: int = 0, events_active: int = 0, messages_received_total: int = 0, messages_received_by_topic: Optional[Counter[str]] = None, events_total: int = 0, events_by_stream: Optional[Counter[str]] = None, events_by_task: Optional[Counter[str]] = None, events_runtime: Optional[Deque[float]] = None, commit_latency: Optional[Deque[float]] = None, send_latency: Optional[Deque[float]] = None, assignment_latency: Optional[Deque[float]] = None, events_s: int = 0, messages_s: int = 0, events_runtime_avg: float = 0.0, topic_buffer_full: Optional[Counter[TP]] = None, rebalances: Optional[int] = None, rebalance_return_latency: Optional[Deque[float]] = None, rebalance_end_latency: Optional[Deque[float]] = None, rebalance_return_avg: float = 0.0, rebalance_end_avg: float = 0.0, time: Callable[[], float] = monotonic, http_response_codes: Optional[Counter[HTTPStatus]] = None, http_response_latency: Optional[Deque[float]] = None, http_response_latency_avg: float = 0.0, **kwargs: Any) -> None:
        ...

    def secs_since(self, start_time: float) -> float:
        ...

    def ms_since(self, start_time: float) -> float:
        ...

    def secs_to_ms(self, timestamp: float) -> float:
        ...

    async def _sampler(self) -> None:
        ...

    def _sample(self, prev_event_total: int, prev_message_total: int) -> Tuple[int, int]:
        ...

    def asdict(self) -> Dict[str, Any]:
        ...

    def _events_by_stream_dict(self) -> Dict[str, int]:
        ...

    def _events_by_task_dict(self) -> Dict[str, int]:
        ...

    def _topic_buffer_full_dict(self) -> Dict[str, int]:
        ...

    def _metric_counts_dict(self) -> Dict[str, int]:
        ...

    def _http_response_codes_dict(self) -> Dict[int, int]:
        ...

    def _tp_committed_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        ...

    def _tp_read_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        ...

    def _tp_end_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        ...

    @classmethod
    def _tp_offsets_as_dict(cls, tp_offsets: TPOffsetMapping) -> Dict[str, Dict[int, int]]:
        ...

    def on_message_in(self, tp: TP, offset: int, message: Any) -> None:
        ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Dict[str, float]:
        ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, float]] = None) -> None:
        ...

    def on_topic_buffer_full(self, tp: TP) -> None:
        ...

    def on_message_out(self, tp: TP, offset: int, message: Any) -> None:
        ...

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        ...

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        ...

    def on_commit_initiated(self, consumer: ConsumerT) -> float:
        ...

    def on_commit_completed(self, consumer: ConsumerT, state: float) -> None:
        ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> float:
        ...

    def on_send_completed(self, producer: ProducerT, state: float, metadata: RecordMetadata) -> None:
        ...

    def on_send_error(self, producer: ProducerT, exc: Exception, state: float) -> None:
        ...

    def count(self, metric_name: str, count: int = 1) -> None:
        ...

    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None:
        ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
        ...

    def on_assignment_start(self, assignor: PartitionAssignorT) -> Dict[str, float]:
        ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Dict[str, float], exc: Exception) -> None:
        ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Dict[str, float]) -> None:
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[str, float]:
        ...

    def on_rebalance_return(self, app: AppT, state: Dict[str, float]) -> None:
        ...

    def on_rebalance_end(self, app: AppT, state: Dict[str, float]) -> None:
        ...

    def on_web_request_start(self, app: AppT, request: web.Request, *, view: Optional[Callable]) -> Dict[str, float]:
        ...

    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[str, float], *, view: Optional[Callable]) -> None:
        ...

    def _normalize(self, name: str, *, pattern: Pattern = RE_NORMALIZE, substitution: str = RE_NORMALIZE_SUBSTITUTION) -> str:
        ...
