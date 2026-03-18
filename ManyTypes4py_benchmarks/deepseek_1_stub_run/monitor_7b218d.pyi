```python
"""Monitor - sensor tracking metrics."""

import asyncio
from collections import deque
from http import HTTPStatus
from time import monotonic
from typing import Any, Callable, Counter, Deque, Dict, Mapping, MutableMapping, Optional, Pattern, Tuple

from faust import web
from faust.types import AppT, CollectionT, EventT, StreamT
from faust.types.assignor import PartitionAssignorT
from faust.types.tuples import Message, PendingMessage, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT
from mode import Service

from .base import Sensor

__all__ = ["TableState", "Monitor"]

MAX_AVG_HISTORY: int = ...
MAX_COMMIT_LATENCY_HISTORY: int = ...
MAX_SEND_LATENCY_HISTORY: int = ...
MAX_ASSIGNMENT_LATENCY_HISTORY: int = ...

TPOffsetMapping = MutableMapping[TP, int]
PartitionOffsetMapping = MutableMapping[int, int]
TPOffsetDict = MutableMapping[str, PartitionOffsetMapping]
RE_NORMALIZE: Pattern[str] = ...
RE_NORMALIZE_SUBSTITUTION: str = ...


class TableState:
    """Represents the current state of a table."""

    table: CollectionT
    keys_retrieved: int
    keys_updated: int
    keys_deleted: int

    def __init__(
        self,
        table: CollectionT,
        *,
        keys_retrieved: int = 0,
        keys_updated: int = 0,
        keys_deleted: int = 0,
    ) -> None: ...
    def asdict(self) -> Dict[str, int]: ...
    def __reduce_keywords__(self) -> Dict[str, Any]: ...


class Monitor(Sensor):
    """Default Faust Sensor.

    This is the default sensor, recording statistics about
    events, etc.
    """

    max_avg_history: int
    max_commit_latency_history: int
    max_send_latency_history: int
    max_assignment_latency_history: int
    tables: MutableMapping[str, TableState]
    messages_active: int
    messages_received_total: int
    messages_received_by_topic: Counter[str]
    messages_s: int
    messages_sent: int
    messages_sent_by_topic: Counter[str]
    events_active: int
    events_total: int
    events_s: int
    events_by_stream: Counter[str]
    events_by_task: Counter[str]
    events_runtime_avg: float
    events_runtime: Deque[float]
    commit_latency: Deque[float]
    send_latency: Deque[float]
    assignment_latency: Deque[float]
    topic_buffer_full: Counter[TP]
    metric_counts: Counter[str]
    tp_committed_offsets: TPOffsetMapping
    tp_read_offsets: TPOffsetMapping
    tp_end_offsets: TPOffsetMapping
    send_errors: int
    assignments_completed: int
    assignments_failed: int
    rebalances: int
    rebalance_return_latency: Deque[float]
    rebalance_end_latency: Deque[float]
    rebalance_return_avg: float
    rebalance_end_avg: float
    http_response_codes: Counter[HTTPStatus]
    http_response_latency: Deque[float]
    http_response_latency_avg: float
    stream_inbound_time: Dict[TP, float]
    time: Callable[[], float]

    def __init__(
        self,
        *,
        max_avg_history: Optional[int] = None,
        max_commit_latency_history: Optional[int] = None,
        max_send_latency_history: Optional[int] = None,
        max_assignment_latency_history: Optional[int] = None,
        messages_sent: int = 0,
        tables: Optional[MutableMapping[str, TableState]] = None,
        messages_active: int = 0,
        events_active: int = 0,
        messages_received_total: int = 0,
        messages_received_by_topic: Optional[Counter[str]] = None,
        events_total: int = 0,
        events_by_stream: Optional[Counter[str]] = None,
        events_by_task: Optional[Counter[str]] = None,
        events_runtime: Optional[Deque[float]] = None,
        commit_latency: Optional[Deque[float]] = None,
        send_latency: Optional[Deque[float]] = None,
        assignment_latency: Optional[Deque[float]] = None,
        events_s: int = 0,
        messages_s: int = 0,
        events_runtime_avg: float = 0.0,
        topic_buffer_full: Optional[Counter[TP]] = None,
        rebalances: Optional[int] = None,
        rebalance_return_latency: Optional[Deque[float]] = None,
        rebalance_end_latency: Optional[Deque[float]] = None,
        rebalance_return_avg: float = 0.0,
        rebalance_end_avg: float = 0.0,
        time: Callable[[], float] = monotonic,
        http_response_codes: Optional[Counter[HTTPStatus]] = None,
        http_response_latency: Optional[Deque[float]] = None,
        http_response_latency_avg: float = 0.0,
        **kwargs: Any,
    ) -> None: ...
    def secs_since(self, start_time: float) -> float: ...
    def ms_since(self, start_time: float) -> float: ...
    def secs_to_ms(self, timestamp: float) -> float: ...
    async def _sampler(self) -> None: ...
    def _sample(self, prev_event_total: int, prev_message_total: int) -> Tuple[int, int]: ...
    def asdict(self) -> Dict[str, Any]: ...
    def _events_by_stream_dict(self) -> Dict[str, int]: ...
    def _events_by_task_dict(self) -> Dict[str, int]: ...
    def _topic_buffer_full_dict(self) -> Dict[str, int]: ...
    def _metric_counts_dict(self) -> Dict[str, int]: ...
    def _http_response_codes_dict(self) -> Dict[int, int]: ...
    def _tp_committed_offsets_dict(self) -> TPOffsetDict: ...
    def _tp_read_offsets_dict(self) -> TPOffsetDict: ...
    def _tp_end_offsets_dict(self) -> TPOffsetDict: ...
    @classmethod
    def _tp_offsets_as_dict(cls, tp_offsets: TPOffsetMapping) -> TPOffsetDict: ...
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Dict[str, Optional[float]]: ...
    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Dict[str, Optional[float]]] = None,
    ) -> None: ...
    def on_topic_buffer_full(self, tp: TP) -> None: ...
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_table_get(self, table: CollectionT, key: Any) -> None: ...
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None: ...
    def on_table_del(self, table: CollectionT, key: Any) -> None: ...
    def _table_or_create(self, table: CollectionT) -> TableState: ...
    def on_commit_initiated(self, consumer: ConsumerT) -> float: ...
    def on_commit_completed(self, consumer: ConsumerT, state: float) -> None: ...
    def on_send_initiated(
        self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int
    ) -> float: ...
    def on_send_completed(
        self, producer: ProducerT, state: float, metadata: RecordMetadata
    ) -> None: ...
    def on_send_error(self, producer: ProducerT, exc: BaseException, state: float) -> None: ...
    def count(self, metric_name: str, count: int = 1) -> None: ...
    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None: ...
    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...
    def on_assignment_start(self, assignor: PartitionAssignorT) -> Dict[str, float]: ...
    def on_assignment_error(
        self, assignor: PartitionAssignorT, state: Dict[str, float], exc: BaseException
    ) -> None: ...
    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Dict[str, float]
    ) -> None: ...
    def on_rebalance_start(self, app: AppT) -> Dict[str, float]: ...
    def on_rebalance_return(self, app: AppT, state: Dict[str, float]) -> None: ...
    def on_rebalance_end(self, app: AppT, state: Dict[str, float]) -> None: ...
    def on_web_request_start(
        self, app: AppT, request: web.Request, *, view: Optional[web.View] = None
    ) -> Dict[str, float]: ...
    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: Optional[web.Response],
        state: Dict[str, float],
        *,
        view: Optional[web.View] = None,
    ) -> None: ...
    def _normalize(
        self,
        name: str,
        *,
        pattern: Pattern[str] = RE_NORMALIZE,
        substitution: str = RE_NORMALIZE_SUBSTITUTION,
    ) -> str: ...
```