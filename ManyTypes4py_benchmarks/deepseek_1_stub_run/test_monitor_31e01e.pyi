```python
from collections.abc import MutableMapping
from http import HTTPStatus
from typing import Any, ClassVar, Deque, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import Protocol

class Message(Protocol):
    time_in: Optional[float]
    time_out: Optional[float]
    time_total: Optional[float]

class Stream(Protocol):
    task_owner: Any

class Topic(Protocol): ...

class Event(Protocol): ...

class Table(Protocol): ...

class Consumer(Protocol): ...

class Producer(Protocol): ...

class TP:
    topic: str
    partition: int
    def __init__(self, topic: str, partition: int) -> None: ...

class TableState:
    table: Table
    keys_retrieved: int
    keys_updated: int
    keys_deleted: int
    def asdict(self) -> Dict[str, int]: ...
    def __reduce_keywords__(self) -> Dict[str, Any]: ...

class Monitor:
    max_avg_history: ClassVar[int]
    max_commit_latency_history: ClassVar[int]
    max_send_latency_history: ClassVar[int]
    max_assignment_latency_history: ClassVar[int]
    
    messages_active: int
    messages_received_total: int
    messages_sent: int
    messages_sent_by_topic: Dict[str, int]
    messages_s: float
    messages_received_by_topic: Dict[str, int]
    events_active: int
    events_total: int
    events_s: float
    events_runtime_avg: float
    events_by_task: Dict[str, int]
    events_by_stream: Dict[str, int]
    commit_latency: Deque[float]
    send_latency: Deque[float]
    assignment_latency: Deque[float]
    assignments_completed: int
    assignments_failed: int
    send_errors: int
    topic_buffer_full: Dict[TP, int]
    tables: Dict[str, TableState]
    tp_read_offsets: Dict[TP, int]
    tp_committed_offsets: Dict[TP, int]
    tp_end_offsets: Dict[TP, int]
    rebalance_end_avg: float
    rebalance_end_latency: Deque[float]
    rebalance_return_avg: float
    rebalance_return_latency: Deque[float]
    rebalances: int
    http_response_codes: Dict[HTTPStatus, int]
    http_response_latency: Deque[float]
    http_response_latency_avg: float
    events_runtime: Deque[float]
    
    def __init__(
        self,
        *,
        max_avg_history: Optional[int] = ...,
        max_commit_latency_history: Optional[int] = ...,
        max_send_latency_history: Optional[int] = ...,
        max_assignment_latency_history: Optional[int] = ...,
        messages_active: int = ...,
        messages_received_total: int = ...,
        messages_sent: int = ...,
        messages_s: float = ...,
        messages_received_by_topic: Optional[Dict[str, int]] = ...,
        events_active: int = ...,
        events_total: int = ...,
        events_s: float = ...,
        events_runtime_avg: float = ...,
        events_by_task: Optional[Dict[str, int]] = ...,
        events_by_stream: Optional[Dict[str, int]] = ...,
        commit_latency: Optional[List[float]] = ...,
        send_latency: Optional[List[float]] = ...,
        topic_buffer_full: Optional[Dict[TP, int]] = ...,
        rebalances: int = ...,
        **kwargs: Any
    ) -> None: ...
    
    def asdict(self) -> Dict[str, Any]: ...
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_stream_event_in(
        self, tp: TP, offset: int, stream: Stream, event: Event
    ) -> Dict[str, Optional[float]]: ...
    def on_stream_event_out(
        self, tp: TP, offset: int, stream: Stream, event: Event, state: Optional[Dict[str, Optional[float]]]
    ) -> None: ...
    def on_topic_buffer_full(self, tp: TP) -> None: ...
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_table_get(self, table: Table, key: Any) -> None: ...
    def on_table_set(self, table: Table, key: Any, value: Any) -> None: ...
    def on_table_del(self, table: Table, key: Any) -> None: ...
    def on_commit_initiated(self, consumer: Consumer) -> float: ...
    def on_commit_completed(self, consumer: Consumer, state: float) -> None: ...
    def on_send_initiated(
        self, producer: Producer, topic: str, message: Any, keysize: int, valsize: int
    ) -> float: ...
    def on_send_completed(
        self, producer: Producer, state: float, metadata: Any
    ) -> None: ...
    def on_send_error(
        self, producer: Producer, state: Any, exc: BaseException
    ) -> None: ...
    def on_assignment_start(self, assignor: Any) -> Dict[str, float]: ...
    def on_assignment_completed(
        self, assignor: Any, state: Dict[str, float]
    ) -> None: ...
    def on_assignment_error(
        self, assignor: Any, state: Dict[str, float], exc: BaseException
    ) -> None: ...
    def on_rebalance_start(self, app: Any) -> Dict[str, float]: ...
    def on_rebalance_return(self, app: Any, state: Dict[str, float]) -> None: ...
    def on_rebalance_end(self, app: Any, state: Dict[str, float]) -> None: ...
    def on_web_request_start(
        self, app: Any, request: Any, *, view: Any
    ) -> Dict[str, float]: ...
    def on_web_request_end(
        self, app: Any, request: Any, response: Any, state: Dict[str, float], *, view: Any
    ) -> None: ...
    def _table_or_create(self, table: Table) -> TableState: ...
    def on_tp_commit(self, offsets: Dict[TP, int]) -> None: ...
    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...
    async def _sampler(self, parent: Any) -> None: ...
    def _sample(self, prev_event_total: int, prev_message_total: int) -> None: ...
    def _events_by_task_dict(self) -> Dict[str, int]: ...
    def _events_by_stream_dict(self) -> Dict[str, int]: ...
    def _topic_buffer_full_dict(self) -> Dict[str, int]: ...
    def _metric_counts_dict(self) -> Dict[str, int]: ...
    def _http_response_codes_dict(self) -> Dict[int, int]: ...

TP1: TP = ...
```