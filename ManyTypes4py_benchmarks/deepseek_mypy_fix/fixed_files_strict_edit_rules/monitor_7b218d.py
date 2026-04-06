"""Monitor - sensor tracking metrics."""
import asyncio
import re
from collections import deque
from http import HTTPStatus
from statistics import median
from time import monotonic
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Pattern, Tuple, Union, cast
from mode import Service, label
from mode.utils.objects import KeywordReduce
from mode.utils.typing import Counter, Deque
from faust import web
from faust.types import AppT, CollectionT, EventT, StreamT
from faust.types.assignor import PartitionAssignorT
from faust.types.tuples import Message, PendingMessage, RecordMetadata, TP
from faust.types.transports import ConsumerT, ProducerT
from faust.utils.functional import deque_pushpopmax
from .base import Sensor
__all__ = ['TableState', 'Monitor']
MAX_AVG_HISTORY = 100
MAX_COMMIT_LATENCY_HISTORY = 30
MAX_SEND_LATENCY_HISTORY = 30
MAX_ASSIGNMENT_LATENCY_HISTORY = 30
TPOffsetMapping = MutableMapping[TP, int]
PartitionOffsetMapping = MutableMapping[int, int]
TPOffsetDict = MutableMapping[str, PartitionOffsetMapping]
RE_NORMALIZE = re.compile('[\\<\\>:\\s]+')
RE_NORMALIZE_SUBSTITUTION = '_'

class TableState(KeywordReduce):
    """Represents the current state of a table."""
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
        """Return table state as dictionary."""
        return {'keys_retrieved': self.keys_retrieved, 'keys_updated': self.keys_updated, 'keys_deleted': self.keys_deleted}

    def __reduce_keywords__(self) -> Dict[str, Any]:
        return {**self.asdict(), 'table': self.table}

class Monitor(Sensor, KeywordReduce):
    """Default Faust Sensor."""
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
    time: Callable[[], float] = monotonic

    def __init__(self, *, max_avg_history: Optional[int] = None, max_commit_latency_history: Optional[int] = None, max_send_latency_history: Optional[int] = None, max_assignment_latency_history: Optional[int] = None, messages_sent: int = 0, tables: Optional[MutableMapping[str, TableState]] = None, messages_active: int = 0, events_active: int = 0, messages_received_total: int = 0, messages_received_by_topic: Optional[Counter[str]] = None, events_total: int = 0, events_by_stream: Optional[Counter[str]] = None, events_by_task: Optional[Counter[str]] = None, events_runtime: Optional[Deque[float]] = None, commit_latency: Optional[Deque[float]] = None, send_latency: Optional[Deque[float]] = None, assignment_latency: Optional[Deque[float]] = None, events_s: int = 0, messages_s: int = 0, events_runtime_avg: float = 0.0, topic_buffer_full: Optional[Counter[TP]] = None, rebalances: Optional[int] = None, rebalance_return_latency: Optional[Deque[float]] = None, rebalance_end_latency: Optional[Deque[float]] = None, rebalance_return_avg: float = 0.0, rebalance_end_avg: float = 0.0, time: Callable[[], float] = monotonic, http_response_codes: Optional[Counter[HTTPStatus]] = None, http_response_latency: Optional[Deque[float]] = None, http_response_latency_avg: float = 0.0, **kwargs: Any) -> None:
        if max_avg_history is not None:
            self.max_avg_history = max_avg_history
        if max_commit_latency_history is not None:
            self.max_commit_latency_history = max_commit_latency_history
        if max_send_latency_history is not None:
            self.max_send_latency_history = max_send_latency_history
        if max_assignment_latency_history is not None:
            self.max_assignment_latency_history = max_assignment_latency_history
        if rebalances is not None:
            self.rebalances = rebalances
        self.tables = {} if tables is None else tables
        self.commit_latency = deque() if commit_latency is None else commit_latency
        self.send_latency = deque() if send_latency is None else send_latency
        self.assignment_latency = deque() if assignment_latency is None else assignment_latency
        self.rebalance_return_latency = deque() if rebalance_return_latency is None else rebalance_return_latency
        self.rebalance_end_latency = deque() if rebalance_end_latency is None else rebalance_end_latency
        self.rebalance_return_avg = rebalance_return_avg
        self.rebalance_end_avg = rebalance_end_avg
        self.messages_active = messages_active
        self.messages_received_total = messages_received_total
        self.messages_received_by_topic = Counter()
        self.messages_sent = messages_sent
        self.messages_sent_by_topic = Counter()
        self.messages_s = messages_s
        self.events_active = events_active
        self.events_total = events_total
        self.events_by_task = Counter()
        self.events_by_stream = Counter()
        self.events_s = events_s
        self.events_runtime_avg = events_runtime_avg
        self.events_runtime = deque() if events_runtime is None else events_runtime
        self.topic_buffer_full = Counter()
        self.time = time
        self.http_response_codes = Counter()
        self.http_response_latency = deque()
        self.http_response_latency_avg = http_response_latency_avg
        self.metric_counts = Counter()
        self.tp_committed_offsets = {}
        self.tp_read_offsets = {}
        self.tp_end_offsets = {}
        self.stream_inbound_time = {}
        Service.__init__(self, **kwargs)

    def secs_since(self, start_time: float) -> float:
        """Given timestamp start, return number of seconds since that time."""
        return self.time() - start_time

    def ms_since(self, start_time: float) -> float:
        """Given timestamp start, return number of ms since that time."""
        return self.secs_to_ms(self.secs_since(start_time))

    def secs_to_ms(self, timestamp: float) -> float:
        """Convert seconds to milliseconds."""
        return timestamp * 1000.0

    @Service.task
    async def _sampler(self) -> None:
        prev_message_total = self.messages_received_total
        prev_event_total = self.events_total
        async for sleep_time in self.itertimer(1.0, name='Monitor.sampler'):
            prev_event_total, prev_message_total = self._sample(prev_event_total, prev_message_total)

    def _sample(self, prev_event_total: int, prev_message_total: int) -> Tuple[int, int]:
        if self.events_runtime:
            self.events_runtime_avg = median(self.events_runtime)
        self.events_s, prev_event_total = (self.events_total - prev_event_total, self.events_total)
        self.messages_s, prev_message_total = (self.messages_received_total - prev_message_total, self.messages_received_total)
        if self.rebalance_return_latency:
            self.rebalance_return_avg = median(self.rebalance_return_latency)
        if self.rebalance_end_latency:
            self.rebalance_end_avg = median(self.rebalance_end_latency)
        if self.http_response_latency:
            self.http_response_latency_avg = median(self.http_response_latency)
        return (prev_event_total, prev_message_total)

    def asdict(self) -> Dict[str, Any]:
        """Return monitor state as dictionary."""
        return {
            'messages_active': self.messages_active,
            'messages_received_total': self.messages_received_total,
            'messages_sent': self.messages_sent,
            'messages_sent_by_topic': self.messages_sent_by_topic,
            'messages_s': self.messages_s,
            'messages_received_by_topic': self.messages_received_by_topic,
            'events_active': self.events_active,
            'events_total': self.events_total,
            'events_s': self.events_s,
            'events_runtime_avg': self.events_runtime_avg,
            'events_by_task': self._events_by_task_dict(),
            'events_by_stream': self._events_by_stream_dict(),
            'commit_latency': self.commit_latency,
            'send_latency': self.send_latency,
            'send_errors': self.send_errors,
            'assignment_latency': self.assignment_latency,
            'assignments_completed': self.assignments_completed,
            'assignments_failed': self.assignments_failed,
            'topic_buffer_full': self._topic_buffer_full_dict(),
            'tables': {name: table.asdict() for name, table in self.tables.items()},
            'metric_counts': self._metric_counts_dict(),
            'topic_committed_offsets': self._tp_committed_offsets_dict(),
            'topic_read_offsets': self._tp_read_offsets_dict(),
            'topic_end_offsets': self._tp_end_offsets_dict(),
            'rebalances': self.rebalances,
            'rebalance_return_latency': self.rebalance_return_latency,
            'rebalance_end_latency': self.rebalance_end_latency,
            'rebalance_return_avg': self.rebalance_return_avg,
            'rebalance_end_avg': self.rebalance_end_avg,
            'http_response_codes': self._http_response_codes_dict(),
            'http_response_latency': self.http_response_latency,
            'http_response_latency_avg': self.http_response_latency_avg
        }

    def _events_by_stream_dict(self) -> Dict[str, int]:
        return {label(stream): count for stream, count in self.events_by_stream.items()}

    def _events_by_task_dict(self) -> Dict[str, int]:
        return {label(task): count for task, count in self.events_by_task.items()}

    def _topic_buffer_full_dict(self) -> Dict[str, int]:
        return {label(topic): count for topic, count in self.topic_buffer_full.items()}

    def _metric_counts_dict(self) -> Dict[str, int]:
        return {key: count for key, count in self.metric_counts.items()}

    def _http_response_codes_dict(self) -> Dict[int, int]:
        return {int(code): count for code, count in self.http_response_codes.items()}

    def _tp_committed_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        return self._tp_offsets_as_dict(self.tp_committed_offsets)

    def _tp_read_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        return self._tp_offsets_as_dict(self.tp_read_offsets)

    def _tp_end_offsets_dict(self) -> Dict[str, Dict[int, int]]:
        return self._tp_offsets_as_dict(self.tp_end_offsets)

    @classmethod
    def _tp_offsets_as_dict(cls, tp_offsets: TPOffsetMapping) -> Dict[str, Dict[int, int]]:
        topic_partition_offsets: Dict[str, Dict[int, int]] = {}
        for tp, offset in tp_offsets.items():
            partition_offsets = topic_partition_offsets.get(tp.topic) or {}
            partition_offsets[tp.partition] = offset
            topic_partition_offsets[tp.topic] = partition_offsets
        return topic_partition_offsets

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        """Call before message is delegated to streams."""
        self.messages_received_total += 1
        self.messages_active += 1
        self.messages_received_by_topic[tp.topic] += 1
        self.tp_read_offsets[tp] = offset
        message.time_in = self.time()

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Dict[str, Optional[float]]:
        """Call when stream starts processing an event."""
        self.events_total += 1
        self.stream_inbound_time[tp] = monotonic()
        self.events_by_stream[str(stream)] += 1
        self.events_by_task[str(stream.task_owner)] += 1
        self.events_active += 1
        return {'time_in': self.time(), 'time_out': None, 'time_total': None}

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, Optional[float]]] = None) -> None:
        """Call when stream is done processing an event."""
        if state is not None:
            time_out = self.time()
            time_in = state['time_in']
            time_total = time_out - time_in
            self.events_active -= 1
            state.update(time_out=time_out, time_total=time_total)
            deque_pushpopmax(self.events_runtime, time_total, self.max_avg_history)

    def on_topic_buffer_full(self, tp: TP) -> None:
        """Call when conductor topic buffer is full and has to wait."""
        self.topic_buffer_full[tp] += 1

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        """Call when message is fully acknowledged and can be committed."""
        self.messages_active -= 1
        time_out = message.time_out = self.time()
        time_in = message.time_in
        if time_in is not None:
            message.time_total = time_out - time_in

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        """Call when value in table is retrieved."""
        self._table_or_create(table).keys_retrieved += 1

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        """Call when new value for key in table is set."""
        self._table_or_create(table).keys_updated += 1

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        """Call when key in a table is deleted."""
        self._table_or_create(table).keys_deleted += 1

    def _table_or_create(self, table: CollectionT) -> TableState:
        try:
            return self.tables[table.name]
        except KeyError:
            state = self.tables