"""Monitor - sensor tracking metrics."""
import asyncio
import re
from collections import deque
from http import HTTPStatus
from statistics import median
from time import monotonic
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Pattern, Tuple, cast
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

    def __init__(self, table, *, keys_retrieved: int=0, keys_updated: int=0,
        keys_deleted: int=0):
        self.table: CollectionT = table
        self.keys_retrieved = keys_retrieved
        self.keys_updated = keys_updated
        self.keys_deleted = keys_deleted

    def asdict(self):
        """Return table state as dictionary."""
        return {'keys_retrieved': self.keys_retrieved, 'keys_updated': self
            .keys_updated, 'keys_deleted': self.keys_deleted}

    def __reduce_keywords__(self):
        return {**self.asdict(), 'table': self.table}


class Monitor(Sensor, KeywordReduce):
    """Default Faust Sensor.

    This is the default sensor, recording statistics about
    events, etc.
    """
    max_avg_history: int = MAX_AVG_HISTORY
    max_commit_latency_history: int = MAX_COMMIT_LATENCY_HISTORY
    max_send_latency_history: int = MAX_SEND_LATENCY_HISTORY
    max_assignment_latency_history: int = MAX_ASSIGNMENT_LATENCY_HISTORY
    tables: MutableMapping[str, TableState] = cast(MutableMapping[str,
        TableState], None)
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
    http_response_codes: Counter[int] = cast(Counter[int], None)
    http_response_latency: Deque[float] = cast(Deque[float], None)
    http_response_latency_avg: float = 0.0
    stream_inbound_time: Dict[str, float] = cast(Dict[str, float], None)

    def __init__(self, *, max_avg_history: Optional[int]=None,
        max_commit_latency_history: Optional[int]=None,
        max_send_latency_history: Optional[int]=None,
        max_assignment_latency_history: Optional[int]=None, messages_sent:
        int=0, tables: Optional[MutableMapping[str, TableState]]=None,
        messages_active: int=0, events_active: int=0,
        messages_received_total: int=0, messages_received_by_topic:
        Optional[Counter[str]]=None, events_total: int=0, events_by_stream:
        Optional[Counter[str]]=None, events_by_task: Optional[Counter[str]]
        =None, events_runtime: Optional[Deque[float]]=None, commit_latency:
        Optional[Deque[float]]=None, send_latency: Optional[Deque[float]]=
        None, assignment_latency: Optional[Deque[float]]=None, events_s:
        int=0, messages_s: int=0, events_runtime_avg: float=0.0,
        topic_buffer_full: Optional[Counter[TP]]=None, rebalances: Optional
        [int]=None, rebalance_return_latency: Optional[Deque[float]]=None,
        rebalance_end_latency: Optional[Deque[float]]=None,
        rebalance_return_avg: float=0.0, rebalance_end_avg: float=0.0, time:
        Callable[[], float]=monotonic, http_response_codes: Optional[
        Counter[int]]=None, http_response_latency: Optional[Deque[float]]=
        None, http_response_latency_avg: float=0.0, **kwargs: Any):
        if max_avg_history is not None:
            self.max_avg_history = max_avg_history
        if max_commit_latency_history is not None:
            self.max_commit_latency_history = max_commit_latency_history
        if max_send_latency_history is not None:
            self.max_send_latency_history = max_send_latency_history
        if max_assignment_latency_history is not None:
            self.max_assignment_latency_history = (
                max_assignment_latency_history)
        if rebalances is not None:
            self.rebalances = rebalances
        self.tables: MutableMapping[str, TableState] = {
            } if tables is None else tables
        self.commit_latency: Deque[float] = deque(
            ) if commit_latency is None else commit_latency
        self.send_latency: Deque[float] = deque(
            ) if send_latency is None else send_latency
        self.assignment_latency: Deque[float] = deque(
            ) if assignment_latency is None else assignment_latency
        self.rebalance_return_latency: Deque[float] = deque(
            ) if rebalance_return_latency is None else rebalance_return_latency
        self.rebalance_end_latency: Deque[float] = deque(
            ) if rebalance_end_latency is None else rebalance_end_latency
        self.rebalance_return_avg: float = rebalance_return_avg
        self.rebalance_end_avg: float = rebalance_end_avg
        self.messages_active: int = messages_active
        self.messages_received_total: int = messages_received_total
        self.messages_received_by_topic: Counter[str] = Counter(
            ) if messages_received_by_topic is None else messages_received_by_topic
        self.messages_sent: int = messages_sent
        self.messages_sent_by_topic: Counter[str] = Counter(
            ) if messages_sent_by_topic is None else messages_sent_by_topic
        self.messages_s: int = messages_s
        self.events_active: int = events_active
        self.events_total: int = events_total
        self.events_by_task: Counter[str] = Counter(
            ) if events_by_task is None else events_by_task
        self.events_by_stream: Counter[str] = Counter(
            ) if events_by_stream is None else events_by_stream
        self.events_s: int = events_s
        self.events_runtime_avg: float = events_runtime_avg
        self.events_runtime: Deque[float] = deque(
            ) if events_runtime is None else events_runtime
        self.topic_buffer_full: Counter[TP] = Counter(
            ) if topic_buffer_full is None else topic_buffer_full
        self.time: Callable[[], float] = time
        self.http_response_codes: Counter[int] = Counter(
            ) if http_response_codes is None else http_response_codes
        self.http_response_latency: Deque[float] = deque(
            ) if http_response_latency is None else http_response_latency
        self.http_response_latency_avg: float = http_response_latency_avg
        self.metric_counts: Counter[str] = Counter()
        self.tp_committed_offsets: TPOffsetMapping = {}
        self.tp_read_offsets: TPOffsetMapping = {}
        self.tp_end_offsets: TPOffsetMapping = {}
        self.stream_inbound_time: Dict[str, float] = {}
        self.send_errors: int = 0
        self.assignments_completed: int = 0
        self.assignments_failed: int = 0
        self.rebalances: int = self.rebalances or 0
        Service.__init__(self, **kwargs)

    def secs_since(self, start_time):
        """Given timestamp start, return number of seconds since that time."""
        return self.time() - start_time

    def ms_since(self, start_time):
        """Given timestamp start, return number of ms since that time."""
        return self.secs_to_ms(self.secs_since(start_time))

    def secs_to_ms(self, timestamp):
        """Convert seconds to milliseconds."""
        return timestamp * 1000.0

    @Service.task
    async def _sampler(self) ->None:
        prev_message_total: int = self.messages_received_total
        prev_event_total: int = self.events_total
        async for sleep_time in self.itertimer(1.0, name='Monitor.sampler'):
            prev_event_total, prev_message_total = self._sample(
                prev_event_total, prev_message_total)

    def _sample(self, prev_event_total, prev_message_total):
        if self.events_runtime:
            self.events_runtime_avg = median(self.events_runtime)
        self.events_s, prev_event_total = (self.events_total -
            prev_event_total, self.events_total)
        self.messages_s, prev_message_total = (self.messages_received_total -
            prev_message_total, self.messages_received_total)
        if self.rebalance_return_latency:
            self.rebalance_return_avg = median(self.rebalance_return_latency)
        if self.rebalance_end_latency:
            self.rebalance_end_avg = median(self.rebalance_end_latency)
        if self.http_response_latency:
            self.http_response_latency_avg = median(self.http_response_latency)
        return prev_event_total, prev_message_total

    def asdict(self):
        """Return monitor state as dictionary."""
        return {'messages_active': self.messages_active,
            'messages_received_total': self.messages_received_total,
            'messages_sent': self.messages_sent, 'messages_sent_by_topic':
            self.messages_sent_by_topic, 'messages_s': self.messages_s,
            'messages_received_by_topic': self.messages_received_by_topic,
            'events_active': self.events_active, 'events_total': self.
            events_total, 'events_s': self.events_s, 'events_runtime_avg':
            self.events_runtime_avg, 'events_by_task': self.
            _events_by_task_dict(), 'events_by_stream': self.
            _events_by_stream_dict(), 'commit_latency': list(self.
            commit_latency), 'send_latency': list(self.send_latency),
            'send_errors': self.send_errors, 'assignment_latency': list(
            self.assignment_latency), 'assignments_completed': self.
            assignments_completed, 'assignments_failed': self.
            assignments_failed, 'topic_buffer_full': self.
            _topic_buffer_full_dict(), 'tables': {name: table.asdict() for 
            name, table in self.tables.items()}, 'metric_counts': self.
            _metric_counts_dict(), 'topic_committed_offsets': self.
            _tp_committed_offsets_dict(), 'topic_read_offsets': self.
            _tp_read_offsets_dict(), 'topic_end_offsets': self.
            _tp_end_offsets_dict(), 'rebalances': self.rebalances,
            'rebalance_return_latency': list(self.rebalance_return_latency),
            'rebalance_end_latency': list(self.rebalance_end_latency),
            'rebalance_return_avg': self.rebalance_return_avg,
            'rebalance_end_avg': self.rebalance_end_avg,
            'http_response_codes': self._http_response_codes_dict(),
            'http_response_latency': list(self.http_response_latency),
            'http_response_latency_avg': self.http_response_latency_avg}

    def _events_by_stream_dict(self):
        return {label(stream): count for stream, count in self.
            events_by_stream.items()}

    def _events_by_task_dict(self):
        return {label(task): count for task, count in self.events_by_task.
            items()}

    def _topic_buffer_full_dict(self):
        return {label(topic): count for topic, count in self.
            topic_buffer_full.items()}

    def _metric_counts_dict(self):
        return {key: count for key, count in self.metric_counts.items()}

    def _http_response_codes_dict(self):
        return {int(code): count for code, count in self.
            http_response_codes.items()}

    def _tp_committed_offsets_dict(self):
        return self._tp_offsets_as_dict(self.tp_committed_offsets)

    def _tp_read_offsets_dict(self):
        return self._tp_offsets_as_dict(self.tp_read_offsets)

    def _tp_end_offsets_dict(self):
        return self._tp_offsets_as_dict(self.tp_end_offsets)

    @classmethod
    def _tp_offsets_as_dict(cls, tp_offsets):
        topic_partition_offsets: TPOffsetDict = {}
        for tp, offset in tp_offsets.items():
            partition_offsets = topic_partition_offsets.get(tp.topic, {})
            partition_offsets[tp.partition] = offset
            topic_partition_offsets[tp.topic] = partition_offsets
        return topic_partition_offsets

    def on_message_in(self, tp, offset, message):
        """Call before message is delegated to streams."""
        self.messages_received_total += 1
        self.messages_active += 1
        self.messages_received_by_topic[tp.topic] += 1
        self.tp_read_offsets[tp] = offset
        message.time_in = self.time()

    def on_stream_event_in(self, tp, offset, stream, event):
        """Call when stream starts processing an event."""
        self.events_total += 1
        self.stream_inbound_time[tp.topic_partition()] = self.time()
        self.events_by_stream[label(stream)] += 1
        self.events_by_task[label(stream.task_owner)] += 1
        self.events_active += 1
        return {'time_in': self.time(), 'time_out': None, 'time_total': None}

    def on_stream_event_out(self, tp, offset, stream, event, state=None):
        """Call when stream is done processing an event."""
        if state is not None:
            time_out: float = self.time()
            time_in: float = state['time_in'] or 0.0
            time_total: float = time_out - time_in
            self.events_active -= 1
            state.update(time_out=time_out, time_total=time_total)
            deque_pushpopmax(self.events_runtime, time_total, self.
                max_avg_history)

    def on_topic_buffer_full(self, tp):
        """Call when conductor topic buffer is full and has to wait."""
        self.topic_buffer_full[tp] += 1

    def on_message_out(self, tp, offset, message):
        """Call when message is fully acknowledged and can be committed."""
        self.messages_active -= 1
        time_out: float = self.time()
        message.time_out = time_out
        time_in: float = message.time_in or 0.0
        if time_in is not None:
            message.time_total = time_out - time_in

    def on_table_get(self, table, key):
        """Call when value in table is retrieved."""
        self._table_or_create(table).keys_retrieved += 1

    def on_table_set(self, table, key, value):
        """Call when new value for key in table is set."""
        self._table_or_create(table).keys_updated += 1

    def on_table_del(self, table, key):
        """Call when key in a table is deleted."""
        self._table_or_create(table).keys_deleted += 1

    def _table_or_create(self, table):
        try:
            return self.tables[table.name]
        except KeyError:
            state = self.tables[table.name] = TableState(table)
            return state

    def on_commit_initiated(self, consumer):
        """Consumer is about to commit topic offset."""
        return self.time()

    def on_commit_completed(self, consumer, state):
        """Call when consumer commit offset operation completed."""
        latency: float = self.time() - state
        deque_pushpopmax(self.commit_latency, latency, self.
            max_commit_latency_history)

    def on_send_initiated(self, producer, topic, message, keysize, valsize):
        """Call when message added to producer buffer."""
        self.messages_sent += 1
        self.messages_sent_by_topic[topic] += 1
        return self.time()

    def on_send_completed(self, producer, state, metadata):
        """Call when producer finished sending message."""
        latency: float = self.time() - state
        deque_pushpopmax(self.send_latency, latency, self.
            max_send_latency_history)

    def on_send_error(self, producer, exc, state):
        """Call when producer was unable to publish message."""
        self.send_errors += 1

    def count(self, metric_name, count=1):
        """Count metric by name."""
        self.metric_counts[metric_name] += count

    def on_tp_commit(self, tp_offsets):
        """Call when offset in topic partition is committed."""
        self.tp_committed_offsets.update(tp_offsets)

    def track_tp_end_offset(self, tp, offset):
        """Track new topic partition end offset for monitoring lags."""
        self.tp_end_offsets[tp] = offset

    def on_assignment_start(self, assignor):
        """Partition assignor is starting to assign partitions."""
        return {'time_start': self.time()}

    def on_assignment_error(self, assignor, state, exc):
        """Partition assignor did not complete assignor due to error."""
        time_total: float = self.time() - state['time_start']
        deque_pushpopmax(self.assignment_latency, time_total, self.
            max_assignment_latency_history)
        self.assignments_failed += 1

    def on_assignment_completed(self, assignor, state):
        """Partition assignor completed assignment."""
        time_total: float = self.time() - state['time_start']
        deque_pushpopmax(self.assignment_latency, time_total, self.
            max_assignment_latency_history)
        self.assignments_completed += 1

    def on_rebalance_start(self, app):
        """Cluster rebalance in progress."""
        self.rebalances = app.rebalancing_count
        return {'time_start': self.time()}

    def on_rebalance_return(self, app, state):
        """Consumer replied assignment is done to broker."""
        time_start: float = state['time_start']
        time_return: float = self.time()
        latency_return: float = time_return - time_start
        state.update(time_return=time_return, latency_return=latency_return)
        deque_pushpopmax(self.rebalance_return_latency, latency_return,
            self.max_avg_history)

    def on_rebalance_end(self, app, state):
        """Cluster rebalance fully completed (including recovery)."""
        time_start: float = state['time_start']
        time_end: float = self.time()
        latency_end: float = time_end - time_start
        state.update(time_end=time_end, latency_end=latency_end)
        deque_pushpopmax(self.rebalance_end_latency, latency_end, self.
            max_avg_history)

    def on_web_request_start(self, app, request, *, view: Optional[web.View
        ]=None):
        """Web server started working on request."""
        return {'time_start': self.time()}

    def on_web_request_end(self, app, request, response, state, *, view:
        Optional[web.View]=None):
        """Web server finished working on request."""
        status_code: int = response.status if response is not None else 500
        time_start: float = state['time_start']
        time_end: float = self.time()
        latency_end: float = time_end - time_start
        state.update(time_end=time_end, latency_end=latency_end,
            status_code=status_code)
        deque_pushpopmax(self.http_response_latency, latency_end, self.
            max_avg_history)
        self.http_response_codes[status_code] += 1

    def _normalize(self, name, *, pattern: Pattern=RE_NORMALIZE,
        substitution: str=RE_NORMALIZE_SUBSTITUTION):
        return pattern.sub(substitution, name)
