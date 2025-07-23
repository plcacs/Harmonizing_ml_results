"""Monitor using datadog."""
import re
from typing import Any, Dict, List, Optional, cast, TypeVar, Union
from mode.utils.objects import cached_property
from faust import web
from faust.exceptions import ImproperlyConfigured
from faust.sensors.monitor import Monitor, TPOffsetMapping
from faust.types import AppT, CollectionT, EventT, Message, PendingMessage, RecordMetadata, StreamT, TP
from faust.types.assignor import PartitionAssignorT
from faust.types.transports import ConsumerT, ProducerT
try:
    import datadog
    from datadog.dogstatsd import DogStatsd
except ImportError:
    datadog = None  # type: ignore

    class DogStatsD:  # type: ignore
        ...
__all__ = ['DatadogMonitor']

T = TypeVar('T')

class DatadogStatsClient:
    """Statsd compliant datadog client."""

    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Any) -> None:
        self.client: DogStatsd = DogStatsd(host=host, port=port, namespace=prefix, **kwargs)
        self.rate: float = rate
        self.sanitize_re: re.Pattern = re.compile('[^0-9a-zA-Z_]')
        self.re_substitution: str = '_'

    def gauge(self, metric: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        self.client.gauge(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def increment(self, metric: str, value: float = 1.0, labels: Optional[Dict[str, Any]] = None) -> None:
        self.client.increment(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def incr(self, metric: str, count: int = 1) -> None:
        """Statsd compatibility."""
        self.increment(metric, value=count)

    def decrement(self, metric: str, value: float = 1.0, labels: Optional[Dict[str, Any]] = None) -> None:
        self.client.decrement(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def decr(self, metric: str, count: float = 1.0) -> None:
        """Statsd compatibility."""
        self.decrement(metric, value=count)

    def timing(self, metric: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        self.client.timing(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def timed(self, metric: Optional[str] = None, labels: Optional[Dict[str, Any]] = None, use_ms: Optional[bool] = None) -> Any:
        return self.client.timed(metric=metric, tags=self._encode_labels(labels), sample_rate=self.rate, use_ms=use_ms)

    def histogram(self, metric: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        self.client.histogram(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def _encode_labels(self, labels: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        def sanitize(s: Any) -> str:
            return self.sanitize_re.sub(self.re_substitution, str(s))
        return [f'{sanitize(k)}:{sanitize(v)}' for k, v in labels.items()] if labels else None

class DatadogMonitor(Monitor):
    """Datadog Faust Sensor."""

    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Any) -> None:
        self.host: str = host
        self.port: int = port
        self.prefix: str = prefix
        self.rate: float = rate
        if datadog is None:
            raise ImproperlyConfigured(f'{type(self).__name__} requires `pip install datadog`.')
        super().__init__(**kwargs)

    def _new_datadog_stats_client(self) -> DatadogStatsClient:
        return DatadogStatsClient(host=self.host, port=self.port, prefix=self.prefix, rate=self.rate)

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        super().on_message_in(tp, offset, message)
        labels = self._format_label(tp)
        self.client.increment('messages_received', labels=labels)
        self.client.increment('messages_active', labels=labels)
        self.client.increment('topic_messages_received', labels={'topic': tp.topic})
        self.client.gauge('read_offset', offset, labels=labels)

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Dict[str, Any]:
        state = super().on_stream_event_in(tp, offset, stream, event)
        labels = self._format_label(tp, stream)
        self.client.increment('events', labels=labels)
        self.client.increment('events_active', labels=labels)
        return state

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, Any]] = None) -> None:
        super().on_stream_event_out(tp, offset, stream, event, state)
        labels = self._format_label(tp, stream)
        self.client.decrement('events_active', labels=labels)
        self.client.timing('events_runtime', self.secs_to_ms(self.events_runtime[-1]), labels=labels)

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        super().on_message_out(tp, offset, message)
        self.client.decrement('messages_active', labels=self._format_label(tp))

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        super().on_table_get(table, key)
        self.client.increment('table_keys_retrieved', labels=self._format_label(table=table))

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        super().on_table_set(table, key, value)
        self.client.increment('table_keys_updated', labels=self._format_label(table=table))

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        super().on_table_del(table, key)
        self.client.increment('table_keys_deleted', labels=self._format_label(table=table))

    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None:
        super().on_commit_completed(consumer, state)
        self.client.timing('commit_latency', self.ms_since(cast(float, state)))

    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> Any:
        self.client.increment('topic_messages_sent', labels={'topic': topic})
        return super().on_send_initiated(producer, topic, message, keysize, valsize)

    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None:
        super().on_send_completed(producer, state, metadata)
        self.client.increment('messages_sent')
        self.client.timing('send_latency', self.ms_since(cast(float, state)))

    def on_send_error(self, producer: ProducerT, exc: BaseException, state: Any) -> None:
        super().on_send_error(producer, exc, state)
        self.client.increment('messages_send_failed')
        self.client.timing('send_latency_for_error', self.ms_since(cast(float, state)))

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Dict[str, Any], exc: BaseException) -> None:
        super().on_assignment_error(assignor, state, exc)
        self.client.increment('assignments_error')
        self.client.timing('assignment_latency', self.ms_since(state['time_start']))

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Dict[str, Any]) -> None:
        super().on_assignment_completed(assignor, state)
        self.client.increment('assignments_complete')
        self.client.timing('assignment_latency', self.ms_since(state['time_start']))

    def on_rebalance_start(self, app: AppT) -> Dict[str, Any]:
        state = super().on_rebalance_start(app)
        self.client.increment('rebalances')
        return state

    def on_rebalance_return(self, app: AppT, state: Dict[str, Any]) -> None:
        super().on_rebalance_return(app, state)
        self.client.decrement('rebalances')
        self.client.increment('rebalances_recovering')
        self.client.timing('rebalance_return_latency', self.ms_since(state['time_return']))

    def on_rebalance_end(self, app: AppT, state: Dict[str, Any]) -> None:
        super().on_rebalance_end(app, state)
        self.client.decrement('rebalances_recovering')
        self.client.timing('rebalance_end_latency', self.ms_since(state['time_end']))

    def count(self, metric_name: str, count: int = 1) -> None:
        super().count(metric_name, count=count)
        self.client.increment(metric_name, value=count)

    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None:
        super().on_tp_commit(tp_offsets)
        for tp, offset in tp_offsets.items():
            self.client.gauge('committed_offset', offset, labels=self._format_label(tp))

    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
        super().track_tp_end_offset(tp, offset)
        self.client.gauge('end_offset', offset, labels=self._format_label(tp))

    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[str, Any], *, view: Any = None) -> None:
        super().on_web_request_end(app, request, response, state, view=view)
        status_code = int(state['status_code'])
        self.client.increment(f'http_status_code.{status_code}')
        self.client.timing('http_response_latency', self.ms_since(state['time_end']))

    def _format_label(self, tp: Optional[TP] = None, stream: Optional[StreamT] = None, table: Optional[CollectionT] = None) -> Dict[str, Any]:
        labels: Dict[str, Any] = {}
        if tp is not None:
            labels.update(self._format_tp_label(tp))
        if stream is not None:
            labels.update(self._format_stream_label(stream))
        if table is not None:
            labels.update(self._format_table_label(table))
        return labels

    def _format_tp_label(self, tp: TP) -> Dict[str, Union[str, int]]:
        return {'topic': tp.topic, 'partition': tp.partition}

    def _format_stream_label(self, stream: StreamT) -> Dict[str, str]:
        return {'stream': self._stream_label(stream)}

    def _stream_label(self, stream: StreamT) -> str:
        return self._normalize(stream.shortlabel.lstrip('Stream:')).strip('_').lower()

    def _format_table_label(self, table: CollectionT) -> Dict[str, str]:
        return {'table': table.name}

    @cached_property
    def client(self) -> DatadogStatsClient:
        """Return the datadog client."""
        return self._new_datadog_stats_client()
