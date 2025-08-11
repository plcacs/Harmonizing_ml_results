"""Monitor using datadog."""
import re
from typing import Any, Dict, List, Optional, cast
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
    datadog = None

    class DogStatsD:
        ...
__all__ = ['DatadogMonitor']

class DatadogStatsClient:
    """Statsd compliant datadog client."""

    def __init__(self, host: typing.Text='localhost', port: int=8125, prefix: typing.Text='faust-app', rate: float=1.0, **kwargs) -> None:
        self.client = DogStatsd(host=host, port=port, namespace=prefix, **kwargs)
        self.rate = rate
        self.sanitize_re = re.compile('[^0-9a-zA-Z_]')
        self.re_substitution = '_'

    def gauge(self, metric: Union[float, str, dict], value: Union[float, str, dict], labels: Union[None, float, str, dict]=None) -> None:
        self.client.gauge(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def increment(self, metric: Union[dict, str, float], value: float=1.0, labels: Union[None, dict, str, float]=None) -> None:
        self.client.increment(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def incr(self, metric: Union[str, int, float], count: int=1) -> None:
        """Statsd compatibility."""
        self.increment(metric, value=count)

    def decrement(self, metric: Union[dict, float, str], value: float=1.0, labels: Union[None, dict, float, str]=None) -> Union[typing.Callable, set[str], int]:
        return self.client.decrement(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def decr(self, metric: Union[str, float, int], count: float=1.0) -> None:
        """Statsd compatibility."""
        self.decrement(metric, value=count)

    def timing(self, metric: Union[float, str, dict], value: Union[float, str, dict], labels: Union[None, float, str, dict]=None) -> None:
        self.client.timing(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def timed(self, metric: Union[None, bool, str, list[str]]=None, labels: Union[None, bool, str, list[str]]=None, use_ms: Union[None, bool, str, list[str]]=None) -> bool:
        return self.client.timed(metric=metric, tags=self._encode_labels(labels), sample_rate=self.rate, use_ms=use_ms)

    def histogram(self, metric: Union[float, dict, str], value: Union[float, dict, str], labels: Union[None, float, dict, str]=None) -> None:
        self.client.histogram(metric, value=value, tags=self._encode_labels(labels), sample_rate=self.rate)

    def _encode_labels(self, labels: dict) -> Union[list[typing.Text], None]:

        def sanitize(s: Any):
            return self.sanitize_re.sub(self.re_substitution, str(s))
        return [f'{sanitize(k)}:{sanitize(v)}' for k, v in labels.items()] if labels else None

class DatadogMonitor(Monitor):
    """Datadog Faust Sensor.

    This sensor, records statistics to datadog agents along
    with computing metrics for the stats server
    """

    def __init__(self, host: typing.Text='localhost', port: int=8125, prefix: typing.Text='faust-app', rate: float=1.0, **kwargs) -> None:
        self.host = host
        self.port = port
        self.prefix = prefix
        self.rate = rate
        if datadog is None:
            raise ImproperlyConfigured(f'{type(self).__name__} requires `pip install datadog`.')
        super().__init__(**kwargs)

    def _new_datadog_stats_client(self) -> DatadogStatsClient:
        return DatadogStatsClient(host=self.host, port=self.port, prefix=self.prefix, rate=self.rate)

    def on_message_in(self, tp: Union[int, faustypes.Message, faustypes.TP], offset: Union[int, faustypes.Message, faustypes.TP], message: Union[faustypes.Message, int, faustypes.TP]) -> None:
        """Call before message is delegated to streams."""
        super().on_message_in(tp, offset, message)
        labels = self._format_label(tp)
        self.client.increment('messages_received', labels=labels)
        self.client.increment('messages_active', labels=labels)
        self.client.increment('topic_messages_received', labels={'topic': tp.topic})
        self.client.gauge('read_offset', offset, labels=labels)

    def on_stream_event_in(self, tp: Union[faustypes.StreamT, int, raiden.utils.BlockNumber], offset: Union[int, faustypes.EventT, faustypes.tuples.TP], stream: Union[faustypes.StreamT, int, raiden.utils.BlockNumber], event: Union[int, faustypes.EventT, faustypes.tuples.TP]):
        """Call when stream starts processing an event."""
        state = super().on_stream_event_in(tp, offset, stream, event)
        labels = self._format_label(tp, stream)
        self.client.increment('events', labels=labels)
        self.client.increment('events_active', labels=labels)
        return state

    def on_stream_event_out(self, tp: Union[int, dict, faustypes.EventT], offset: Union[int, dict, faustypes.EventT], stream: Union[int, dict, faustypes.EventT], event: Union[int, dict, faustypes.EventT], state: Union[None, int, dict, faustypes.EventT]=None) -> None:
        """Call when stream is done processing an event."""
        super().on_stream_event_out(tp, offset, stream, event, state)
        labels = self._format_label(tp, stream)
        self.client.decrement('events_active', labels=labels)
        self.client.timing('events_runtime', self.secs_to_ms(self.events_runtime[-1]), labels=labels)

    def on_message_out(self, tp: Union[faustypes.TP, int, faustypes.Message], offset: Union[faustypes.Message, int, faustypes.TP], message: Union[faustypes.Message, int, faustypes.TP]) -> None:
        """Call when message is fully acknowledged and can be committed."""
        super().on_message_out(tp, offset, message)
        self.client.decrement('messages_active', labels=self._format_label(tp))

    def on_table_get(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when value in table is retrieved."""
        super().on_table_get(table, key)
        self.client.increment('table_keys_retrieved', labels=self._format_label(table=table))

    def on_table_set(self, table: Union[faustypes.CollectionT, str], key: Union[faustypes.CollectionT, str], value: Union[faustypes.CollectionT, str]) -> None:
        """Call when new value for key in table is set."""
        super().on_table_set(table, key, value)
        self.client.increment('table_keys_updated', labels=self._format_label(table=table))

    def on_table_del(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when key in a table is deleted."""
        super().on_table_del(table, key)
        self.client.increment('table_keys_deleted', labels=self._format_label(table=table))

    def on_commit_completed(self, consumer: Union[faustypes.transports.ConsumerT, mode.Service.T, str], state: Any) -> None:
        """Call when consumer commit offset operation completed."""
        super().on_commit_completed(consumer, state)
        self.client.timing('commit_latency', self.ms_since(cast(float, state)))

    def on_send_initiated(self, producer: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], topic: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], message: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], keysize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], valsize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage]) -> Union[aiohttp.web.Request, bool]:
        """Call when message added to producer buffer."""
        self.client.increment('topic_messages_sent', labels={'topic': topic})
        return super().on_send_initiated(producer, topic, message, keysize, valsize)

    def on_send_completed(self, producer: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata], state: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata, dict], metadata: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata]) -> None:
        """Call when producer finished sending message."""
        super().on_send_completed(producer, state, metadata)
        self.client.increment('messages_sent')
        self.client.timing('send_latency', self.ms_since(cast(float, state)))

    def on_send_error(self, producer: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], exc: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], state: Union[BaseException, faustypes.transports.ProducerT, typing.Callable[..., None]]) -> None:
        """Call when producer was unable to publish message."""
        super().on_send_error(producer, exc, state)
        self.client.increment('messages_send_failed')
        self.client.timing('send_latency_for_error', self.ms_since(cast(float, state)))

    def on_assignment_error(self, assignor: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], state: Union[dict, faustypes.assignor.PartitionAssignorT], exc: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict]) -> None:
        """Partition assignor did not complete assignor due to error."""
        super().on_assignment_error(assignor, state, exc)
        self.client.increment('assignments_error')
        self.client.timing('assignment_latency', self.ms_since(state['time_start']))

    def on_assignment_completed(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT], state: Union[dict, faustypes.assignor.PartitionAssignorT]) -> None:
        """Partition assignor completed assignment."""
        super().on_assignment_completed(assignor, state)
        self.client.increment('assignments_complete')
        self.client.timing('assignment_latency', self.ms_since(state['time_start']))

    def on_rebalance_start(self, app: Union[aiohttp.web.Application, faustypes.AppT]) -> Union[tuple[str], list[str]]:
        """Cluster rebalance in progress."""
        state = super().on_rebalance_start(app)
        self.client.increment('rebalances')
        return state

    def on_rebalance_return(self, app: Union[faustypes.AppT, dict], state: Union[dict, faustypes.AppT]) -> None:
        """Consumer replied assignment is done to broker."""
        super().on_rebalance_return(app, state)
        self.client.decrement('rebalances')
        self.client.increment('rebalances_recovering')
        self.client.timing('rebalance_return_latency', self.ms_since(state['time_return']))

    def on_rebalance_end(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        super().on_rebalance_end(app, state)
        self.client.decrement('rebalances_recovering')
        self.client.timing('rebalance_end_latency', self.ms_since(state['time_end']))

    def count(self, metric_name: Union[str, int], count: int=1) -> None:
        """Count metric by name."""
        super().count(metric_name, count=count)
        self.client.increment(metric_name, value=count)

    def on_tp_commit(self, tp_offsets: Union[faustypes.TP, TPOffsetMapping]) -> None:
        """Call when offset in topic partition is committed."""
        super().on_tp_commit(tp_offsets)
        for tp, offset in tp_offsets.items():
            self.client.gauge('committed_offset', offset, labels=self._format_label(tp))

    def track_tp_end_offset(self, tp: Union[int, faustypes.TP, str], offset: Union[int, faustypes.TP, str]) -> None:
        """Track new topic partition end offset for monitoring lags."""
        super().track_tp_end_offset(tp, offset)
        self.client.gauge('end_offset', offset, labels=self._format_label(tp))

    def on_web_request_end(self, app: Union[faustypes.AppT, dict], request: Union[faustypes.AppT, dict], response: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict], *, view: Union[None, faustypes.AppT, dict]=None) -> None:
        """Web server finished working on request."""
        super().on_web_request_end(app, request, response, state, view=view)
        status_code = int(state['status_code'])
        self.client.increment(f'http_status_code.{status_code}')
        self.client.timing('http_response_latency', self.ms_since(state['time_end']))

    def _format_label(self, tp: Union[None, dict[str, typing.Any], bytes, str]=None, stream: Union[None, str, bytes, dict[str, typing.Any]]=None, table: Union[None, bytes, dict[str, typing.Any], typing.Any]=None) -> dict:
        labels = {}
        if tp is not None:
            labels.update(self._format_tp_label(tp))
        if stream is not None:
            labels.update(self._format_stream_label(stream))
        if table is not None:
            labels.update(self._format_table_label(table))
        return labels

    def _format_tp_label(self, tp: Union[faustypes.TP, typing.Type]) -> dict[typing.Text, ]:
        return {'topic': tp.topic, 'partition': tp.partition}

    def _format_stream_label(self, stream: Union[typing.BinaryIO, typing.IO]) -> dict[typing.Text, ]:
        return {'stream': self._stream_label(stream)}

    def _stream_label(self, stream: Union[faustypes.StreamT, bytes]) -> Union[str, None]:
        return self._normalize(stream.shortlabel.lstrip('Stream:')).strip('_').lower()

    def _format_table_label(self, table: Union[faustypes.CollectionT, dict[str, dict[str, str]]]) -> dict[typing.Text, ]:
        return {'table': table.name}

    @cached_property
    def client(self) -> Union[tuple[str], str, None]:
        """Return the datadog client."""
        return self._new_datadog_stats_client()