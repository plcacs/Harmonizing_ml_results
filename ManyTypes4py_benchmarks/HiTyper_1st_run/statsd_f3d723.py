"""Monitor using Statsd."""
import typing
from typing import Any, Dict, Optional, cast
from mode.utils.objects import cached_property
from faust import web
from faust.exceptions import ImproperlyConfigured
from faust.types import AppT, CollectionT, EventT, Message, PendingMessage, RecordMetadata, StreamT, TP
from faust.types.assignor import PartitionAssignorT
from faust.types.transports import ConsumerT, ProducerT
from .monitor import Monitor, TPOffsetMapping
try:
    import statsd
except ImportError:
    statsd = None
if typing.TYPE_CHECKING:
    from statsd import StatsClient
else:

    class StatsClient:
        ...
__all__ = ['StatsdMonitor']

class StatsdMonitor(Monitor):
    """Statsd Faust Sensor.

    This sensor, records statistics to Statsd along with computing metrics
    for the stats server
    """

    def __init__(self, host: typing.Text='localhost', port: int=8125, prefix: typing.Text='faust-app', rate: float=1.0, **kwargs) -> None:
        self.host = host
        self.port = port
        self.prefix = prefix
        self.rate = rate
        if statsd is None:
            raise ImproperlyConfigured('StatsMonitor requires `pip install statsd`.')
        super().__init__(**kwargs)

    def _new_statsd_client(self) -> StatsClient:
        return statsd.StatsClient(host=self.host, port=self.port, prefix=self.prefix)

    def on_message_in(self, tp: Union[int, faustypes.TP, faustypes.Message], offset: Union[int, faustypes.Message, faustypes.TP], message: Union[faustypes.Message, int, faustypes.TP]) -> None:
        """Call before message is delegated to streams."""
        super().on_message_in(tp, offset, message)
        self.client.incr('messages_received', rate=self.rate)
        self.client.incr('messages_active', rate=self.rate)
        self.client.incr(f'topic.{tp.topic}.messages_received', rate=self.rate)
        self.client.gauge(f'read_offset.{tp.topic}.{tp.partition}', offset)

    def on_stream_event_in(self, tp: Union[int, faustypes.EventT, faustypes.tuples.TP], offset: Union[int, faustypes.EventT, faustypes.tuples.TP], stream: Union[int, faustypes.EventT, faustypes.tuples.TP], event: Union[int, faustypes.EventT, faustypes.tuples.TP]):
        """Call when stream starts processing an event."""
        state = super().on_stream_event_in(tp, offset, stream, event)
        self.client.incr('events', rate=self.rate)
        self.client.incr(f'stream.{self._stream_label(stream)}.events', rate=self.rate)
        self.client.incr('events_active', rate=self.rate)
        return state

    def _stream_label(self, stream: Union[faustypes.StreamT, bytes]) -> Union[str, None]:
        return self._normalize(stream.shortlabel.lstrip('Stream:')).strip('_').lower()

    def on_stream_event_out(self, tp: Union[int, dict, faustypes.EventT], offset: Union[int, dict, faustypes.EventT], stream: Union[int, dict, faustypes.EventT], event: Union[int, dict, faustypes.EventT], state: Union[None, int, dict, faustypes.EventT]=None) -> None:
        """Call when stream is done processing an event."""
        super().on_stream_event_out(tp, offset, stream, event, state)
        self.client.decr('events_active', rate=self.rate)
        self.client.timing('events_runtime', self.secs_to_ms(self.events_runtime[-1]), rate=self.rate)

    def on_message_out(self, tp: Union[faustypes.Message, int, faustypes.TP], offset: Union[faustypes.Message, int, faustypes.TP], message: Union[faustypes.Message, int, faustypes.TP]) -> None:
        """Call when message is fully acknowledged and can be committed."""
        super().on_message_out(tp, offset, message)
        self.client.decr('messages_active', rate=self.rate)

    def on_table_get(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when value in table is retrieved."""
        super().on_table_get(table, key)
        self.client.incr(f'table.{table.name}.keys_retrieved', rate=self.rate)

    def on_table_set(self, table: faustypes.CollectionT, key: Union[faustypes.CollectionT, str], value: Union[faustypes.CollectionT, str]) -> None:
        """Call when new value for key in table is set."""
        super().on_table_set(table, key, value)
        self.client.incr(f'table.{table.name}.keys_updated', rate=self.rate)

    def on_table_del(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when key in a table is deleted."""
        super().on_table_del(table, key)
        self.client.incr(f'table.{table.name}.keys_deleted', rate=self.rate)

    def on_commit_completed(self, consumer: Union[faustypes.transports.ConsumerT, mode.Service.T, str], state: Any) -> None:
        """Call when consumer commit offset operation completed."""
        super().on_commit_completed(consumer, state)
        self.client.timing('commit_latency', self.ms_since(cast(float, state)), rate=self.rate)

    def on_send_initiated(self, producer: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], topic: Union[str, int, faustypes.transports.ProducerT], message: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], keysize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], valsize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage]) -> Union[aiohttp.web.Request, bool]:
        """Call when message added to producer buffer."""
        self.client.incr(f'topic.{topic}.messages_sent', rate=self.rate)
        return super().on_send_initiated(producer, topic, message, keysize, valsize)

    def on_send_completed(self, producer: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata], state: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata, dict], metadata: Union[faustypes.transports.ProducerT, faustypes.RecordMetadata]) -> None:
        """Call when producer finished sending message."""
        super().on_send_completed(producer, state, metadata)
        self.client.incr('messages_sent', rate=self.rate)
        self.client.timing('send_latency', self.ms_since(cast(float, state)), rate=self.rate)

    def on_send_error(self, producer: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], exc: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], state: Union[BaseException, faustypes.transports.ProducerT, typing.Callable[..., None]]) -> None:
        """Call when producer was unable to publish message."""
        super().on_send_error(producer, exc, state)
        self.client.incr('messages_sent_error', rate=self.rate)
        self.client.timing('send_latency_for_error', self.ms_since(cast(float, state)), rate=self.rate)

    def on_assignment_error(self, assignor: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], state: Union[dict, faustypes.assignor.PartitionAssignorT], exc: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict]) -> None:
        """Partition assignor did not complete assignor due to error."""
        super().on_assignment_error(assignor, state, exc)
        self.client.incr('assignments_error', rate=self.rate)
        self.client.timing('assignment_latency', self.ms_since(state['time_start']), rate=self.rate)

    def on_assignment_completed(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT], state: Union[dict, faustypes.assignor.PartitionAssignorT]) -> None:
        """Partition assignor completed assignment."""
        super().on_assignment_completed(assignor, state)
        self.client.incr('assignments_complete', rate=self.rate)
        self.client.timing('assignment_latency', self.ms_since(state['time_start']), rate=self.rate)

    def on_rebalance_start(self, app: Union[aiohttp.web.Application, faustypes.AppT]) -> Union[tuple[str], list[str]]:
        """Cluster rebalance in progress."""
        state = super().on_rebalance_start(app)
        self.client.incr('rebalances', rate=self.rate)
        return state

    def on_rebalance_return(self, app: Union[faustypes.AppT, dict], state: Union[dict, faustypes.AppT]) -> None:
        """Consumer replied assignment is done to broker."""
        super().on_rebalance_return(app, state)
        self.client.decr('rebalances', rate=self.rate)
        self.client.incr('rebalances_recovering', rate=self.rate)
        self.client.timing('rebalance_return_latency', self.ms_since(state['time_return']), rate=self.rate)

    def on_rebalance_end(self, app: Union[faustypes.AppT, dict], state: Union[dict, faustypes.AppT]) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        super().on_rebalance_end(app, state)
        self.client.decr('rebalances_recovering', rate=self.rate)
        self.client.timing('rebalance_end_latency', self.ms_since(state['time_end']), rate=self.rate)

    def count(self, metric_name: Union[str, int], count: int=1) -> None:
        """Count metric by name."""
        super().count(metric_name, count=count)
        self.client.incr(metric_name, count=count, rate=self.rate)

    def on_tp_commit(self, tp_offsets: Union[faustypes.TP, TPOffsetMapping]) -> None:
        """Call when offset in topic partition is committed."""
        super().on_tp_commit(tp_offsets)
        for tp, offset in tp_offsets.items():
            metric_name = f'committed_offset.{tp.topic}.{tp.partition}'
            self.client.gauge(metric_name, offset)

    def track_tp_end_offset(self, tp: Union[int, faustypes.TP, str], offset: Union[int, faustypes.TP]) -> None:
        """Track new topic partition end offset for monitoring lags."""
        super().track_tp_end_offset(tp, offset)
        metric_name = f'end_offset.{tp.topic}.{tp.partition}'
        self.client.gauge(metric_name, offset)

    def on_web_request_end(self, app: Union[faustypes.AppT, dict], request: Union[faustypes.AppT, dict], response: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict], *, view: Union[None, faustypes.AppT, dict]=None) -> None:
        """Web server finished working on request."""
        super().on_web_request_end(app, request, response, state, view=view)
        status_code = int(state['status_code'])
        self.client.incr(f'http_status_code.{status_code}', rate=self.rate)
        self.client.timing('http_response_latency', self.ms_since(state['time_end']), rate=self.rate)

    @cached_property
    def client(self):
        """Return statsd client."""
        return self._new_statsd_client()