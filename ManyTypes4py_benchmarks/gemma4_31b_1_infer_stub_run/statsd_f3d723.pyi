from typing import Any, Dict, Optional, Union, overload
from faust.types import (
    AppT,
    CollectionT,
    EventT,
    Message,
    PendingMessage,
    RecordMetadata,
    StreamT,
    TP,
)
from faust.types.assignor import PartitionAssignorT
from faust.types.transports import ConsumerT, ProducerT
from .monitor import Monitor, TPOffsetMapping
from statsd import StatsClient

__all__ = ['StatsdMonitor']

class StatsdMonitor(Monitor):
    """Statsd Faust Sensor.

    This sensor, records statistics to Statsd along with computing metrics
    for the stats server
    """
    host: str
    port: int
    prefix: str
    rate: float
    client: StatsClient

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8125,
        prefix: str = 'faust-app',
        rate: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

    def _new_statsd_client(self) -> StatsClient: ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Any: ...

    def _stream_label(self, stream: StreamT) -> str: ...

    def on_stream_event_out(
        self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Any] = None
    ) -> None: ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_table_get(self, table: CollectionT, key: Any) -> None: ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None: ...

    def on_table_del(self, table: CollectionT, key: Any) -> None: ...

    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None: ...

    def on_send_initiated(
        self, producer: ProducerT, topic: str, message: Any, keysize: int, valsize: int
    ) -> Any: ...

    def on_send_completed(
        self, producer: ProducerT, state: Any, metadata: RecordMetadata
    ) -> None: ...

    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None: ...

    def on_assignment_error(
        self, assignor: PartitionAssignorT, state: Dict[str, Any], exc: Exception
    ) -> None: ...

    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Dict[str, Any]
    ) -> None: ...

    def on_rebalance_start(self, app: AppT) -> Any: ...

    def on_rebalance_return(self, app: AppT, state: Dict[str, Any]) -> None: ...

    def on_rebalance_end(self, app: AppT, state: Dict[str, Any]) -> None: ...

    def count(self, metric_name: str, count: int = 1) -> None: ...

    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None: ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...

    def on_web_request_end(
        self,
        app: AppT,
        request: Any,
        response: Any,
        state: Dict[str, Any],
        *,
        view: Optional[Any] = None,
    ) -> None: ...