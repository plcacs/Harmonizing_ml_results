"""Monitor using Statsd."""

import typing
from typing import Any, Dict, List, Optional, Union
from faust.types import (
    AppT,
    CollectionT,
    EventT,
    Message,
    PendingMessage,
    RecordMetadata,
    StreamT,
    TP,
    PartitionAssignorT,
    ConsumerT,
    ProducerT,
)
from .monitor import Monitor, TPOffsetMapping

if typing.TYPE_CHECKING:
    import statsd
    from statsd import StatsClient
else:
    class StatsClient: ...
    statsd = None

__all__ = ['StatsdMonitor']

class StatsdMonitor(Monitor):
    """Statsd Faust Sensor."""

    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Any) -> None: ...

    def _new_statsd_client(self) -> StatsClient: ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Dict: ...

    def _stream_label(self, stream: StreamT) -> str: ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict] = None) -> None: ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...

    def on_table_get(self, table: CollectionT, key: Any) -> None: ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None: ...

    def on_table_del(self, table: CollectionT, key: Any) -> None: ...

    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None: ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> None: ...

    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None: ...

    def on_send_error(self, producer: ProducerT, exc: BaseException, state: Any) -> None: ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Any, exc: BaseException) -> None: ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Any) -> None: ...

    def on_rebalance_start(self, app: AppT) -> Dict: ...

    def on_rebalance_return(self, app: AppT, state: Dict) -> None: ...

    def on_rebalance_end(self, app: AppT, state: Dict) -> None: ...

    def count(self, metric_name: str, count: int = 1) -> None: ...

    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None: ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None: ...

    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: web.Response,
        state: Dict,
        *,
        view: Optional[web.View] = None,
    ) -> None: ...

    @property
    def client(self) -> StatsClient: ...