"""Monitor using Statsd."""

import typing
from typing import Any, Dict, Optional
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

if typing.TYPE_CHECKING:
    import statsd
    from statsd import StatsClient
else:
    class StatsClient:
        ...

__all__ = ['StatsdMonitor']

class StatsdMonitor:
    """Statsd Faust Sensor."""

    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'faust-app', rate: float = 1.0, **kwargs: Dict[str, Any]) -> None:
        ...

    def _new_statsd_client(self) -> StatsClient:
        ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Optional[float]:
        ...

    def _stream_label(self, stream: StreamT) -> str:
        ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[float] = None) -> None:
        ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        ...

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        ...

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        ...

    def on_commit_completed(self, consumer: ConsumerT, state: Dict[str, Any]) -> None:
        ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> None:
        ...

    def on_send_completed(self, producer: ProducerT, state: Dict[str, Any], metadata: RecordMetadata) -> None:
        ...

    def on_send_error(self, producer: ProducerT, exc: BaseException, state: Dict[str, Any]) -> None:
        ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Dict[str, Any], exc: BaseException) -> None:
        ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Dict[str, Any]) -> None:
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[str, Any]:
        ...

    def on_rebalance_return(self, app: AppT, state: Dict[str, Any]) -> None:
        ...

    def on_rebalance_end(self, app: AppT, state: Dict[str, Any]) -> None:
        ...

    def count(self, metric_name: str, count: int = 1) -> None:
        ...

    def on_tp_commit(self, tp_offsets: Dict[TP, int]) -> None:
        ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
        ...

    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[str, Any], *, view: Optional[web.View] = None) -> None:
        ...

    @property
    def client(self) -> StatsClient:
        ...