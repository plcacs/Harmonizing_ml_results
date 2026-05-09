"""Monitor using Promethus."""

import typing
from typing import Any, cast, Dict, List, Optional, Union
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
from faust.types.transports import ConsumerT, ProducerT
from faust.types.assignor import PartitionAssignorT
from aiohttp.web import Response
from prometheus_client import Counter, Gauge, Histogram

__all__: List[str] = ['PrometheusMonitor']

class PrometheusMonitor:
    """Prometheus Faust Sensor."""
    ERROR: str = ...
    COMPLETED: str = ...
    KEYS_RETRIEVED: str = ...
    KEYS_UPDATED: str = ...
    KEYS_DELETED: str = ...

    def __init__(self, app: AppT, pattern: str = '/metrics', **kwargs: Any) -> None:
        ...

    def _initialize_metrics(self) -> None:
        ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        ...

    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Dict[str, float]:
        ...

    def _stream_label(self, stream: StreamT) -> str:
        ...

    def on_stream_event_out(
        self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, float]] = None
    ) -> None:
        ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        ...

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        ...

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        ...

    def on_commit_completed(self, consumer: ConsumerT, state: Dict[str, float]) -> None:
        ...

    def on_send_initiated(
        self, producer: ProducerT, topic: str, message: Message, keysize: int, valsize: int
    ) -> Any:
        ...

    def on_send_completed(
        self, producer: ProducerT, state: Dict[str, float], metadata: RecordMetadata
    ) -> None:
        ...

    def on_send_error(
        self, producer: ProducerT, exc: Exception, state: Dict[str, float]
    ) -> None:
        ...

    def on_assignment_error(
        self, assignor: PartitionAssignorT, state: Dict[str, float], exc: Exception
    ) -> None:
        ...

    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Dict[str, float]
    ) -> None:
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[str, float]:
        ...

    def on_rebalance_return(
        self, app: AppT, state: Dict[str, float]
    ) -> None:
        ...

    def on_rebalance_end(
        self, app: AppT, state: Dict[str, float]
    ) -> None:
        ...

    def count(self, metric_name: str, count: int = 1) -> None:
        ...

    def on_tp_commit(self, tp_offsets: Dict[TP, int]) -> None:
        ...

    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
        ...

    def on_web_request_end(
        self, app: AppT, request: Any, response: Response, state: Dict[str, float], *, view: Optional[Any] = None
    ) -> None:
        ...

    def expose_metrics(self) -> None:
        ...