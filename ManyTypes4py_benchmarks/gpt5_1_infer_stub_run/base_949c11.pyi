from typing import Any, Dict, Iterator, Mapping, Optional, Set
from mode import Service
from faust import web
from faust.types import AppT, CollectionT, EventT, StreamT
from faust.types.assignor import PartitionAssignorT
from faust.types.tuples import Message, PendingMessage, RecordMetadata, TP
from faust.types.sensors import SensorDelegateT, SensorT
from faust.types.transports import ConsumerT, ProducerT

__all__: list[str] = ...

class Sensor(SensorT, Service):
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Optional[object]: ...
    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[object] = ...,
    ) -> None: ...
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_topic_buffer_full(self, tp: TP) -> None: ...
    def on_table_get(self, table: CollectionT, key: object) -> None: ...
    def on_table_set(self, table: CollectionT, key: object, value: object) -> None: ...
    def on_table_del(self, table: CollectionT, key: object) -> None: ...
    def on_commit_initiated(self, consumer: ConsumerT) -> Optional[object]: ...
    def on_commit_completed(self, consumer: ConsumerT, state: Optional[object]) -> None: ...
    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: PendingMessage,
        keysize: int,
        valsize: int,
    ) -> Optional[object]: ...
    def on_send_completed(
        self,
        producer: ProducerT,
        state: Optional[object],
        metadata: RecordMetadata,
    ) -> None: ...
    def on_send_error(
        self,
        producer: ProducerT,
        exc: BaseException,
        state: Optional[object],
    ) -> None: ...
    def on_assignment_start(self, assignor: PartitionAssignorT) -> Dict[str, float]: ...
    def on_assignment_error(
        self,
        assignor: PartitionAssignorT,
        state: Mapping[str, float],
        exc: BaseException,
    ) -> None: ...
    def on_assignment_completed(
        self,
        assignor: PartitionAssignorT,
        state: Mapping[str, float],
    ) -> None: ...
    def on_rebalance_start(self, app: AppT) -> Dict[str, float]: ...
    def on_rebalance_return(self, app: AppT, state: Mapping[str, float]) -> None: ...
    def on_rebalance_end(self, app: AppT, state: Mapping[str, float]) -> None: ...
    def on_web_request_start(
        self,
        app: AppT,
        request: web.Request,
        *,
        view: Optional[web.View] = ...,
    ) -> Dict[str, float]: ...
    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: web.Response,
        state: Mapping[str, float],
        *,
        view: Optional[web.View] = ...,
    ) -> None: ...
    def asdict(self) -> Dict[str, Any]: ...

class SensorDelegate(SensorDelegateT):
    app: AppT
    _sensors: Set[SensorT]

    def __init__(self, app: AppT) -> None: ...
    def add(self, sensor: SensorT) -> None: ...
    def remove(self, sensor: SensorT) -> None: ...
    def __iter__(self) -> Iterator[SensorT]: ...
    def on_message_in(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_stream_event_in(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
    ) -> Dict[SensorT, Optional[object]]: ...
    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Mapping[SensorT, Optional[object]]] = ...,
    ) -> None: ...
    def on_topic_buffer_full(self, tp: TP) -> None: ...
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None: ...
    def on_table_get(self, table: CollectionT, key: object) -> None: ...
    def on_table_set(self, table: CollectionT, key: object, value: object) -> None: ...
    def on_table_del(self, table: CollectionT, key: object) -> None: ...
    def on_commit_initiated(self, consumer: ConsumerT) -> Dict[SensorT, Optional[object]]: ...
    def on_commit_completed(
        self,
        consumer: ConsumerT,
        state: Mapping[SensorT, Optional[object]],
    ) -> None: ...
    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: PendingMessage,
        keysize: int,
        valsize: int,
    ) -> Dict[SensorT, Optional[object]]: ...
    def on_send_completed(
        self,
        producer: ProducerT,
        state: Mapping[SensorT, Optional[object]],
        metadata: RecordMetadata,
    ) -> None: ...
    def on_send_error(
        self,
        producer: ProducerT,
        exc: BaseException,
        state: Mapping[SensorT, Optional[object]],
    ) -> None: ...
    def on_assignment_start(
        self,
        assignor: PartitionAssignorT,
    ) -> Dict[SensorT, Mapping[str, float]]: ...
    def on_assignment_error(
        self,
        assignor: PartitionAssignorT,
        state: Mapping[SensorT, Mapping[str, float]],
        exc: BaseException,
    ) -> None: ...
    def on_assignment_completed(
        self,
        assignor: PartitionAssignorT,
        state: Mapping[SensorT, Mapping[str, float]],
    ) -> None: ...
    def on_rebalance_start(self, app: AppT) -> Dict[SensorT, Mapping[str, float]]: ...
    def on_rebalance_return(
        self,
        app: AppT,
        state: Mapping[SensorT, Mapping[str, float]],
    ) -> None: ...
    def on_rebalance_end(
        self,
        app: AppT,
        state: Mapping[SensorT, Mapping[str, float]],
    ) -> None: ...
    def on_web_request_start(
        self,
        app: AppT,
        request: web.Request,
        *,
        view: Optional[web.View] = ...,
    ) -> Dict[SensorT, Mapping[str, float]]: ...
    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: web.Response,
        state: Mapping[SensorT, Mapping[str, float]],
        *,
        view: Optional[web.View] = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...