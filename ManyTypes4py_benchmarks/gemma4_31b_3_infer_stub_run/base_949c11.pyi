"""Base-interface for sensors."""
from typing import Any, Dict, Iterator, Optional, Set
from mode import Service
from faust import web
from faust.types import AppT, CollectionT, EventT, StreamT
from faust.types.assignor import PartitionAssignorT
from faust.types.tuples import Message, PendingMessage, RecordMetadata, TP
from faust.types.sensors import SensorDelegateT, SensorT
from faust.types.transports import ConsumerT, ProducerT

__all__ = ['Sensor', 'SensorDelegate']

class Sensor(SensorT, Service):
    """Base class for sensors.

    This sensor does not do anything at all, but can be subclassed
    to create new monitors.
    """

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        """Message received by a consumer."""
        ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Optional[Any]:
        """Message sent to a stream as an event."""
        ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Any] = ...) -> None:
        """Event was acknowledged by stream."""
        ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        """All streams finished processing message."""
        ...

    def on_topic_buffer_full(self, tp: TP) -> None:
        """Topic buffer full so conductor had to wait."""
        ...

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        """Key retrieved from table."""
        ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        """Value set for key in table."""
        ...

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        """Key deleted from table."""
        ...

    def on_commit_initiated(self, consumer: ConsumerT) -> Optional[Any]:
        """Consumer is about to commit topic offset."""
        ...

    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None:
        """Consumer finished committing topic offset."""
        ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: Message, keysize: int, valsize: int) -> Optional[Any]:
        """About to send a message."""
        ...

    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None:
        """Message successfully sent."""
        ...

    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None:
        """Error while sending message."""
        ...

    def on_assignment_start(self, assignor: PartitionAssignorT) -> Dict[str, float]:
        """Partition assignor is starting to assign partitions."""
        ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Any, exc: Exception) -> None:
        """Partition assignor did not complete assignor due to error."""
        ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Any) -> None:
        """Partition assignor completed assignment."""
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[str, float]:
        """Cluster rebalance in progress."""
        ...

    def on_rebalance_return(self, app: AppT, state: Any) -> None:
        """Consumer replied assignment is done to broker."""
        ...

    def on_rebalance_end(self, app: AppT, state: Any) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        ...

    def on_web_request_start(self, app: AppT, request: web.Request, *, view: Optional[Any] = ...) -> Dict[str, float]:
        """Web server started working on request."""
        ...

    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Any, *, view: Optional[Any] = ...) -> None:
        """Web server finished working on request."""
        ...

    def asdict(self) -> Dict[str, Any]:
        """Convert sensor state to dictionary."""
        ...

class SensorDelegate(SensorDelegateT):
    """A class that delegates sensor methods to a list of sensors."""

    def __init__(self, app: AppT) -> None: ...
    app: AppT
    _sensors: Set[Sensor]

    def add(self, sensor: Sensor) -> None:
        """Add sensor."""
        ...

    def remove(self, sensor: Sensor) -> None:
        """Remove sensor."""
        ...

    def __iter__(self) -> Iterator[Sensor]: ...

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        """Call before message is delegated to streams."""
        ...

    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Dict[Sensor, Optional[Any]]:
        """Call when stream starts processing an event."""
        ...

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[Sensor, Any]] = ...) -> None:
        """Call when stream is done processing an event."""
        ...

    def on_topic_buffer_full(self, tp: TP) -> None:
        """Call when conductor topic buffer is full and has to wait."""
        ...

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        """Call when message is fully acknowledged and can be committed."""
        ...

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        """Call when value in table is retrieved."""
        ...

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        """Call when new value for key in table is set."""
        ...

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        """Call when key in a table is deleted."""
        ...

    def on_commit_initiated(self, consumer: ConsumerT) -> Dict[Sensor, Optional[Any]]:
        """Call when consumer commit offset operation starts."""
        ...

    def on_commit_completed(self, consumer: ConsumerT, state: Dict[Sensor, Any]) -> None:
        """Call when consumer commit offset operation completed."""
        ...

    def on_send_initiated(self, producer: ProducerT, topic: str, message: Message, keysize: int, valsize: int) -> Dict[Sensor, Optional[Any]]:
        """Call when message added to producer buffer."""
        ...

    def on_send_completed(self, producer: ProducerT, state: Dict[Sensor, Any], metadata: RecordMetadata) -> None:
        """Call when producer finished sending message."""
        ...

    def on_send_error(self, producer: ProducerT, exc: Exception, state: Dict[Sensor, Any]) -> None:
        """Call when producer was unable to publish message."""
        ...

    def on_assignment_start(self, assignor: PartitionAssignorT) -> Dict[Sensor, Dict[str, float]]:
        """Partition assignor is starting to assign partitions."""
        ...

    def on_assignment_error(self, assignor: PartitionAssignorT, state: Dict[Sensor, Any], exc: Exception) -> None:
        """Partition assignor did not complete assignor due to error."""
        ...

    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Dict[Sensor, Any]) -> None:
        """Partition assignor completed assignment."""
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[Sensor, Dict[str, float]]:
        """Cluster rebalance in progress."""
        ...

    def on_rebalance_return(self, app: AppT, state: Dict[Sensor, Any]) -> None:
        """Consumer replied assignment is done to broker."""
        ...

    def on_rebalance_end(self, app: AppT, state: Dict[Sensor, Any]) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        ...

    def on_web_request_start(self, app: AppT, request: web.Request, *, view: Optional[Any] = ...) -> Dict[Sensor, Dict[str, float]]:
        """Web server started working on request."""
        ...

    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[Sensor, Any], *, view: Optional[Any] = ...) -> None:
        """Web server finished working on request."""
        ...

    def __repr__(self) -> str: ...