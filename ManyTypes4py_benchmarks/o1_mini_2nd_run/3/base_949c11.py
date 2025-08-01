"""Base-interface for sensors."""
from time import monotonic
from typing import Any, Dict, Iterator, Optional, Set, Type, Union
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

    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Optional[Any]:
        """Message sent to a stream as an event."""
        return None

    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Any] = None,
    ) -> None:
        """Event was acknowledged by stream.

        Notes:
            Acknowledged means a stream finished processing the event, but
            given that multiple streams may be handling the same event,
            the message cannot be committed before all streams have
            processed it.  When all streams have acknowledged the event,
            it will go through :meth:`on_message_out` just before offsets
            are committed.
        """
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

    def on_commit_initiated(self, consumer: ConsumerT) -> Any:
        """Consumer is about to commit topic offset."""
        ...

    def on_commit_completed(
        self, consumer: ConsumerT, state: Any
    ) -> None:
        """Consumer finished committing topic offset."""
        ...

    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: Message,
        keysize: int,
        valsize: int,
    ) -> Any:
        """About to send a message."""
        ...

    def on_send_completed(
        self,
        producer: ProducerT,
        state: Any,
        metadata: RecordMetadata,
    ) -> None:
        """Message successfully sent."""
        ...

    def on_send_error(
        self,
        producer: ProducerT,
        exc: Exception,
        state: Any,
    ) -> None:
        """Error while sending message."""
        ...

    def on_assignment_start(
        self, assignor: PartitionAssignorT
    ) -> Dict[SensorT, Any]:
        """Partition assignor is starting to assign partitions."""
        return {'time_start': monotonic()}

    def on_assignment_error(
        self,
        assignor: PartitionAssignorT,
        state: Any,
        exc: Exception,
    ) -> None:
        """Partition assignor did not complete assignor due to error."""
        ...

    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Any
    ) -> None:
        """Partition assignor completed assignment."""
        ...

    def on_rebalance_start(self, app: AppT) -> Dict[SensorT, Any]:
        """Cluster rebalance in progress."""
        return {'time_start': monotonic()}

    def on_rebalance_return(
        self, app: AppT, state: Any
    ) -> None:
        """Consumer replied assignment is done to broker."""
        ...

    def on_rebalance_end(
        self, app: AppT, state: Any
    ) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        ...

    def on_web_request_start(
        self,
        app: AppT,
        request: web.Request,
        *,
        view: Optional[web.View] = None,
    ) -> Dict[SensorT, Any]:
        """Web server started working on request."""
        return {'time_start': monotonic()}

    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: web.Response,
        state: Any,
        *,
        view: Optional[web.View] = None,
    ) -> None:
        """Web server finished working on request."""
        ...

    def asdict(self) -> Dict[str, Any]:
        """Convert sensor state to dictionary."""
        return {}


class SensorDelegate(SensorDelegateT):
    """A class that delegates sensor methods to a list of sensors."""

    def __init__(self, app: AppT) -> None:
        self.app: AppT = app
        self._sensors: Set[SensorT] = set()

    def add(self, sensor: SensorT) -> None:
        """Add sensor."""
        sensor.beacon = self.app.beacon.new(sensor)
        self._sensors.add(sensor)

    def remove(self, sensor: SensorT) -> None:
        """Remove sensor."""
        self._sensors.remove(sensor)

    def __iter__(self) -> Iterator[SensorT]:
        return iter(self._sensors)

    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
        """Call before message is delegated to streams."""
        for sensor in self._sensors:
            sensor.on_message_in(tp, offset, message)

    def on_stream_event_in(
        self, tp: TP, offset: int, stream: StreamT, event: EventT
    ) -> Dict[SensorT, Any]:
        """Call when stream starts processing an event."""
        return {sensor: sensor.on_stream_event_in(tp, offset, stream, event) for sensor in self._sensors}

    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Dict[SensorT, Any]] = None,
    ) -> None:
        """Call when stream is done processing an event."""
        sensor_state = state or {}
        for sensor in self._sensors:
            sensor.on_stream_event_out(tp, offset, stream, event, sensor_state.get(sensor))

    def on_topic_buffer_full(self, tp: TP) -> None:
        """Call when conductor topic buffer is full and has to wait."""
        for sensor in self._sensors:
            sensor.on_topic_buffer_full(tp)

    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
        """Call when message is fully acknowledged and can be committed."""
        for sensor in self._sensors:
            sensor.on_message_out(tp, offset, message)

    def on_table_get(self, table: CollectionT, key: Any) -> None:
        """Call when value in table is retrieved."""
        for sensor in self._sensors:
            sensor.on_table_get(table, key)

    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
        """Call when new value for key in table is set."""
        for sensor in self._sensors:
            sensor.on_table_set(table, key, value)

    def on_table_del(self, table: CollectionT, key: Any) -> None:
        """Call when key in a table is deleted."""
        for sensor in self._sensors:
            sensor.on_table_del(table, key)

    def on_commit_initiated(self, consumer: ConsumerT) -> Dict[SensorT, Any]:
        """Call when consumer commit offset operation starts."""
        return {sensor: sensor.on_commit_initiated(consumer) for sensor in self._sensors}

    def on_commit_completed(
        self, consumer: ConsumerT, state: Dict[SensorT, Any]
    ) -> None:
        """Call when consumer commit offset operation completed."""
        for sensor in self._sensors:
            sensor.on_commit_completed(consumer, state[sensor])

    def on_send_initiated(
        self,
        producer: ProducerT,
        topic: str,
        message: Message,
        keysize: int,
        valsize: int,
    ) -> Dict[SensorT, Any]:
        """Call when message added to producer buffer."""
        return {sensor: sensor.on_send_initiated(producer, topic, message, keysize, valsize) for sensor in self._sensors}

    def on_send_completed(
        self,
        producer: ProducerT,
        state: Dict[SensorT, Any],
        metadata: RecordMetadata,
    ) -> None:
        """Call when producer finished sending message."""
        for sensor in self._sensors:
            sensor.on_send_completed(producer, state[sensor], metadata)

    def on_send_error(
        self,
        producer: ProducerT,
        exc: Exception,
        state: Dict[SensorT, Any],
    ) -> None:
        """Call when producer was unable to publish message."""
        for sensor in self._sensors:
            sensor.on_send_error(producer, exc, state[sensor])

    def on_assignment_start(
        self, assignor: PartitionAssignorT
    ) -> Dict[SensorT, Any]:
        """Call when partition assignor is starting to assign partitions."""
        return {sensor: sensor.on_assignment_start(assignor) for sensor in self._sensors}

    def on_assignment_error(
        self,
        assignor: PartitionAssignorT,
        state: Dict[SensorT, Any],
        exc: Exception,
    ) -> None:
        """Call when partition assignor did not complete assignor due to error."""
        for sensor in self._sensors:
            sensor.on_assignment_error(assignor, state[sensor], exc)

    def on_assignment_completed(
        self, assignor: PartitionAssignorT, state: Dict[SensorT, Any]
    ) -> None:
        """Call when partition assignor completed assignment."""
        for sensor in self._sensors:
            sensor.on_assignment_completed(assignor, state[sensor])

    def on_rebalance_start(self, app: AppT) -> Dict[SensorT, Any]:
        """Call when cluster rebalance is in progress."""
        return {sensor: sensor.on_rebalance_start(app) for sensor in self._sensors}

    def on_rebalance_return(
        self, app: AppT, state: Dict[SensorT, Any]
    ) -> None:
        """Call when consumer replied assignment is done to broker."""
        for sensor in self._sensors:
            sensor.on_rebalance_return(app, state[sensor])

    def on_rebalance_end(
        self, app: AppT, state: Dict[SensorT, Any]
    ) -> None:
        """Call when cluster rebalance fully completed (including recovery)."""
        for sensor in self._sensors:
            sensor.on_rebalance_end(app, state[sensor])

    def on_web_request_start(
        self,
        app: AppT,
        request: web.Request,
        *,
        view: Optional[web.View] = None,
    ) -> Dict[SensorT, Any]:
        """Call when web server started working on request."""
        return {sensor: sensor.on_web_request_start(app, request, view=view) for sensor in self._sensors}

    def on_web_request_end(
        self,
        app: AppT,
        request: web.Request,
        response: web.Response,
        state: Dict[SensorT, Any],
        *,
        view: Optional[web.View] = None,
    ) -> None:
        """Call when web server finished working on request."""
        for sensor in self._sensors:
            sensor.on_web_request_end(app, request, response, state[sensor], view=view)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self._sensors!r}>'
