"""Base-interface for sensors."""
from time import monotonic
from typing import Any, Dict, Iterator, Mapping, Optional, Set
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

    def on_message_in(self, tp: Union[int, faustypes.tuples.Message, faustypes.tuples.TP], offset: Union[int, faustypes.tuples.Message, faustypes.tuples.TP], message: Union[int, faustypes.tuples.Message, faustypes.tuples.TP]) -> None:
        """Message received by a consumer."""
        ...

    def on_stream_event_in(self, tp: Union[int, faustypes.EventT, faustypes.tuples.TP], offset: Union[int, faustypes.EventT, faustypes.tuples.TP], stream: Union[int, faustypes.EventT, faustypes.tuples.TP], event: Union[int, faustypes.EventT, faustypes.tuples.TP]) -> dict:
        """Message sent to a stream as an event."""
        return None

    def on_stream_event_out(self, tp: Union[faustypes.StreamT, int, faustypes.tuples.TP], offset: Union[faustypes.StreamT, int, faustypes.tuples.TP], stream: Union[faustypes.StreamT, int, faustypes.tuples.TP], event: Union[faustypes.StreamT, int, faustypes.tuples.TP], state: Union[None, faustypes.StreamT, faustypes.tuples.TP, int]=None) -> None:
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

    def on_message_out(self, tp: Union[int, faustypes.tuples.TP, faustypes.tuples.Message], offset: Union[int, faustypes.tuples.TP, faustypes.tuples.Message], message: Union[int, faustypes.tuples.TP, faustypes.tuples.Message]) -> None:
        """All streams finished processing message."""
        ...

    def on_topic_buffer_full(self, tp: Union[faustypes.tuples.TP, int, faustypes.tuples.FutureMessage]) -> None:
        """Topic buffer full so conductor had to wait."""
        ...

    def on_table_get(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Key retrieved from table."""
        ...

    def on_table_set(self, table: faustypes.CollectionT, key: faustypes.CollectionT, value: faustypes.CollectionT) -> None:
        """Value set for key in table."""
        ...

    def on_table_del(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Key deleted from table."""
        ...

    def on_commit_initiated(self, consumer: Union[faustypes.transports.ConsumerT, faustypes.transports.ProducerT]) -> dict:
        """Consumer is about to commit topic offset."""
        ...

    def on_commit_completed(self, consumer: faustypes.transports.ConsumerT, state: faustypes.transports.ConsumerT) -> None:
        """Consumer finished committing topic offset."""
        ...

    def on_send_initiated(self, producer: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], topic: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], message: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], keysize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], valsize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage]) -> dict:
        """About to send a message."""
        ...

    def on_send_completed(self, producer: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata], state: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata], metadata: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata]) -> None:
        """Message successfully sent."""
        ...

    def on_send_error(self, producer: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], exc: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], state: Union[BaseException, faustypes.transports.ProducerT, typing.Callable]) -> None:
        """Error while sending message."""
        ...

    def on_assignment_start(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict]) -> dict:
        """Partition assignor is starting to assign partitions."""
        return {'time_start': monotonic()}

    def on_assignment_error(self, assignor: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], state: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], exc: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict]) -> None:
        """Partition assignor did not complete assignor due to error."""
        ...

    def on_assignment_completed(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT], state: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT]) -> None:
        """Partition assignor completed assignment."""
        ...

    def on_rebalance_start(self, app: Union[faustypes.AppT, web.Application]) -> dict:
        """Cluster rebalance in progress."""
        return {'time_start': monotonic()}

    def on_rebalance_return(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        """Consumer replied assignment is done to broker."""
        ...

    def on_rebalance_end(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        ...

    def on_web_request_start(self, app: faustypes.AppT, request: faustypes.AppT, *, view: Union[None, faustypes.AppT]=None) -> dict:
        """Web server started working on request."""
        return {'time_start': monotonic()}

    def on_web_request_end(self, app: Union[faustypes.AppT, dict], request: Union[faustypes.AppT, dict], response: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict], *, view: Union[None, faustypes.AppT, dict]=None) -> None:
        """Web server finished working on request."""
        ...

    def asdict(self) -> dict:
        """Convert sensor state to dictionary."""
        return {}

class SensorDelegate(SensorDelegateT):
    """A class that delegates sensor methods to a list of sensors."""

    def __init__(self, app: Union[faustypes.AppT, starlette.types.ASGIApp, typing.Any, None]) -> None:
        self.app = app
        self._sensors = set()

    def add(self, sensor: Union[prefecengine.state.State, dict, list[typing.Mapping]]) -> None:
        """Add sensor."""
        sensor.beacon = self.app.beacon.new(sensor)
        self._sensors.add(sensor)

    def remove(self, sensor: Union[list[services.Service], list[T], typing.Callable]) -> None:
        """Remove sensor."""
        self._sensors.remove(sensor)

    def __iter__(self):
        return iter(self._sensors)

    def on_message_in(self, tp: Union[int, faustypes.tuples.Message, faustypes.tuples.TP], offset: Union[int, faustypes.tuples.Message, faustypes.tuples.TP], message: Union[int, faustypes.tuples.Message, faustypes.tuples.TP]) -> None:
        """Call before message is delegated to streams."""
        for sensor in self._sensors:
            sensor.on_message_in(tp, offset, message)

    def on_stream_event_in(self, tp: Union[int, faustypes.EventT, faustypes.tuples.TP], offset: Union[int, faustypes.EventT, faustypes.tuples.TP], stream: Union[int, faustypes.EventT, faustypes.tuples.TP], event: Union[int, faustypes.EventT, faustypes.tuples.TP]) -> dict:
        """Call when stream starts processing an event."""
        return {sensor: sensor.on_stream_event_in(tp, offset, stream, event) for sensor in self._sensors}

    def on_stream_event_out(self, tp: Union[faustypes.StreamT, int, faustypes.tuples.TP], offset: Union[faustypes.StreamT, int, faustypes.tuples.TP], stream: Union[faustypes.StreamT, int, faustypes.tuples.TP], event: Union[faustypes.StreamT, int, faustypes.tuples.TP], state: Union[None, faustypes.StreamT, faustypes.tuples.TP, int]=None) -> None:
        """Call when stream is done processing an event."""
        sensor_state = state or {}
        for sensor in self._sensors:
            sensor.on_stream_event_out(tp, offset, stream, event, sensor_state.get(sensor))

    def on_topic_buffer_full(self, tp: Union[faustypes.tuples.TP, int, faustypes.tuples.FutureMessage]) -> None:
        """Call when conductor topic buffer is full and has to wait."""
        for sensor in self._sensors:
            sensor.on_topic_buffer_full(tp)

    def on_message_out(self, tp: Union[int, faustypes.tuples.TP, faustypes.tuples.Message], offset: Union[int, faustypes.tuples.TP, faustypes.tuples.Message], message: Union[int, faustypes.tuples.TP, faustypes.tuples.Message]) -> None:
        """Call when message is fully acknowledged and can be committed."""
        for sensor in self._sensors:
            sensor.on_message_out(tp, offset, message)

    def on_table_get(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when value in table is retrieved."""
        for sensor in self._sensors:
            sensor.on_table_get(table, key)

    def on_table_set(self, table: faustypes.CollectionT, key: faustypes.CollectionT, value: faustypes.CollectionT) -> None:
        """Call when new value for key in table is set."""
        for sensor in self._sensors:
            sensor.on_table_set(table, key, value)

    def on_table_del(self, table: faustypes.CollectionT, key: faustypes.CollectionT) -> None:
        """Call when key in a table is deleted."""
        for sensor in self._sensors:
            sensor.on_table_del(table, key)

    def on_commit_initiated(self, consumer: Union[faustypes.transports.ConsumerT, faustypes.transports.ProducerT]) -> dict:
        """Call when consumer commit offset operation starts."""
        return {sensor: sensor.on_commit_initiated(consumer) for sensor in self._sensors}

    def on_commit_completed(self, consumer: faustypes.transports.ConsumerT, state: faustypes.transports.ConsumerT) -> None:
        """Call when consumer commit offset operation completed."""
        for sensor in self._sensors:
            sensor.on_commit_completed(consumer, state[sensor])

    def on_send_initiated(self, producer: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], topic: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], message: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], keysize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage], valsize: Union[int, faustypes.transports.ProducerT, faustypes.tuples.PendingMessage]) -> dict:
        """Call when message added to producer buffer."""
        return {sensor: sensor.on_send_initiated(producer, topic, message, keysize, valsize) for sensor in self._sensors}

    def on_send_completed(self, producer: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata], state: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata], metadata: Union[faustypes.transports.ProducerT, faustypes.tuples.RecordMetadata]) -> None:
        """Call when producer finished sending message."""
        for sensor in self._sensors:
            sensor.on_send_completed(producer, state[sensor], metadata)

    def on_send_error(self, producer: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], exc: Union[BaseException, faustypes.transports.ProducerT, typing.Callable], state: Union[BaseException, faustypes.transports.ProducerT, typing.Callable]) -> None:
        """Call when producer was unable to publish message."""
        for sensor in self._sensors:
            sensor.on_send_error(producer, exc, state[sensor])

    def on_assignment_start(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict]) -> dict:
        """Partition assignor is starting to assign partitions."""
        return {sensor: sensor.on_assignment_start(assignor) for sensor in self._sensors}

    def on_assignment_error(self, assignor: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], state: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict], exc: Union[BaseException, faustypes.assignor.PartitionAssignorT, dict]) -> None:
        """Partition assignor did not complete assignor due to error."""
        for sensor in self._sensors:
            sensor.on_assignment_error(assignor, state[sensor], exc)

    def on_assignment_completed(self, assignor: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT], state: Union[faustypes.assignor.PartitionAssignorT, dict, faustypes.transports.ProducerT]) -> None:
        """Partition assignor completed assignment."""
        for sensor in self._sensors:
            sensor.on_assignment_completed(assignor, state[sensor])

    def on_rebalance_start(self, app: Union[faustypes.AppT, web.Application]) -> dict:
        """Cluster rebalance in progress."""
        return {sensor: sensor.on_rebalance_start(app) for sensor in self._sensors}

    def on_rebalance_return(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        """Consumer replied assignment is done to broker."""
        for sensor in self._sensors:
            sensor.on_rebalance_return(app, state[sensor])

    def on_rebalance_end(self, app: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict]) -> None:
        """Cluster rebalance fully completed (including recovery)."""
        for sensor in self._sensors:
            sensor.on_rebalance_end(app, state[sensor])

    def on_web_request_start(self, app: faustypes.AppT, request: faustypes.AppT, *, view: Union[None, faustypes.AppT]=None) -> dict:
        """Web server started working on request."""
        return {sensor: sensor.on_web_request_start(app, request, view=view) for sensor in self._sensors}

    def on_web_request_end(self, app: Union[faustypes.AppT, dict], request: Union[faustypes.AppT, dict], response: Union[faustypes.AppT, dict], state: Union[faustypes.AppT, dict], *, view: Union[None, faustypes.AppT, dict]=None) -> None:
        """Web server finished working on request."""
        for sensor in self._sensors:
            sensor.on_web_request_end(app, request, response, state[sensor], view=view)

    def __repr__(self) -> typing.Text:
        return f'<{type(self).__name__}: {self._sensors!r}>'