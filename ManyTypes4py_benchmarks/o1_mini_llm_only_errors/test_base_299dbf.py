import pytest
from faust import web
from faust import Event, Stream, Table, Topic
from faust.assignor import PartitionAssignor
from faust.sensors import Sensor
from faust.transport.consumer import Consumer
from faust.transport.producer import Producer
from faust.types import Message, TP
from mode.utils.mocks import Mock

TP1 = TP('foo', 0)

@pytest.fixture
def message() -> Mock:
    return Mock(name='message', autospec=Message)

@pytest.fixture
def stream() -> Mock:
    return Mock(name='stream', autospec=Stream)

@pytest.fixture
def event() -> Mock:
    return Mock(name='event', autospec=Event)

@pytest.fixture
def topic() -> Mock:
    return Mock(name='topic', autospec=Topic)

@pytest.fixture
def table() -> Mock:
    return Mock(name='table', autospec=Table)

@pytest.fixture
def consumer() -> Mock:
    return Mock(name='consumer', autospec=Consumer)

@pytest.fixture
def producer() -> Mock:
    return Mock(name='producer', autospec=Producer)

@pytest.fixture
def assignor() -> Mock:
    return Mock(name='assignor', autospec=PartitionAssignor)

@pytest.fixture
def view() -> Mock:
    return Mock(name='view', autospec=web.View)

@pytest.fixture
def req() -> Mock:
    return Mock(name='request', autospec=web.Request)

@pytest.fixture
def response() -> Mock:
    return Mock(name='response', autospec=web.Response)

class test_Sensor:

    @pytest.fixture
    def sensor(self, *, app) -> Sensor:
        return Sensor()

    def test_on_message_in(self, *, sensor: Sensor, message: Mock) -> None:
        sensor.on_message_in(TP1, 3, message)

    def test_on_stream_event_in(self, *, sensor: Sensor, stream: Mock, event: Mock) -> None:
        sensor.on_stream_event_in(TP1, 3, stream, event)

    def test_on_stream_event_out(self, *, sensor: Sensor, stream: Mock, event: Mock) -> None:
        state = sensor.on_stream_event_in(TP1, 3, stream, event)
        sensor.on_stream_event_out(TP1, 3, stream, event, state)
        sensor.on_stream_event_out(TP1, 3, stream, event, None)

    def test_on_message_out(self, *, sensor: Sensor, message: Mock) -> None:
        sensor.on_message_out(TP1, 3, message)

    def test_on_topic_buffer_full(self, *, sensor: Sensor) -> None:
        sensor.on_topic_buffer_full(TP1)

    def test_on_table_get(self, *, sensor: Sensor, table: Mock) -> None:
        sensor.on_table_get(table, 'key')

    def test_on_table_set(self, *, sensor: Sensor, table: Mock) -> None:
        sensor.on_table_set(table, 'key', 'value')

    def test_on_table_del(self, *, sensor: Sensor, table: Mock) -> None:
        sensor.on_table_del(table, 'key')

    def test_on_commit_initiated(self, *, sensor: Sensor, consumer: Mock) -> None:
        sensor.on_commit_initiated(consumer)

    def test_on_commit_completed(self, *, sensor: Sensor, consumer: Mock) -> None:
        sensor.on_commit_completed(consumer, Mock(name='state'))

    def test_on_send_initiated(self, *, sensor: Sensor, producer: Mock) -> None:
        sensor.on_send_initiated(producer, 'topic', 'message', 30, 40)

    def test_on_send_completed(self, *, sensor: Sensor, producer: Mock) -> None:
        sensor.on_send_completed(producer, Mock(name='state'), Mock(name='metadata'))

    def test_on_assignment(self, *, sensor: Sensor, assignor: Mock) -> None:
        state = sensor.on_assignment_start(assignor)
        assert state['time_start']
        sensor.on_assignment_error(assignor, state, KeyError())
        sensor.on_assignment_completed(assignor, state)

    def test_on_rebalance(self, *, sensor: Sensor, app) -> None:
        state = sensor.on_rebalance_start(app)
        assert state['time_start']
        sensor.on_rebalance_return(app, state)
        sensor.on_rebalance_end(app, state)

    def test_on_web_request(self, *, sensor: Sensor, app, req: Mock, response: Mock, view: Mock) -> None:
        state = sensor.on_web_request_start(app, req, view=view)
        assert state['time_start']
        sensor.on_web_request_end(app, req, response, state, view=view)

    def test_on_send_error(self, *, sensor: Sensor, producer: Mock) -> None:
        sensor.on_send_error(producer, KeyError('foo'), Mock(name='state'))

    def test_asdict(self, *, sensor: Sensor) -> None:
        assert sensor.asdict() == {}

class test_SensorDelegate:

    @pytest.fixture
    def sensor(self) -> Mock:
        return Mock(name='sensor', autospec=Sensor)

    @pytest.fixture
    def sensors(self, *, app, sensor: Mock) -> Any:
        sensors = app.sensors
        sensors.add(sensor)
        return sensors

    def test_remove(self, *, sensors: Any, sensor: Mock) -> None:
        assert list(iter(sensors))
        sensors.remove(sensor)
        assert not list(iter(sensors))

    def test_on_message_in(self, *, sensors: Any, sensor: Mock, message: Mock) -> None:
        sensors.on_message_in(TP1, 303, message)
        sensor.on_message_in.assert_called_once_with(TP1, 303, message)

    def test_on_stream_event_in_out(self, *, sensors: Any, sensor: Mock, stream: Mock, event: Mock) -> None:
        state = sensors.on_stream_event_in(TP1, 303, stream, event)
        sensor.on_stream_event_in.assert_called_once_with(TP1, 303, stream, event)
        sensors.on_stream_event_out(TP1, 303, stream, event, state)
        sensor.on_stream_event_out.assert_called_once_with(TP1, 303, stream, event, state[sensor])

    def test_on_topic_buffer_full(self, *, sensors: Any, sensor: Mock) -> None:
        sensors.on_topic_buffer_full(TP1)
        sensor.on_topic_buffer_full.assert_called_once_with(TP1)

    def test_on_message_out(self, *, sensors: Any, sensor: Mock, message: Mock) -> None:
        sensors.on_message_out(TP1, 303, message)
        sensor.on_message_out.assert_called_once_with(TP1, 303, message)

    def test_on_table_get(self, *, sensors: Any, sensor: Mock, table: Mock) -> None:
        sensors.on_table_get(table, 'key')
        sensor.on_table_get.assert_called_once_with(table, 'key')

    def test_on_table_set(self, *, sensors: Any, sensor: Mock, table: Mock) -> None:
        sensors.on_table_set(table, 'key', 'value')
        sensor.on_table_set.assert_called_once_with(table, 'key', 'value')

    def test_on_table_del(self, *, sensors: Any, sensor: Mock, table: Mock) -> None:
        sensors.on_table_del(table, 'key')
        sensor.on_table_del.assert_called_once_with(table, 'key')

    def test_on_commit(self, *, sensors: Any, sensor: Mock, consumer: Mock) -> None:
        state = sensors.on_commit_initiated(consumer)
        sensor.on_commit_initiated.assert_called_once_with(consumer)
        sensors.on_commit_completed(consumer, state)
        sensor.on_commit_completed.assert_called_once_with(consumer, state[sensor])

    def test_on_send(self, *, sensors: Any, sensor: Mock, producer: Mock) -> None:
        metadata = Mock(name='metadata')
        state = sensors.on_send_initiated(producer, 'topic', 'message', 303, 606)
        sensor.on_send_initiated.assert_called_once_with(producer, 'topic', 'message', 303, 606)
        sensors.on_send_completed(producer, state, metadata)
        sensor.on_send_completed.assert_called_once_with(producer, state[sensor], metadata)
        exc = KeyError('foo')
        sensors.on_send_error(producer, exc, state)
        sensor.on_send_error.assert_called_once_with(producer, exc, state[sensor])

    def test_on_assignment(self, *, sensors: Any, sensor: Mock, assignor: Mock) -> None:
        state = sensors.on_assignment_start(assignor)
        sensor.on_assignment_start.assert_called_once_with(assignor)
        sensors.on_assignment_completed(assignor, state)
        sensor.on_assignment_completed.assert_called_once_with(assignor, state[sensor])
        exc = KeyError('bar')
        sensors.on_assignment_error(assignor, state, exc)
        sensor.on_assignment_error.assert_called_once_with(assignor, state[sensor], exc)

    def test_on_rebalance(self, *, sensors: Any, sensor: Mock, app) -> None:
        state = sensors.on_rebalance_start(app)
        sensor.on_rebalance_start.assert_called_once_with(app)
        sensors.on_rebalance_return(app, state)
        sensor.on_rebalance_return.assert_called_once_with(app, state[sensor])
        sensors.on_rebalance_end(app, state)
        sensor.on_rebalance_end.assert_called_once_with(app, state[sensor])

    def test_on_web_request(self, *, sensors: Any, sensor: Mock, app, req: Mock, response: Mock, view: Mock) -> None:
        state = sensors.on_web_request_start(app, req, view=view)
        sensor.on_web_request_start.assert_called_once_with(app, req, view=view)
        sensors.on_web_request_end(app, req, response, state, view=view)
        sensor.on_web_request_end.assert_called_once_with(app, req, response, state[sensor], view=view)

    def test_repr(self, *, sensors: Any) -> None:
        assert repr(sensors)
