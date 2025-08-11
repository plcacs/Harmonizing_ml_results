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
    def sensor(self, *, app) -> Mock:
        return Sensor()

    def test_on_message_in(self, *, sensor: Union[can.message.Message, bytes], message: Union[telethon.tl.custom.Message, utils.pluginmgr.Command]) -> None:
        sensor.on_message_in(TP1, 3, message)

    def test_on_stream_event_in(self, *, sensor: typing.Callable, stream: typing.Callable, event: typing.Callable) -> None:
        sensor.on_stream_event_in(TP1, 3, stream, event)

    def test_on_stream_event_out(self, *, sensor: faustypes.assignor.PartitionAssignorT, stream: faustypes.assignor.PartitionAssignorT, event: faustypes.assignor.PartitionAssignorT) -> None:
        state = sensor.on_stream_event_in(TP1, 3, stream, event)
        sensor.on_stream_event_out(TP1, 3, stream, event, state)
        sensor.on_stream_event_out(TP1, 3, stream, event, None)

    def test_on_message_out(self, *, sensor: can.message.Message, message: Union[telethon.tl.custom.Message, utils.pluginmgr.Command]) -> None:
        sensor.on_message_out(TP1, 3, message)

    def test_on_topic_buffer_full(self, *, sensor: Union[set, dict]) -> None:
        sensor.on_topic_buffer_full(TP1)

    def test_on_table_get(self, *, sensor: str, table: str) -> None:
        sensor.on_table_get(table, 'key')

    def test_on_table_set(self, *, sensor: Union[str, sa.Table], table: str) -> None:
        sensor.on_table_set(table, 'key', 'value')

    def test_on_table_del(self, *, sensor: str, table: str) -> None:
        sensor.on_table_del(table, 'key')

    def test_on_commit_initiated(self, *, sensor: Any, consumer: Any) -> None:
        sensor.on_commit_initiated(consumer)

    def test_on_commit_completed(self, *, sensor: unittesmock.Mock, consumer: unittesmock.Mock) -> None:
        sensor.on_commit_completed(consumer, Mock(name='state'))

    def test_on_send_initiated(self, *, sensor: Any, producer: Any) -> None:
        sensor.on_send_initiated(producer, 'topic', 'message', 30, 40)

    def test_on_send_completed(self, *, sensor: Any, producer: Any) -> None:
        sensor.on_send_completed(producer, Mock(name='state'), Mock(name='metadata'))

    def test_on_assignment(self, *, sensor: Any, assignor: set) -> None:
        state = sensor.on_assignment_start(assignor)
        assert state['time_start']
        sensor.on_assignment_error(assignor, state, KeyError())
        sensor.on_assignment_completed(assignor, state)

    def test_on_rebalance(self, *, sensor: Any, app: Any) -> None:
        state = sensor.on_rebalance_start(app)
        assert state['time_start']
        sensor.on_rebalance_return(app, state)
        sensor.on_rebalance_end(app, state)

    def test_on_web_request(self, *, sensor: Union[dict, tests.utils.FakeRequest], app: Any, req: Any, response: dict, view: Any) -> None:
        state = sensor.on_web_request_start(app, req, view=view)
        assert state['time_start']
        sensor.on_web_request_end(app, req, response, state, view=view)

    def test_on_send_error(self, *, sensor: Any, producer: Any) -> None:
        sensor.on_send_error(producer, KeyError('foo'), Mock(name='state'))

    def test_asdict(self, *, sensor: Union[dict, str]) -> None:
        assert sensor.asdict() == {}

class test_SensorDelegate:

    @pytest.fixture
    def sensor(self) -> Mock:
        return Mock(name='sensor', autospec=Sensor)

    @pytest.fixture
    def sensors(self, *, app: aiohttp.web.Application, sensor: Union[list[dict], asyncworker.types.registry.TypesRegistry, typing.Callable[str, None]]):
        sensors = app.sensors
        sensors.add(sensor)
        return sensors

    def test_remove(self, *, sensors: Union[set, list[dict], types.ServiceT], sensor: Union[set, list, str]) -> None:
        assert list(iter(sensors))
        sensors.remove(sensor)
        assert not list(iter(sensors))

    def test_on_message_in(self, *, sensors: Union[bytes, can.message.Message, utils.pluginmgr.Command], sensor: Union[can.message.Message, bytes], message: Union[telethon.tl.custom.Message, utils.pluginmgr.Command]) -> None:
        sensors.on_message_in(TP1, 303, message)
        sensor.on_message_in.assert_called_once_with(TP1, 303, message)

    def test_on_stream_event_in_out(self, *, sensors: dict, sensor: Any, stream: dict, event: dict) -> None:
        state = sensors.on_stream_event_in(TP1, 303, stream, event)
        sensor.on_stream_event_in.assert_called_once_with(TP1, 303, stream, event)
        sensors.on_stream_event_out(TP1, 303, stream, event, state)
        sensor.on_stream_event_out.assert_called_once_with(TP1, 303, stream, event, state[sensor])

    def test_on_topic_buffer_full(self, *, sensors: Union[dict, bool], sensor: Union[set, dict]) -> None:
        sensors.on_topic_buffer_full(TP1)
        sensor.on_topic_buffer_full.assert_called_once_with(TP1)

    def test_on_message_out(self, *, sensors: utils.pluginmgr.Command, sensor: can.message.Message, message: Union[telethon.tl.custom.Message, utils.pluginmgr.Command]) -> None:
        sensors.on_message_out(TP1, 303, message)
        sensor.on_message_out.assert_called_once_with(TP1, 303, message)

    def test_on_table_get(self, *, sensors: Union[str, dict], sensor: str, table: str) -> None:
        sensors.on_table_get(table, 'key')
        sensor.on_table_get.assert_called_once_with(table, 'key')

    def test_on_table_set(self, *, sensors: str, sensor: Union[str, sa.Table], table: str) -> None:
        sensors.on_table_set(table, 'key', 'value')
        sensor.on_table_set.assert_called_once_with(table, 'key', 'value')

    def test_on_table_del(self, *, sensors: str, sensor: str, table: str) -> None:
        sensors.on_table_del(table, 'key')
        sensor.on_table_del.assert_called_once_with(table, 'key')

    def test_on_commit(self, *, sensors: Any, sensor: typing.Iterable, consumer: Any) -> None:
        state = sensors.on_commit_initiated(consumer)
        sensor.on_commit_initiated.assert_called_once_with(consumer)
        sensors.on_commit_completed(consumer, state)
        sensor.on_commit_completed.assert_called_once_with(consumer, state[sensor])

    def test_on_send(self, *, sensors: Union[dict, tonga.services.producer.base.BaseProducer], sensor: Any, producer: Any) -> None:
        metadata = Mock(name='metadata')
        state = sensors.on_send_initiated(producer, 'topic', 'message', 303, 606)
        sensor.on_send_initiated.assert_called_once_with(producer, 'topic', 'message', 303, 606)
        sensors.on_send_completed(producer, state, metadata)
        sensor.on_send_completed.assert_called_once_with(producer, state[sensor], metadata)
        exc = KeyError('foo')
        sensors.on_send_error(producer, exc, state)
        sensor.on_send_error.assert_called_once_with(producer, exc, state[sensor])

    def test_on_assignment(self, *, sensors: str, sensor: Any, assignor: set) -> None:
        state = sensors.on_assignment_start(assignor)
        sensor.on_assignment_start.assert_called_once_with(assignor)
        sensors.on_assignment_completed(assignor, state)
        sensor.on_assignment_completed.assert_called_once_with(assignor, state[sensor])
        exc = KeyError('bar')
        sensors.on_assignment_error(assignor, state, exc)
        sensor.on_assignment_error.assert_called_once_with(assignor, state[sensor], exc)

    def test_on_rebalance(self, *, sensors: Any, sensor: Any, app: Any) -> None:
        state = sensors.on_rebalance_start(app)
        sensor.on_rebalance_start.assert_called_once_with(app)
        sensors.on_rebalance_return(app, state)
        sensor.on_rebalance_return.assert_called_once_with(app, state[sensor])
        sensors.on_rebalance_end(app, state)
        sensor.on_rebalance_end.assert_called_once_with(app, state[sensor])

    def test_on_web_request(self, *, sensors: dict, sensor: Union[dict, tests.utils.FakeRequest], app: Any, req: Any, response: dict, view: Any) -> None:
        state = sensors.on_web_request_start(app, req, view=view)
        sensor.on_web_request_start.assert_called_once_with(app, req, view=view)
        sensors.on_web_request_end(app, req, response, state, view=view)
        sensor.on_web_request_end.assert_called_once_with(app, req, response, state[sensor], view=view)

    def test_repr(self, *, sensors: Any) -> None:
        assert repr(sensors)