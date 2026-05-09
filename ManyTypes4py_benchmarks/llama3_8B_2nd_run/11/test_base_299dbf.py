import pytest
from faust import web
from faust import Event, Stream, Table, Topic
from faust.assignor import PartitionAssignor
from faust.sensors import Sensor
from faust.transport.consumer import Consumer
from faust.transport.producer import Producer
from faust.types import Message, TP
from typing import Any, Dict

TP1: TP = TP('foo', 0)

@pytest.fixture
def message() -> Any:
    return Mock(name='message', autospec=Message)

@pytest.fixture
def stream() -> Any:
    return Mock(name='stream', autospec=Stream)

@pytest.fixture
def event() -> Any:
    return Mock(name='event', autospec=Event)

@pytest.fixture
def topic() -> Any:
    return Mock(name='topic', autospec=Topic)

@pytest.fixture
def table() -> Any:
    return Mock(name='table', autospec=Table)

@pytest.fixture
def consumer() -> Any:
    return Mock(name='consumer', autospec=Consumer)

@pytest.fixture
def producer() -> Any:
    return Mock(name='producer', autospec=Producer)

@pytest.fixture
def assignor() -> Any:
    return Mock(name='assignor', autospec=PartitionAssignor)

@pytest.fixture
def view() -> Any:
    return Mock(name='view', autospec=web.View)

@pytest.fixture
def req() -> Any:
    return Mock(name='request', autospec=web.Request)

@pytest.fixture
def response() -> Any:
    return Mock(name='response', autospec=web.Response)

class test_Sensor:
    @pytest.fixture
    def sensor(self, *, app: Any) -> Sensor:
        return Sensor()

    def test_on_message_in(self, *, sensor: Sensor, message: Any) -> None:
        sensor.on_message_in(TP1, 3, message)

    def test_on_stream_event_in(self, *, sensor: Sensor, stream: Any, event: Any) -> None:
        sensor.on_stream_event_in(TP1, 3, stream, event)

    # ... and so on
