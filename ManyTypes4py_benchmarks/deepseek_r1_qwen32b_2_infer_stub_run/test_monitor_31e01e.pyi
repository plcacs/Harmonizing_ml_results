from collections import deque
from http import HTTPStatus
from statistics import median
from typing import Any, Dict, List, Optional, Union
import pytest
from faust import Event, Stream, Table, Topic
from faust.transport.consumer import Consumer
from faust.transport.producer import Producer
from faust.types import Message, TP
from faust.sensors.monitor import Monitor, TableState
from mode.utils.mocks import AsyncMock, Mock

TP1 = TP('foo', 0)

class test_Monitor:
    @pytest.fixture
    def time(self) -> Mock:
        ...

    @pytest.fixture
    def message(self) -> Mock:
        ...

    @pytest.fixture
    def stream(self) -> Mock:
        ...

    @pytest.fixture
    def topic(self) -> Mock:
        ...

    @pytest.fixture
    def event(self) -> Mock:
        ...

    @pytest.fixture
    def table(self) -> Mock:
        ...

    @pytest.fixture
    def mon(self, *, time: Mock) -> Monitor:
        ...

    def create_monitor(self, **kwargs: Any) -> Monitor:
        ...

    def create_populated_monitor(
        self,
        messages_active: int = 101,
        messages_received_total: int = 1001,
        messages_sent: int = 303,
        messages_s: int = 1000,
        messages_received_by_topic: Dict[str, int] = {'foo': 103},
        events_active: int = 202,
        events_total: int = 2002,
        events_s: int = 3000,
        events_runtime_avg: float = 0.03,
        events_by_task: Dict[str, int] = {'mytask': 105},
        events_by_stream: Dict[str, int] = {'stream': 105},
        commit_latency: List[float] = [1.03, 2.33, 16.33],
        send_latency: List[float] = [0.01, 0.04, 0.06, 0.01],
        topic_buffer_full: Dict[str, int] = {'topic': 808},
        **kwargs: Any
    ) -> Monitor:
        ...

    def test_init_max_avg_history(self) -> None:
        ...

    def test_init_max_avg_history__default(self) -> None:
        ...

    def test_init_max_commit_latency_history(self) -> None:
        ...

    def test_init_max_commit_latency_history__default(self) -> None:
        ...

    def test_init_max_send_latency_history(self) -> None:
        ...

    def test_init_max_send_latency_history__default(self) -> None:
        ...

    def test_init_max_assignment_latency_history(self) -> None:
        ...

    def test_init_max_assignment_latency_history__default(self) -> None:
        ...

    def test_init_rebalances(self) -> None:
        ...

    def test_asdict(self) -> dict:
        ...

    def test_on_message_in(self, *, message: Message, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_stream_event_in(self, *, event: Event, mon: Monitor, stream: Stream, time: Mock) -> None:
        ...

    def test_on_stream_event_out(self, *, event: Event, mon: Monitor, stream: Stream, time: Mock) -> None:
        ...

    def test_on_stream_event_out__missing_state(self, *, event: Event, mon: Monitor, stream: Stream, time: Mock) -> None:
        ...

    def test_on_topic_buffer_full(self, *, mon: Monitor) -> None:
        ...

    def test_on_message_out(self, *, message: Message, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_table_get(self, *, mon: Monitor, table: Table) -> None:
        ...

    def test_on_table_set(self, *, mon: Monitor, table: Table) -> None:
        ...

    def test_on_table_del(self, *, mon: Monitor, table: Table) -> None:
        ...

    def test_on_commit_initiated(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_commit_completed(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_send_initiated(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_send_completed(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_send_error(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_assignment_start(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_assignment_completed(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_assignment_error(self, *, mon: Monitor, time: Mock) -> None:
        ...

    def test_on_rebalance_start(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def test_on_rebalance_return(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def test_on_rebalance_end(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def test_on_web_request_start(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def test_on_web_request_end(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def test_on_web_request_end__None_response(self, *, mon: Monitor, time: Mock, app: Any) -> None:
        ...

    def assert_on_web_request_end(
        self,
        mon: Monitor,
        time: Mock,
        app: Any,
        response: Optional[Any],
        expected_status: int
    ) -> None:
        ...

    def test_TableState_asdict(self, *, mon: Monitor, table: Table) -> None:
        ...

    def test_on_tp_commit(self, *, mon: Monitor) -> None:
        ...

    def test_track_tp_end_offsets(self, *, mon: Monitor) -> None:
        ...

    @pytest.mark.asyncio
    async def test_service_sampler(self, *, mon: Monitor) -> None:
        ...

    def test__sample(self, *, mon: Monitor) -> None:
        ...