import asyncio
from collections import deque
from http import HTTPStatus
from statistics import median
from typing import Any, Callable, Dict, Deque, Generator, Tuple, Optional

import pytest
from faust import Event, Stream, Table, Topic
from faust.transport.consumer import Consumer
from faust.transport.producer import Producer
from faust.types import Message, TP
from faust.sensors.monitor import Monitor, TableState
from mode.utils.mocks import AsyncMock, Mock

TP1: TP = TP('foo', 0)


class test_Monitor:

    @pytest.fixture
    def time(self) -> Callable[[], float]:
        timefun = Mock(name='time()')
        timefun.return_value = 101.1
        return timefun

    @pytest.fixture
    def message(self) -> Message:
        return Mock(name='message', autospec=Message)

    @pytest.fixture
    def stream(self) -> Stream:
        return Mock(name='stream', autospec=Stream)

    @pytest.fixture
    def topic(self) -> Topic:
        return Mock(name='topic', autospec=Topic)

    @pytest.fixture
    def event(self) -> Event:
        return Mock(name='event', autospec=Event)

    @pytest.fixture
    def table(self) -> Table:
        return Mock(name='table', autospec=Table)

    @pytest.fixture
    def mon(self, *, time: Callable[[], float]) -> Monitor:
        mon = self.create_monitor()
        mon.time = time
        return mon

    def create_monitor(self, **kwargs: Any) -> Monitor:
        return Monitor(**kwargs)

    def create_populated_monitor(
            self,
            messages_active: int = 101,
            messages_received_total: int = 1001,
            messages_sent: int = 303,
            messages_s: int = 1000,
            messages_received_by_topic: Dict[str, int] = {'foo': 103},  # noqa
            events_active: int = 202,
            events_total: int = 2002,
            events_s: int = 3000,
            events_runtime_avg: float = 0.03,
            events_by_task: Dict[str, int] = {'mytask': 105},  # noqa
            events_by_stream: Dict[str, int] = {'stream': 105},  # noqa
            commit_latency: Any = [1.03, 2.33, 16.33],  # noqa
            send_latency: Any = [0.01, 0.04, 0.06, 0.010],  # noqa
            topic_buffer_full: Dict[str, int] = {'topic': 808},  # noqa
            **kwargs: Any
    ) -> Monitor:
        return self.create_monitor(
            messages_active=messages_active,
            messages_received_total=messages_received_total,
            messages_sent=messages_sent,
            messages_s=messages_s,
            messages_received_by_topic=messages_received_by_topic,
            events_active=events_active,
            events_total=events_total,
            events_s=events_s,
            events_runtime_avg=events_runtime_avg,
            events_by_task=events_by_task,
            events_by_stream=events_by_stream,
            commit_latency=commit_latency,
            send_latency=send_latency,
            topic_buffer_full=topic_buffer_full,
            **kwargs
        )

    def test_init_max_avg_history(self) -> None:
        assert Monitor().max_avg_history == Monitor.max_avg_history

    def test_init_max_avg_history__default(self) -> None:
        assert Monitor(max_avg_history=33).max_avg_history == 33

    def test_init_max_commit_latency_history(self) -> None:
        assert (Monitor().max_commit_latency_history ==
                Monitor.max_commit_latency_history)

    def test_init_max_commit_latency_history__default(self) -> None:
        assert Monitor(
            max_commit_latency_history=33,
        ).max_commit_latency_history == 33

    def test_init_max_send_latency_history(self) -> None:
        assert (Monitor().max_send_latency_history ==
                Monitor.max_send_latency_history)

    def test_init_max_send_latency_history__default(self) -> None:
        assert Monitor(
            max_send_latency_history=33,
        ).max_send_latency_history == 33

    def test_init_max_assignment_latency_history(self) -> None:
        assert (Monitor().max_assignment_latency_history ==
                Monitor.max_assignment_latency_history)

    def test_init_max_assignment_latency_history__default(self) -> None:
        assert Monitor(
            max_assignment_latency_history=33,
        ).max_assignment_latency_history == 33

    def test_init_rebalances(self) -> None:
        assert Monitor(rebalances=99).rebalances == 99

    def test_asdict(self, *, mon: Monitor) -> None:
        asdict_result: Dict[str, Any] = mon.asdict()
        assert asdict_result == {
            'messages_active': mon.messages_active,
            'messages_received_total': mon.messages_received_total,
            'messages_sent': mon.messages_sent,
            'messages_sent_by_topic': mon.messages_sent_by_topic,
            'messages_s': mon.messages_s,
            'messages_received_by_topic': mon.messages_received_by_topic,
            'events_active': mon.events_active,
            'events_total': mon.events_total,
            'events_s': mon.events_s,
            'events_runtime_avg': mon.events_runtime_avg,
            'events_by_task': mon._events_by_task_dict(),
            'events_by_stream': mon._events_by_stream_dict(),
            'commit_latency': mon.commit_latency,
            'send_latency': mon.send_latency,
            'assignment_latency': mon.assignment_latency,
            'assignments_completed': mon.assignments_completed,
            'assignments_failed': mon.assignments_failed,
            'send_errors': mon.send_errors,
            'topic_buffer_full': mon._topic_buffer_full_dict(),
            'metric_counts': mon._metric_counts_dict(),
            'tables': {
                name: table.asdict() for name, table in mon.tables.items()
            },
            'topic_committed_offsets': {},
            'topic_read_offsets': {},
            'topic_end_offsets': {},
            'rebalance_end_avg': mon.rebalance_end_avg,
            'rebalance_end_latency': mon.rebalance_end_latency,
            'rebalance_return_avg': mon.rebalance_return_avg,
            'rebalance_return_latency': mon.rebalance_return_latency,
            'rebalances': mon.rebalances,
            'http_response_codes': mon._http_response_codes_dict(),
            'http_response_latency': mon.http_response_latency,
            'http_response_latency_avg': mon.http_response_latency_avg,
        }

    def test_on_message_in(self, *, message: Message, mon: Monitor, time: Callable[[], float]) -> None:
        for i in range(1, 11):
            offset: int = 3 + i
            mon.on_message_in(TP1, offset, message)

            assert mon.messages_received_total == i
            assert mon.messages_active == i
            assert mon.messages_received_by_topic[TP1.topic] == i
            assert message.time_in is time()
            assert mon.tp_read_offsets[TP1] == offset

    def test_on_stream_event_in(self, *, event: Event, mon: Monitor, stream: Stream, time: Callable[[], float]) -> None:
        for i in range(1, 11):
            state: Dict[str, Optional[float]] = mon.on_stream_event_in(TP1, 3 + i, stream, event)

            assert mon.events_total == i
            assert mon.events_by_stream[str(stream)] == i
            assert mon.events_by_task[str(stream.task_owner)] == i
            assert mon.events_active == i
            assert state == {
                'time_in': time(),
                'time_out': None,
                'time_total': None,
            }

    def test_on_stream_event_out(self, *, event: Event, mon: Monitor, stream: Stream, time: Callable[[], float]) -> None:
        other_time: float = 303.3
        mon.events_active = 10
        for i in range(1, 11):
            state: Dict[str, Optional[float]] = {
                'time_in': other_time,
                'time_out': None,
                'time_total': None,
            }
            mon.on_stream_event_out(TP1, 3 + i, stream, event, state)

            assert mon.events_active == 10 - i
            assert state == {
                'time_in': other_time,
                'time_out': time(),
                'time_total': time() - other_time,
            }
            assert mon.events_runtime[-1] == time() - other_time

    def test_on_stream_event_out__missing_state(self, *, event: Event, mon: Monitor, stream: Stream, time: Callable[[], float]) -> None:
        # should not be an error
        mon.on_stream_event_out(TP1, 3, stream, event, None)

    def test_on_topic_buffer_full(self, *, mon: Monitor) -> None:
        for i in range(1, 11):
            mon.on_topic_buffer_full(TP1)
            assert mon.topic_buffer_full[TP1] == i

    def test_on_message_out(self, *, message: Message, mon: Monitor, time: Callable[[], float]) -> None:
        mon.messages_active = 10
        message.time_in = 10.7

        for i in range(1, 11):
            mon.on_message_out(TP1, 3 + i, message)
            assert mon.messages_active == 10 - i
            assert message.time_out == time()
            assert message.time_total == time() - message.time_in
        message.time_in = None
        mon.on_message_out(TP1, 3 + 11, message)

    def test_on_table_get(self, *, mon: Monitor, table: Table) -> None:
        for i in range(1, 11):
            mon.on_table_get(table, 'k')
            assert mon._table_or_create(table).keys_retrieved == i

    def test_on_table_set(self, *, mon: Monitor, table: Table) -> None:
        for i in range(1, 11):
            mon.on_table_set(table, 'k', 'v')
            assert mon._table_or_create(table).keys_updated == i

    def test_on_table_del(self, *, mon: Monitor, table: Table) -> None:
        for i in range(1, 11):
            mon.on_table_del(table, 'k')
            assert mon._table_or_create(table).keys_deleted == i

    def test_on_commit_initiated(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        consumer: Consumer = Mock(name='consumer', autospec=Consumer)
        result: float = mon.on_commit_initiated(consumer)
        assert result == time()

    def test_on_commit_completed(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        other_time: float = 56.7
        consumer: Consumer = Mock(name='consumer', autospec=Consumer)
        mon.on_commit_completed(consumer, other_time)
        assert mon.commit_latency[-1] == time() - other_time

    def test_on_send_initiated(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        for i in range(1, 11):
            producer: Producer = Mock(name='producer', autospec=Producer)
            state: float = mon.on_send_initiated(producer, 'topic', 'message', 2, 4)
            assert mon.messages_sent == i
            assert mon.messages_sent_by_topic['topic'] == i
            assert state == time()

    def test_on_send_completed(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        other_time: float = 56.7
        producer: Producer = Mock(name='producer', autospec=Producer)
        metadata = Mock(name='metadata')
        mon.on_send_completed(producer, other_time, metadata)
        assert mon.send_latency[-1] == time() - other_time

    def test_on_send_error(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        producer: Producer = Mock(name='producer', autospec=Producer)
        state = Mock(name='state')
        mon.on_send_error(producer, state, KeyError('foo'))
        assert mon.send_errors == 1

    def test_on_assignment_start(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        assignor: Any = Mock(name='assignor')
        state: Dict[str, float] = mon.on_assignment_start(assignor)
        assert state['time_start'] == time()

    def test_on_assignment_completed(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        other_time: float = 56.7
        assignor: Any = Mock(name='assignor')
        assert mon.assignments_completed == 0
        mon.on_assignment_completed(assignor, {'time_start': other_time})
        assert mon.assignment_latency[-1] == time() - other_time
        assert mon.assignments_completed == 1

    def test_on_assignment_error(self, *, mon: Monitor, time: Callable[[], float]) -> None:
        other_time: float = 56.7
        assignor: Any = Mock(name='assignor')
        assert mon.assignments_failed == 0
        mon.on_assignment_error(assignor, {'time_start': other_time}, KeyError())
        assert mon.assignment_latency[-1] == time() - other_time
        assert mon.assignments_failed == 1

    def test_on_rebalance_start(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        assert mon.rebalances == 0
        app.rebalancing_count = 1
        state: Dict[str, float] = mon.on_rebalance_start(app)
        assert state['time_start'] == time()
        assert mon.rebalances == 1

    def test_on_rebalance_return(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        other_time: float = 56.7
        state: Dict[str, float] = {'time_start': other_time}
        mon.on_rebalance_return(app, state)
        assert mon.rebalance_return_latency[-1] == time() - other_time
        assert state['time_return'] == time()
        assert state['latency_return'] == time() - other_time

    def test_on_rebalance_end(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        other_time: float = 56.7
        state: Dict[str, float] = {'time_start': other_time}
        mon.on_rebalance_end(app, state)
        assert mon.rebalance_end_latency[-1] == time() - other_time
        assert state['time_end'] == time()
        assert state['latency_end'] == time() - other_time

    def test_on_web_request_start(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        request: Any = Mock(name='request')
        view: Any = Mock(name='view')
        state: Dict[str, float] = mon.on_web_request_start(app, request, view=view)
        assert state['time_start'] == time()

    def test_on_web_request_end(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        response: Any = Mock(name='response')
        response.status = 404
        self.assert_on_web_request_end(mon, time, app, response, expected_status=404)

    def test_on_web_request_end__None_response(self, *, mon: Monitor, time: Callable[[], float], app: Any) -> None:
        self.assert_on_web_request_end(mon, time, app, None, expected_status=500)

    def assert_on_web_request_end(self, mon: Monitor, time: Callable[[], float], app: Any, response: Optional[Any],
                                  expected_status: int) -> None:
        request: Any = Mock(name='request')
        view: Any = Mock(name='view')
        other_time: float = 156.9
        state: Dict[str, Any] = {'time_start': other_time}
        mon.on_web_request_end(app, request, response, state, view=view)
        assert state['time_end'] == time()
        assert state['latency_end'] == time() - other_time
        assert state['status_code'] == HTTPStatus(expected_status)

        assert mon.http_response_latency[-1] == time() - other_time
        assert mon.http_response_codes[HTTPStatus(expected_status)] == 1

    def test_TableState_asdict(self, *, mon: Monitor, table: Table) -> None:
        state: TableState = mon._table_or_create(table)
        assert isinstance(state, TableState)
        assert state.table is table
        assert state.keys_retrieved == 0
        assert state.keys_updated == 0
        assert state.keys_deleted == 0

        expected_asdict: Dict[str, int] = {
            'keys_retrieved': 0,
            'keys_updated': 0,
            'keys_deleted': 0,
        }
        assert state.asdict() == expected_asdict
        assert state.__reduce_keywords__() == {
            **state.asdict(),
            'table': table,
        }

    def test_on_tp_commit(self, *, mon: Monitor) -> None:
        topic: str = 'foo'
        for offset in range(20):
            partitions = list(range(4))
            tps = {TP(topic=topic, partition=p) for p in partitions}
            commit_offsets: Dict[TP, int] = {tp: offset for tp in tps}
            mon.on_tp_commit(commit_offsets)
            assert all(mon.tp_committed_offsets[tp] == commit_offsets[tp]
                       for tp in tps)
            offsets_dict: Dict[int, int] = mon.asdict()['topic_committed_offsets'][topic]
            assert all(offsets_dict[p] == offset for p in partitions)

    def test_track_tp_end_offsets(self, *, mon: Monitor) -> None:
        tp: TP = TP(topic='foo', partition=2)
        for offset in range(20):
            mon.track_tp_end_offset(tp, offset)
            assert mon.tp_end_offsets[tp] == offset
            offsets_dict: Dict[int, int] = mon.asdict()['topic_end_offsets'][tp.topic]
            assert offsets_dict[tp.partition] == offset

    @pytest.mark.asyncio
    async def test_service_sampler(self, *, mon: Monitor) -> None:
        mon = Monitor()

        i: int = 0
        mon.events_runtime = []  # type: ignore
        mon.sleep = AsyncMock(name='sleep')

        def on_sample(prev_events: int, prev_messages: int) -> Tuple[int, int]:
            nonlocal i
            mon.events_runtime.append(i + 0.34)  # type: ignore
            i += 1
            if i > 10:
                mon._stopped.set()
            return prev_events, prev_messages

        mon._sample = Mock(name='_sample')
        mon._sample.side_effect = on_sample

        await mon._sampler(mon)

    def test__sample(self, *, mon: Monitor) -> None:
        prev_event_total: int = 0
        prev_message_total: int = 0
        mon.events_runtime = []  # type: ignore
        mon._sample(prev_event_total, prev_message_total)
        mon.events_runtime = deque(range(100))
        mon.rebalance_return_latency = deque(range(100))
        mon.rebalance_end_latency = deque(range(100))
        mon.http_response_latency = deque(range(100))
        prev_event_total = 0
        prev_message_total = 0
        mon._sample(prev_event_total, prev_message_total)

        assert mon.events_runtime_avg == median(mon.events_runtime)
        assert mon.events_s == 0  # XXX this is wrong!

        assert mon.rebalance_return_avg == median(mon.rebalance_return_latency)
        assert mon.rebalance_end_avg == median(mon.rebalance_end_latency)
        assert mon.http_response_latency_avg == median(mon.http_response_latency)
