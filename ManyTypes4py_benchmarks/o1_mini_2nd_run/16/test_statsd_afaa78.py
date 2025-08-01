import pytest
from faust import web
from faust.exceptions import ImproperlyConfigured
from faust.sensors.statsd import StatsdMonitor
from faust.types import TP
from mode.utils.mocks import Mock, call
from typing import Any, Dict, Optional

TP1: TP = TP('foo', 3)


class test_StatsdMonitor:

    @pytest.fixture
    def time(self) -> Mock:
        timefun: Mock = Mock(name='time()')
        timefun.return_value = 101.1
        return timefun

    @pytest.fixture()
    def statsd(self, *, monkeypatch: Any) -> Mock:
        statsd: Mock = Mock(name='statsd')
        monkeypatch.setattr('faust.sensors.statsd.statsd', statsd)
        return statsd

    @pytest.fixture()
    def stream(self) -> Mock:
        stream: Mock = Mock(name='stream')
        stream.shortlabel = 'Stream: Topic: foo'
        return stream

    @pytest.fixture()
    def event(self) -> Mock:
        return Mock(name='event')

    @pytest.fixture()
    def table(self) -> Mock:
        table: Mock = Mock(name='table')
        table.name = 'table1'
        return table

    @pytest.fixture()
    def req(self) -> Mock:
        return Mock(name='request', autospec=web.Request)

    @pytest.fixture()
    def response(self) -> Mock:
        return Mock(name='response', autospec=web.Response)

    @pytest.fixture()
    def view(self) -> Mock:
        return Mock(name='view', autospec=web.View)

    @pytest.fixture()
    def mon(self, *, statsd: Mock, time: Mock) -> StatsdMonitor:
        return StatsdMonitor(time=time)

    def test_statsd(self, *, mon: StatsdMonitor) -> None:
        assert mon.client

    def test_raises_if_statsd_not_installed(self, *, monkeypatch: Any) -> None:
        monkeypatch.setattr('faust.sensors.statsd.statsd', None)
        with pytest.raises(ImproperlyConfigured):
            StatsdMonitor()

    def test_on_message_in_out(self, *, mon: StatsdMonitor) -> None:
        message: Mock = Mock(name='message')
        mon.on_message_in(TP1, 400, message)
        mon.client.incr.assert_has_calls([
            call('messages_received', rate=mon.rate),
            call('messages_active', rate=mon.rate),
            call('topic.foo.messages_received', rate=mon.rate)
        ])
        mon.client.gauge.assert_called_once_with('read_offset.foo.3', 400)
        mon.on_message_out(TP1, 400, message)
        mon.client.decr.assert_called_once_with('messages_active', rate=mon.rate)

    def test_on_stream_event_in_out(self, *, mon: StatsdMonitor, stream: Mock, event: Mock) -> None:
        state: Any = mon.on_stream_event_in(TP1, 401, stream, event)
        mon.client.incr.assert_has_calls([
            call('events', rate=mon.rate),
            call('stream.topic_foo.events', rate=mon.rate),
            call('events_active', rate=mon.rate)
        ])
        mon.on_stream_event_out(TP1, 401, stream, event, state)
        mon.client.decr.assert_called_once_with('events_active', rate=mon.rate)
        mon.client.timing.assert_called_once_with(
            'events_runtime', mon.secs_to_ms(mon.events_runtime[-1]), rate=mon.rate
        )

    def test_on_table_get(self, *, mon: StatsdMonitor, table: Mock) -> None:
        mon.on_table_get(table, 'key')
        mon.client.incr.assert_called_once_with('table.table1.keys_retrieved', rate=mon.rate)

    def test_on_table_set(self, *, mon: StatsdMonitor, table: Mock) -> None:
        mon.on_table_set(table, 'key', 'value')
        mon.client.incr.assert_called_once_with('table.table1.keys_updated', rate=mon.rate)

    def test_on_table_del(self, *, mon: StatsdMonitor, table: Mock) -> None:
        mon.on_table_del(table, 'key')
        mon.client.incr.assert_called_once_with('table.table1.keys_deleted', rate=mon.rate)

    def test_on_commit_completed(self, *, mon: StatsdMonitor) -> None:
        consumer: Mock = Mock(name='consumer')
        state: Any = mon.on_commit_initiated(consumer)
        mon.on_commit_completed(consumer, state)
        mon.client.timing.assert_called_once_with(
            'commit_latency', mon.ms_since(float(state)), rate=mon.rate
        )

    def test_on_send_initiated_completed(self, *, mon: StatsdMonitor) -> None:
        producer: Mock = Mock(name='producer')
        state: Any = mon.on_send_initiated(producer, 'topic1', 'message', 321, 123)
        mon.on_send_completed(producer, state, Mock(name='metadata'))
        mon.client.incr.assert_has_calls([
            call('topic.topic1.messages_sent', rate=mon.rate),
            call('messages_sent', rate=mon.rate)
        ])
        mon.client.timing.assert_called_once_with(
            'send_latency', mon.ms_since(float(state)), rate=mon.rate
        )
        mon.on_send_error(producer, KeyError('foo'), state)
        mon.client.incr.assert_has_calls([
            call('messages_sent_error', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('send_latency_for_error', mon.ms_since(float(state)), rate=mon.rate)
        ])

    def test_on_assignment_start_completed(self, *, mon: StatsdMonitor) -> None:
        assignor: Mock = Mock(name='assignor')
        state: Any = mon.on_assignment_start(assignor)
        mon.on_assignment_completed(assignor, state)
        mon.client.incr.assert_has_calls([
            call('assignments_complete', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('assignment_latency', mon.ms_since(state['time_start']), rate=mon.rate)
        ])

    def test_on_assignment_start_failed(self, *, mon: StatsdMonitor) -> None:
        assignor: Mock = Mock(name='assignor')
        state: Any = mon.on_assignment_start(assignor)
        mon.on_assignment_error(assignor, state, KeyError())
        mon.client.incr.assert_has_calls([
            call('assignments_error', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('assignment_latency', mon.ms_since(state['time_start']), rate=mon.rate)
        ])

    def test_on_rebalance(self, *, mon: StatsdMonitor) -> None:
        app: Mock = Mock(name='app')
        state: Any = mon.on_rebalance_start(app)
        mon.client.incr.assert_has_calls([
            call('rebalances', rate=mon.rate)
        ])
        mon.on_rebalance_return(app, state)
        mon.client.incr.assert_has_calls([
            call('rebalances', rate=mon.rate),
            call('rebalances_recovering', rate=mon.rate)
        ])
        mon.client.decr.assert_has_calls([
            call('rebalances', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('rebalance_return_latency', mon.ms_since(state['time_return']), rate=mon.rate)
        ])
        mon.on_rebalance_end(app, state)
        mon.client.decr.assert_has_calls([
            call('rebalances', rate=mon.rate),
            call('rebalances_recovering', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('rebalance_return_latency', mon.ms_since(state['time_return']), rate=mon.rate),
            call('rebalance_end_latency', mon.ms_since(state['time_end']), rate=mon.rate)
        ])

    def test_on_web_request(self, *, mon: StatsdMonitor, req: Mock, response: Mock, view: Mock) -> None:
        response.status = 404
        self.assert_on_web_request(mon, req, response, view, expected_status=404)

    def test_on_web_request__None_response(self, *, mon: StatsdMonitor, req: Mock, view: Mock) -> None:
        self.assert_on_web_request(mon, req, None, view, expected_status=500)

    def assert_on_web_request(
        self,
        mon: StatsdMonitor,
        request: Mock,
        response: Optional[Mock],
        view: Mock,
        expected_status: int
    ) -> None:
        app: Mock = Mock(name='app')
        state: Any = mon.on_web_request_start(app, request, view=view)
        mon.on_web_request_end(app, request, response, state, view=view)
        mon.client.incr.assert_has_calls([
            call(f'http_status_code.{expected_status}', rate=mon.rate)
        ])
        mon.client.timing.assert_has_calls([
            call('http_response_latency', mon.ms_since(state['time_end']), rate=mon.rate)
        ])

    def test_count(self, *, mon: StatsdMonitor) -> None:
        mon.count('metric_name', count=3)
        mon.client.incr.assert_called_once_with('metric_name', count=3, rate=mon.rate)

    def test_on_tp_commit(self, *, mon: StatsdMonitor) -> None:
        offsets: Dict[TP, int] = {
            TP('foo', 0): 1001,
            TP('foo', 1): 2002,
            TP('bar', 3): 3003
        }
        mon.on_tp_commit(offsets)
        mon.client.gauge.assert_has_calls([
            call('committed_offset.foo.0', 1001),
            call('committed_offset.foo.1', 2002),
            call('committed_offset.bar.3', 3003)
        ])

    def test_track_tp_end_offsets(self, *, mon: StatsdMonitor) -> None:
        mon.track_tp_end_offset(TP('foo', 0), 4004)
        mon.client.gauge.assert_called_once_with('end_offset.foo.0', 4004)
