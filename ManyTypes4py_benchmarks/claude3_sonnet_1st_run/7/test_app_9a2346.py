import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
from mode.utils.compat import want_bytes
from mode.utils.mocks import AsyncMock, Mock, call
from faust.livecheck import LiveCheck
from faust.livecheck.app import LiveCheckSensor
from faust.livecheck.exceptions import TestFailed
from faust.livecheck.locals import current_test_stack
from faust.livecheck.models import SignalEvent, TestExecution, TestReport
from faust.livecheck.signals import BaseSignal

class test_LiveCheckSensor:

    @pytest.fixture()
    def sensor(self) -> LiveCheckSensor:
        return LiveCheckSensor()

    def test_on_stream_event__no_test(self, *, sensor: LiveCheckSensor) -> None:
        stream: Mock = Mock()
        stream.current_test = None
        event: Mock = Mock()
        event.headers = {}
        state: Any = sensor.on_stream_event_in(('topic', 'foo'), 3, stream, event)
        sensor.on_stream_event_out(('topic', 'foo'), 3, stream, event, state)

    def test_on_stream_event(self, *, sensor: LiveCheckSensor, execution: TestExecution) -> None:
        stream: Mock = Mock()
        stream.current_test = None
        event: Mock = Mock()
        event.headers = execution.as_headers()
        assert current_test_stack.top is None
        state: Any = sensor.on_stream_event_in(('topic', 'foo'), 3, stream, event)
        assert current_test_stack.top.id == execution.id
        assert stream.current_test.id == execution.id
        sensor.on_stream_event_out(('topic', 'foo'), 3, stream, event, state)
        assert current_test_stack.top is None
        assert stream.current_test is None

class test_LiveCheck:

    @pytest.mark.parametrize('kwarg,value,expected_value', [('test_topic_name', 'test-topic', 'test-topic'), ('bus_topic_name', 'bus-topic', 'bus-topic'), ('report_topic_name', 'report-topic', 'report-topic'), ('bus_concurrency', 1000, 1000), ('test_concurrency', 1001, 1001), ('send_reports', False, False)])
    def test_constructor(self, kwarg: str, value: Any, expected_value: Any) -> None:
        app: LiveCheck = LiveCheck('foo', **{kwarg: value})
        assert getattr(app, kwarg) == value

    def test_current_test(self, *, livecheck: LiveCheck) -> None:
        test: Mock = Mock()
        assert livecheck.current_test is None
        with current_test_stack.push(test):
            assert livecheck.current_test is test
        assert livecheck.current_test is None

    def test__can_resolve(self, livecheck: LiveCheck) -> None:
        assert isinstance(livecheck._can_resolve, asyncio.Event)

    def test_on_produce_attach_test_headers(self, *, livecheck: LiveCheck, app: Any, execution: TestExecution) -> None:
        headers: List[Tuple[str, bytes]] = [('Foo-Bar-Baz', b'moo')]
        original_headers: List[Tuple[str, bytes]] = list(headers)
        with current_test_stack.push(execution):
            livecheck.on_produce_attach_test_headers(sender=app, key=b'k', value=b'v', partition=3, headers=headers)
            kafka_headers: Dict[str, bytes] = {k: want_bytes(v) for k, v in execution.as_headers().items()}
            assert headers == original_headers + list(kafka_headers.items())

    def test_on_produce_attach_test_headers__no_test(self, *, livecheck: LiveCheck, app: Any) -> None:
        assert livecheck.current_test is None
        headers: List[Tuple[str, bytes]] = []
        livecheck.on_produce_attach_test_headers(sender=app, key=b'k', value=b'v', partition=3, headers=headers)
        assert not headers

    def test_on_produce_attach_test_headers__missing(self, *, livecheck: LiveCheck, app: Any) -> None:
        test: Mock = Mock()
        with current_test_stack.push(test):
            with pytest.raises(TypeError):
                livecheck.on_produce_attach_test_headers(sender=app, key=b'k', value=b'v', partition=3, headers=None)

    def test_case_decorator(self, *, livecheck: LiveCheck) -> None:

        class SignalWithNoneOrigin(livecheck.Signal):
            __origin__: Optional[str] = None

        @livecheck.case()
        class test_foo:
            signal3 = livecheck.Signal()
        assert isinstance(test_foo.signal1, BaseSignal)
        assert isinstance(test_foo.signal2, SignalWithNoneOrigin)
        assert isinstance(test_foo.signal3, BaseSignal)
        assert test_foo.signal1.case is test_foo
        assert test_foo.signal2.case is test_foo
        assert test_foo.signal1.index == 1
        assert test_foo.signal2.index == 2
        assert test_foo.signal3.index == 3

    def test_add_case(self, *, livecheck: LiveCheck) -> None:
        case: Mock = Mock()
        livecheck.add_case(case)
        assert livecheck.cases[case.name] is case

    @pytest.mark.asyncio
    async def test_post_report(self, *, livecheck: LiveCheck) -> None:
        report: Mock = Mock()
        report.test = None
        livecheck.reports.send = AsyncMock()
        await livecheck.post_report(report)
        livecheck.reports.send.assert_called_once_with(key=None, value=report)

    @pytest.mark.asyncio
    async def test_post_report__with_test(self, *, livecheck: LiveCheck) -> None:
        report: Mock = Mock()
        report.test = Mock()
        report.test.id = 'id'
        livecheck.reports.send = AsyncMock()
        await livecheck.post_report(report)
        livecheck.reports.send.assert_called_once_with(key='id', value=report)

    @pytest.mark.asyncio
    async def test_on_start(self, *, livecheck: LiveCheck) -> None:
        livecheck._install_bus_agent = Mock()
        livecheck._install_test_execution_agent = Mock()
        await livecheck.on_start()
        livecheck._install_bus_agent.assert_called_once_with()
        livecheck._install_test_execution_agent.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_on_started(self, *, livecheck: LiveCheck) -> None:
        case1: Mock = Mock(name='case1')
        case2: Mock = Mock(name='case2')
        livecheck.cases = {'a': case1, 'b': case2}
        livecheck.add_runtime_dependency = AsyncMock()
        await livecheck.on_started()
        livecheck.add_runtime_dependency.assert_has_calls([call.coro(case1), call.coro(case2)])

    def test__install_bus_agent(self, *, livecheck: LiveCheck) -> Any:
        livecheck.agent = Mock()
        ag: Any = livecheck._install_bus_agent()
        assert ag is livecheck.agent.return_value.return_value
        livecheck.agent.assert_called_once_with(channel=livecheck.bus, concurrency=livecheck.bus_concurrency)

    def test__install_test_execution_agent(self, *, livecheck: LiveCheck) -> Any:
        livecheck.agent = Mock()
        ag: Any = livecheck._install_test_execution_agent()
        assert ag is livecheck.agent.return_value.return_value
        livecheck.agent.assert_called_once_with(channel=livecheck.pending_tests, concurrency=livecheck.test_concurrency)

    @pytest.mark.asyncio
    async def test__populate_signals(self, *, livecheck: LiveCheck, execution: TestExecution) -> None:
        events: Mock = Mock()
        signal: SignalEvent = SignalEvent(signal_name='foo', case_name=execution.case_name, key=b'k', value=b'v')
        signal2: SignalEvent = SignalEvent(signal_name='foo', case_name='bar', key=b'k2', value=b'v2')
        case: Mock = livecheck.cases[execution.case_name] = Mock(resolve_signal=AsyncMock())
        livecheck.cases.pop('bar', None)

        async def iterate_events() -> Any:
            yield (execution.id, signal)
            yield (execution.id, signal2)
        events.items.side_effect = iterate_events
        await livecheck._populate_signals(events)
        case.resolve_signal.assert_called_once_with(execution.id, signal)

    @pytest.mark.asyncio
    async def test__excecute_tests(self, *, livecheck: LiveCheck, execution: TestExecution) -> None:
        tests: Mock = Mock()
        execution2: TestExecution = execution.derive(case_name='bar')

        async def iterate_tests() -> Any:
            yield (execution.id, execution)
            yield (execution.id, execution2)
        tests.items.side_effect = iterate_tests
        case: Mock = livecheck.cases[execution.case_name] = Mock(execute=AsyncMock())
        livecheck.cases.pop('bar', None)
        await livecheck._execute_tests(tests)
        case.execute.assert_called_once_with(execution)

    @pytest.mark.asyncio
    async def test__excecute_tests__raises(self, *, livecheck: LiveCheck, execution: TestExecution) -> None:
        tests: Mock = Mock()

        async def iterate_tests() -> Any:
            yield (execution.id, execution)
        tests.items.side_effect = iterate_tests
        case: Mock = livecheck.cases[execution.case_name] = Mock(execute=AsyncMock(side_effect=TestFailed()))
        await livecheck._execute_tests(tests)
        case.execute.assert_called_once_with(execution)

    @pytest.mark.parametrize('name,origin,expected', [('__main__.test_foo', 'examples.f.y', 'examples.f.y.test_foo'), ('examples.test_foo', 'examples.f.y', 'examples.test_foo')])
    def test_prepare_case_name(self, name: str, origin: str, expected: str, *, livecheck: LiveCheck) -> None:
        livecheck.conf.origin = origin
        assert livecheck._prepare_case_name(name) == expected

    def test_prepare_case_name__no_origin(self, *, livecheck: LiveCheck) -> None:
        livecheck.conf.origin = None
        with pytest.raises(RuntimeError):
            livecheck._prepare_case_name('__main__.test_foo')

    def test_bus(self, *, livecheck: LiveCheck) -> Any:
        livecheck.topic = Mock()
        assert livecheck.bus is livecheck.topic.return_value
        livecheck.topic.assert_called_once_with(livecheck.bus_topic_name, key_type=str, value_type=SignalEvent)

    def test_pending_tests(self, *, livecheck: LiveCheck) -> Any:
        livecheck.topic = Mock()
        assert livecheck.pending_tests is livecheck.topic.return_value
        livecheck.topic.assert_called_once_with(livecheck.test_topic_name, key_type=str, value_type=TestExecution)

    def test_reports(self, *, livecheck: LiveCheck) -> Any:
        livecheck.topic = Mock()
        assert livecheck.reports is livecheck.topic.return_value
        livecheck.topic.assert_called_once_with(livecheck.report_topic_name, key_type=str, value_type=TestReport)
