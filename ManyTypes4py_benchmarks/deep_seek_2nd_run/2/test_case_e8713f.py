from contextlib import contextmanager
from collections import deque
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Optional, Generator, Tuple, Union, cast
import pytest
from mode import label
from mode.utils.mocks import ANY, AsyncMock, Mock, patch
from faust.livecheck import Case
from faust.livecheck.exceptions import SuiteFailed
from faust.livecheck.models import State, TestReport

class test_Case:

    @pytest.mark.parametrize('arg,value,expected', [
        ('active', False, False),
        ('probability', 3.33, 3.33),
        ('warn_stalled_after', 4.44, 4.44),
        ('test_expires', 5.55, timedelta(seconds=5.55)),
        ('frequency', 6.66, 6.66),
        ('realtime_logs', True, True),
        ('max_history', 3000, 3000),
        ('max_consecutive_failures', 3, 3),
        ('url_timeout_total', 7.77, 7.77),
        ('url_timeout_connect', 8.88, 8.88),
        ('url_error_retries', 6, 6),
        ('url_error_delay_min', 9.99, 9.99),
        ('url_error_delay_backoff', 10.1, 10.1),
        ('url_error_delay_max', 11.11, 11.11)
    ])
    def test_constructor(self, arg: str, value: Any, expected: Any, *, livecheck: Any) -> None:
        kwargs = {'app': livecheck, 'name': 'n'}
        case = Case(**{arg: value}, **kwargs)
        assert getattr(case, arg) == expected

    @pytest.mark.asyncio
    async def test__sampler(self, *, case: Case) -> None:
        case._sample = AsyncMock()
        case.sleep = AsyncMock()

        def on_sample() -> None:
            if case._sample.call_count == 3:
                case._stopped.set()
        case._sample.coro.side_effect = on_sample
        await case._sampler(case)
        assert case._sample.call_count == 3

    @pytest.mark.asyncio
    async def test__sample(self, *, case: Case) -> None:
        await case._sample()
        case.frequency_history.extend(range(100))
        case.latency_history.extend(range(100, 200))
        case.runtime_history.extend(range(200, 300))
        await case._sample()
        assert case.frequency_avg == median(case.frequency_history)
        assert case.latency_avg == median(case.latency_history)
        assert case.runtime_avg == median(case.runtime_history)

    @pytest.mark.asyncio
    async def test_maybe_trigger(self, *, case: Case) -> None:
        case.trigger = AsyncMock('trigger')
        with patch('faust.livecheck.case.uuid'):
            with patch('faust.livecheck.case.uniform') as uniform:
                uniform.return_value = 1.0
                async with case.maybe_trigger() as test:
                    assert test is None
                case.trigger.assert_not_called()
                uniform.return_value = 0.0
                async with case.maybe_trigger() as test:
                    assert test is case.trigger.coro.return_value
                    assert case.current_test is test

    @pytest.mark.asyncio
    async def test_run(self, *, case: Case) -> None:
        with pytest.raises(NotImplementedError):
            await Case.run(case)

    @pytest.mark.asyncio
    async def test_trigger(self, *, case: Case) -> None:
        case.app = Mock(pending_tests=Mock(send=AsyncMock()))
        t = await case.trigger('id1', 30, kw=2)
        assert t.id == 'id1'
        case.app.pending_tests.send.coro.assert_called_once_with(key='id1', value=t)

    def test_now(self, *, case: Case) -> None:
        assert isinstance(case._now(), datetime)
        assert case._now().tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_resolve_signal(self, *, case: Case) -> None:
        key = 'k'
        event = Mock(name='event')
        case.signals[event.signal_name] = Mock(resolve=AsyncMock())
        await case.resolve_signal(key, event)
        case.signals[event.signal_name].resolve.coro.assert_called_once_with(key, event)

    @pytest.mark.asyncio
    async def test_execute(self, *, case: Case, execution: Any, current_execution_stack: Any, frozen_monotonic: Any) -> None:
        case.Runner = Mock(name='case.Runner')
        runner = case.Runner.return_value
        runner.execute = AsyncMock(name='runner.execute')
        await case.execute(execution)
        case.Runner.assert_called_once_with(case, execution, started=frozen_monotonic.return_value)
        runner.execute.coro.assert_called_once_with()
        current_execution_stack.push.assert_called_once_with(runner)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('started,last_received,frequency', [
        (100.0, None, None),
        (100.0, 50.0, 10.0),
        (100.0, 50.0, None)
    ])
    async def test_on_test_start(self, started: float, last_received: Optional[float], frequency: Optional[float], *, case: Case, runner: Any) -> None:
        case.latency_history = deque([0.03] * case.max_history)
        case.frequency_history = deque([0.04] * case.max_history)
        runner.started = started
        case.frequency = frequency
        case.last_test_received = last_received
        await case.on_test_start(runner)
        if last_received:
            time_since = started - last_received
            if frequency:
                latency = time_since - frequency
                assert case.latency_history[-1] == latency
                assert len(case.latency_history) == case.max_history
            assert case.frequency_history[-1] == time_since
            assert len(case.frequency_history) == case.max_history

    @pytest.yield_fixture()
    def frozen_monotonic(self) -> Generator[Mock, None, None]:
        with self._patch_monotonic() as monotonic:
            yield monotonic

    def _patch_monotonic(self) -> Mock:
        return patch('faust.livecheck.case.monotonic')

    @pytest.mark.asyncio
    async def test_on_test_skipped(self, *, case: Case, runner: Any, frozen_monotonic: Mock) -> None:
        await case.on_test_skipped(runner)
        assert case.last_test_received is frozen_monotonic.return_value

    @pytest.mark.asyncio
    async def test_on_test_failed(self, *, case: Case, runner: Any) -> None:
        case._set_test_error_state = AsyncMock()
        await case.on_test_failed(runner, KeyError())
        case._set_test_error_state.coro.assert_called_once_with(State.FAIL)

    @pytest.mark.asyncio
    async def test_on_test_error(self, *, case: Case, runner: Any) -> None:
        case._set_test_error_state = AsyncMock()
        await case.on_test_error(runner, KeyError())
        case._set_test_error_state.coro.assert_called_once_with(State.ERROR)

    @pytest.mark.asyncio
    async def test_on_test_timeout(self, *, case: Case, runner: Any) -> None:
        case._set_test_error_state = AsyncMock()
        await case.on_test_timeout(runner, KeyError())
        case._set_test_error_state.coro.assert_called_once_with(State.TIMEOUT)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('state,failures,fail_suite', [
        (State.FAIL, 0, False),
        (State.FAIL, 9, True),
        (State.ERROR, 0, False),
        (State.ERROR, 9, True),
        (State.STALL, 0, False),
        (State.STALL, 9, True)
    ])
    async def test__set_test_error_state(self, state: State, failures: int, fail_suite: bool, *, case: Case) -> None:
        case.max_consecutive_failures = 10
        case.consecutive_failures = failures
        case.on_suite_fail = AsyncMock()
        await case._set_test_error_state(state)
        assert case.status == state
        assert case.consecutive_failures == failures + 1
        assert case.total_failures == 1
        assert case.total_by_state[state] == 1
        if fail_suite:
            case.on_suite_fail.coro.assert_called_once_with(ANY)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('initial_state,ts,now,failed,expected_state', [
        (State.INIT, 300, 300.0, None, State.PASS),
        (State.PASS, 300, 300.0, 100.1, State.PASS),
        (State.FAIL, 300, 300.0, 299.0, State.FAIL),
        (State.FAIL, 100, 300.0, None, State.PASS),
        (State.FAIL, 101, 300.0, 100.0, State.PASS),
        (State.FAIL, 99, 300.0, 100.0, State.FAIL)
    ])
    async def test_on_suite_pass(self, initial_state: State, ts: int, now: float, failed: Optional[float], expected_state: State, *, case: Case, runner: Any, execution: Any) -> None:
        assert runner.test is execution
        runner.test.timestamp = Mock()
        runner.test.timestamp.timestamp.return_value = ts
        case.runtime_history = deque([3.03] * case.max_history)
        runner.runtime = 300.0
        with self.seconds_since_last_fail(case, now=now, failed=failed):
            case.status = initial_state
            await case.on_test_pass(runner)
            assert case.status == expected_state
            assert len(case.runtime_history) == case.max_history
            assert case.runtime_history[-1] == 300.0

    @pytest.mark.asyncio
    async def test_post_report(self, *, case: Case) -> None:
        report = Mock()
        case.app.post_report = AsyncMock()
        await case.post_report(report)
        case.app.post_report.assert_called_once_with(report)

    @pytest.mark.asyncio
    async def test__send_frequency__first_stop(self, *, case: Case, loop: Any) -> None:
        case.frequency = 0.1
        case.sleep = AsyncMock()
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                case._stopped.set()
                yield 0.1
                yield 0.2
                yield 0.3
                yield 0.4
            ti.side_effect = on_itertimer
            await case._send_frequency(case)

    @pytest.mark.asyncio
    async def test__send_frequency__no_frequency(self, *, case: Case, loop: Any) -> None:
        case.frequency = 0.0
        case.sleep = AsyncMock()
        case.make_fake_request = AsyncMock()
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                case._stopped.set()
                yield 0.1
                yield 0.2
                yield 0.3
                yield 0.4
            ti.side_effect = on_itertimer
            await case._send_frequency(case)
        case.make_fake_request.assert_not_called()

    @pytest.mark.asyncio
    async def test__send_frequency__last_stop(self, *, case: Case) -> None:
        case.frequency = 0.1
        case.sleep = AsyncMock()
        case.app.is_leader = Mock(return_value=False)
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                for val in [0.1, 0.2, 0.3, 0.4]:
                    await case.sleep(val)
                    yield val
            ti.side_effect = on_itertimer

            async def on_sleep(secs: float, **kwargs: Any) -> None:
                if case.sleep.call_count >= 2:
                    case._stopped.set()
            case.sleep.side_effect = on_sleep
            await case._send_frequency(case)

    @pytest.mark.asyncio
    async def test__send_frequency__no_frequency_None(self, *, case: Case) -> None:
        case.frequency = None
        await case._send_frequency(case)

    @pytest.mark.asyncio
    async def test__send_frequency__timer_ends(self, *, case: Case) -> None:
        case.sleep = AsyncMock()
        case.frequency = 10.0
        case.app.is_leader = Mock(return_value=False)
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                for val in [0.1, 0.2, 0.3, 0.4]:
                    yield val
            ti.side_effect = on_itertimer
            await case._send_frequency(case)

    @pytest.mark.asyncio
    async def test__send_frequency(self, *, case: Case) -> None:
        case.frequency = 0.1
        case.make_fake_request = AsyncMock()
        case.sleep = AsyncMock()
        case.app.is_leader = Mock(return_value=False)

        def on_make_fake_request() -> None:
            if case.make_fake_request.call_count == 3:
                case._stopped.set()
        case.make_fake_request.coro.side_effect = on_make_fake_request

        def on_is_leader() -> bool:
            if case.app.is_leader.call_count >= 2:
                return True
            return False
        case.app.is_leader.side_effect = on_is_leader
        await case._send_frequency(case)
        assert case.make_fake_request.call_count == 3

    @pytest.mark.asyncio
    async def test__check_frequency(self, *, case: Case) -> None:
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    yield val
            ti.side_effect = on_itertimer
            case.sleep = AsyncMock()
            await case._check_frequency(case)

    @pytest.mark.asyncio
    async def test__check_frequency__last(self, *, case: Case, frozen_monotonic: Mock) -> None:
        frozen_monotonic.return_value = 600.0
        case.warn_stalled_after = 10.0
        case.on_suite_fail = AsyncMock()
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                case.last_test_received = 10.0
                for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    yield val
            ti.side_effect = on_itertimer
            case.sleep = AsyncMock()
            await case._check_frequency(case)
        case.on_suite_fail.assert_called_once_with(ANY, State.STALL)

    @pytest.mark.asyncio
    async def test__check_frequency__should_stop1(self, *, case: Case) -> None:
        with patch('mode.services.Timer') as ti:

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                case._stopped.set()
                yield 0.1
                yield 0.2
                yield 0.3
                yield 0.4
                yield 0.5
            ti.side_effect = on_itertimer
            case.sleep = AsyncMock()
            await case._check_frequency(case)

    @pytest.mark.asyncio
    async def test__check_frequency__last_stop(self, *, case: Case) -> None:
        with patch('mode.services.Timer') as ti:
            case._stopped.clear()
            assert not case.should_stop

            async def on_itertimer(*args: Any, **kwargs: Any) -> Generator[float, None, None]:
                for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    await case.sleep(val)
                    yield val
            ti.side_effect = on_itertimer
            case.sleep = AsyncMock()

            async def on_sleep(arg: float, **kwargs: Any) -> None:
                if case.sleep.call_count >= 2:
                    case._stopped.set()
            case.sleep.side_effect = on_sleep
            await case._check_frequency(case)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('initial_state,now,f