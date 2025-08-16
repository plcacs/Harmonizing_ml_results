from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from mode.utils.logging import CompositeLogger
from mode.utils.times import humanize_seconds
from mode.utils.typing import NoReturn
from faust.models import maybe_model
from .exceptions import LiveCheckError, TestFailed, TestRaised, TestSkipped, TestTimeout
from .locals import current_test_stack
from .models import State, TestExecution, TestReport
from .signals import BaseSignal

class TestRunner:
    state: State = State.INIT
    report: Optional[TestReport] = None
    error: Optional[Exception] = None

    def __init__(self, case: Any, test: Any, started: float) -> None:
        self.case = case
        self.test = test
        self.started = started
        self.ended: Optional[float] = None
        self.runtime: Optional[float] = None
        self.logs: List[Tuple[str, Tuple[Any, ...]]] = []
        self.log: CompositeLogger = CompositeLogger(self.case.log.logger, formatter=self._format_log)
        self.signal_latency: Dict[str, float] = {}

    async def execute(self) -> None:
        async with current_test_stack.push(self.test):
            if not self.case.active:
                await self.skip('case inactive')
            elif self.test.is_expired:
                await self.skip('expired')
            args = self._prepare_args(self.test.test_args)
            kwargs = self._prepare_kwargs(self.test.test_kwargs)
            await self.on_start()
            try:
                await self.case.run(*args, **kwargs)
            except asyncio.CancelledError:
                pass
            except TestSkipped as exc:
                await self.on_skipped(exc)
                raise
            except TestTimeout as exc:
                await self.on_timeout(exc)
                raise
            except AssertionError as exc:
                await self.on_failed(exc)
                raise TestFailed(exc) from exc
            except LiveCheckError as exc:
                await self.on_error(exc)
                raise
            except Exception as exc:
                await self.on_error(exc)
                raise TestRaised(exc) from exc
            else:
                await self.on_pass()

    async def skip(self, reason: str) -> NoReturn:
        exc = TestSkipped(f'Test {self.test.ident} skipped: {reason}')
        try:
            raise exc
        except TestSkipped as exc:
            await self.on_skipped(exc)
            raise
        else:
            assert False

    def _prepare_args(self, args: Iterable[Any]) -> Tuple[Any, ...]:
        to_value = self._prepare_val
        return tuple((to_value(arg) for arg in args))

    def _prepare_kwargs(self, kwargs: Mapping[str, Any]) -> Dict[Any, Any]:
        to_value = self._prepare_val
        return {to_value(k): to_value(v) for k, v in kwargs.items()}

    def _prepare_val(self, arg: Any) -> Any:
        return maybe_model(arg)

    def _format_log(self, severity: int, msg: str, *args: Any, **kwargs: Any) -> str:
        return f'[{self.test.shortident}] {msg}'

    async def on_skipped(self, exc: TestSkipped) -> None:
        self.state = State.SKIP
        self.log.info('Skipped expired test: %s expires=%s', self.test.ident, self.test.expires)
        await self.case.on_test_skipped(self)

    async def on_start(self) -> None:
        self.log_info('≈≈≈ Test %s executing... (issued %s) ≈≈≈', self.case.name, self.test.human_date)
        await self.case.on_test_start(self)

    async def on_signal_wait(self, signal: BaseSignal, timeout: float) -> None:
        self.log_info('∆ %r/%r %s (%rs)...', signal.index, self.case.total_signals, signal.name.upper(), timeout)

    async def on_signal_received(self, signal: BaseSignal, time_start: float, time_end: float) -> None:
        latency = time_end - time_start
        self.signal_latency[signal.name] = latency

    async def on_failed(self, exc: AssertionError) -> None:
        self.end()
        self.error = exc
        self.state = State.FAIL
        self.log.exception('Test failed: %r', exc)
        await self.case.on_test_failed(self, exc)
        await self._finalize_report()

    async def on_error(self, exc: Exception) -> None:
        self.end()
        self.state = State.ERROR
        self.error = exc
        self.log.exception('Test raised: %r', exc)
        await self.case.on_test_error(self, exc)
        await self._finalize_report()

    async def on_timeout(self, exc: TestTimeout) -> None:
        self.end()
        self.error = exc
        self.state = State.TIMEOUT
        self.log.exception('Test timed-out: %r', exc)
        await self.case.on_test_timeout(self, exc)
        await self._finalize_report()

    async def on_pass(self) -> None:
        self.end()
        self.error = None
        self.state = State.PASS
        human_secs = humanize_seconds(self.runtime or 0.0, microseconds=True, now='~0.0 seconds')
        self.log_info('Test OK in %s √', human_secs)
        self._flush_logs()
        await self.case.on_test_pass(self)
        await self._finalize_report()

    async def _finalize_report(self) -> None:
        tb = None
        error = self.error
        if error:
            tb = '\n'.join(traceback.format_tb(error.__traceback__))
        self.report = TestReport(case_name=self.case.name, state=self.state, test=self.test, runtime=self.runtime, signal_latency=self.signal_latency, error=str(error) if error else None, traceback=tb)
        await self.case.post_report(self.report)

    def log_info(self, msg: str, *args: Any) -> None:
        if self.case.realtime_logs:
            self.log.info(msg, *args)
        else:
            self.logs.append((msg, args))

    def end(self) -> None:
        self.ended = monotonic()
        self.runtime = self.ended - self.started

    def _flush_logs(self, severity: int = logging.INFO) -> None:
        logs = self.logs
        try:
            self.log.logger.log(severity, '\n'.join((self._format_log(severity, msg % log_args) for msg, log_args in logs)))
        finally:
            logs.clear()
