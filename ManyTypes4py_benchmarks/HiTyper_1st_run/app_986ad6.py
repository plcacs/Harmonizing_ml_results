"""LiveCheck - Faust Application."""
import asyncio
from datetime import timedelta
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, cast
from mode.signals import BaseSignalT
from mode.utils.compat import want_bytes
from mode.utils.objects import annotations, cached_property, qualname
from mode.utils.times import Seconds
import faust
from faust.app.base import SCAN_CATEGORIES
from faust.sensors.base import Sensor
from faust.types import AgentT, AppT, EventT, StreamT, TP, TopicT
from faust.utils import venusian
from . import patches
from .case import Case
from .exceptions import LiveCheckError
from .locals import current_test, current_test_stack
from .models import SignalEvent, TestExecution, TestReport
from .signals import BaseSignal, Signal
__all__ = ['LiveCheck']
SCAN_CASE = 'livecheck.case'
_Case = Case
patches.patch_all()

class LiveCheckSensor(Sensor):

    def on_stream_event_in(self, tp: Union[int, streams.StreamT, tuples.TP], offset: Union[int, streams.StreamT, tuples.TP], stream: Union[faustypes.StreamT, types.FrameType, int], event: Union[faustypes.StreamT, int, faustypes.tuples.TP]) -> None:
        """Call when stream starts processing event."""
        test = TestExecution.from_headers(event.headers)
        if test is not None:
            stream.current_test = test
            current_test_stack.push_without_automatic_cleanup(test)
        return None

    def on_stream_event_out(self, tp: Union[dict, int, faustypes.TP], offset: Union[dict, int, faustypes.TP], stream: Union[faustypes.StreamT, int, faustypes.tuples.TP], event: Union[dict, int, faustypes.TP], state: Union[None, dict, int, faustypes.TP]=None) -> None:
        """Call when stream is finished handling event."""
        has_active_test = getattr(stream, 'current_test', None)
        if has_active_test:
            stream.current_test = None
            current_test_stack.pop()

class LiveCheck(faust.App):
    """LiveCheck application."""
    SCAN_CATEGORIES = list(SCAN_CATEGORIES) + [SCAN_CASE]
    Signal = Signal
    Case = _Case
    bus_concurrency = 30
    test_concurrency = 100
    send_reports = True
    test_topic_name = 'livecheck'
    bus_topic_name = 'livecheck-bus'
    report_topic_name = 'livecheck-report'

    @classmethod
    def for_app(cls: Union[str, int, aiohttp.web.Application], app: Union[int, dict[str, typing.Any]], *, prefix: typing.Text='livecheck-', web_port: int=9999, test_topic_name: Union[None, str, int]=None, bus_topic_name: Union[None, str, int]=None, report_topic_name: Union[None, str, int]=None, bus_concurrency: Union[None, str, int]=None, test_concurrency: Union[None, str, int]=None, send_reports: Union[None, str, int]=None, **kwargs) -> Union[lemon.app.Lemon, asyncworker.app.App, flask.app.Flask]:
        """Create LiveCheck application targeting specific app.

        The target app will be used to configure the LiveCheck app.
        """
        app_id, passed_kwargs = app._default_options
        livecheck_id = f'{prefix}{app_id}'
        override = {'web_port': web_port, 'test_topic_name': test_topic_name, 'bus_topic_name': bus_topic_name, 'report_topic_name': report_topic_name, 'bus_concurrency': bus_concurrency, 'test_concurrency': test_concurrency, 'send_reports': send_reports, **kwargs}
        options = {**passed_kwargs, **override}
        livecheck_app = cls(livecheck_id, **options)
        livecheck_app._contribute_to_app(app)
        return livecheck_app

    def _contribute_to_app(self, app: Union[aiohttp.web.Application, app.App, faustypes.AppT]) -> None:
        from .patches.aiohttp import LiveCheckMiddleware
        web_app = app.web.web_app
        web_app.middlewares.append(LiveCheckMiddleware())
        app.sensors.add(LiveCheckSensor())
        app.livecheck = self

    def __init__(self, id: Union[str, int, bytes], *, test_topic_name: Union[None, str]=None, bus_topic_name: Union[None, str]=None, report_topic_name: Union[None, str]=None, bus_concurrency: Union[None, int, str]=None, test_concurrency: Union[None, str]=None, send_reports: Union[None, str, list["Node"]]=None, **kwargs) -> None:
        super().__init__(id, **kwargs)
        if test_topic_name is not None:
            self.test_topic_name = test_topic_name
        if bus_topic_name is not None:
            self.bus_topic_name = bus_topic_name
        if report_topic_name is not None:
            self.report_topic_name = report_topic_name
        if bus_concurrency is not None:
            self.bus_concurrency = bus_concurrency
        if test_concurrency is not None:
            self.test_concurrency = test_concurrency
        if send_reports is not None:
            self.send_reports = send_reports
        self.cases = {}
        self._resolved_signals = {}
        patches.patch_all()
        self._apply_monkeypatches()
        self._connect_signals()

    @property
    def current_test(self) -> str:
        """Return the current test context (if any)."""
        return current_test()

    @cached_property
    def _can_resolve(self):
        return asyncio.Event()

    def _apply_monkeypatches(self) -> None:
        patches.patch_all()

    def _connect_signals(self) -> None:
        AppT.on_produce_message.connect(self.on_produce_attach_test_headers)

    def on_produce_attach_test_headers(self, sender: Union[bytes, str, int], key: Union[None, bytes, str, int]=None, value: Union[None, bytes, str, int]=None, partition: Union[None, bytes, str, int]=None, timestamp: Union[None, bytes, str, int]=None, headers: Union[list[tuple[typing.Union[str,bytes]]], str]=None, signal: Union[None, bytes, str, int]=None, **kwargs) -> None:
        """Attach test headers to Kafka produce requests."""
        test = current_test()
        if test is not None:
            if headers is None:
                raise TypeError('Produce request missing headers list')
            headers.extend([(k, want_bytes(v)) for k, v in test.as_headers().items()])

    def case(self, *, name: Union[None, float, bool, mode.utils.times.Seconds]=None, probability: Union[None, float, bool, mode.utils.times.Seconds]=None, warn_stalled_after: Union[float, bool, mode.utils.times.Seconds]=timedelta(minutes=30), active=None, test_expires=None, frequency=None, max_history=None, max_consecutive_failures=None, url_timeout_total=None, url_timeout_connect=None, url_error_retries=None, url_error_delay_min=None, url_error_delay_backoff=None, url_error_delay_max=None, base=Case):
        """Decorate class to be used as a test case.

        Returns:
            :class:`faust.livecheck.Case`.
        """
        base_case = base

        def _inner(cls: Any):
            case_cls = type(cls.__name__, (cls, base_case), {'__module__': cls.__module__, 'app': self})
            signal_types = dict(self._extract_signals(case_cls, base_case))
            signals = []
            for i, (attr_name, attr_type) in enumerate(signal_types.items()):
                signal = getattr(case_cls, attr_name, None)
                if signal is None:
                    signal = attr_type(name=attr_name, index=i + 1)
                    setattr(case_cls, attr_name, signal)
                    signals.append(signal)
                else:
                    signal.index = i + 1
            case = self.add_case(case_cls(app=self, name=self._prepare_case_name(name or qualname(cls)), active=active, probability=probability, warn_stalled_after=warn_stalled_after, signals=signals, test_expires=test_expires, frequency=frequency, max_history=max_history, max_consecutive_failures=max_consecutive_failures, url_timeout_total=url_timeout_total, url_timeout_connect=url_timeout_connect, url_error_retries=url_error_retries, url_error_delay_min=url_error_delay_min, url_error_delay_backoff=url_error_delay_backoff, url_error_delay_max=url_error_delay_max))
            venusian.attach(cast(Callable, case), category=SCAN_CASE)
            return case
        return _inner

    def _extract_signals(self, case_cls: Union[cmk.base.config.ExitSpec, eventsourcing_helpers.models.AggregateRoot], base_case: Union[cmk.base.config.ExitSpec, eventsourcing_helpers.models.AggregateRoot]) -> typing.Generator[tuple]:
        fields, defaults = annotations(case_cls, stop=base_case, skip_classvar=True, localns={case_cls.__name__: case_cls})
        for attr_name, attr_type in fields.items():
            actual_type = getattr(attr_type, '__origin__', attr_type)
            if actual_type is None:
                actual_type = attr_type
            try:
                if issubclass(actual_type, BaseSignal):
                    yield (attr_name, attr_type)
            except TypeError:
                pass

    def add_case(self, case: Union[dict, list, bool]) -> Union[dict, list, bool]:
        """Add and register new test case."""
        self.cases[case.name] = case
        return case

    async def post_report(self, report):
        """Publish test report to reporting topic."""
        key = None
        if report.test is not None:
            key = report.test.id
        await self.reports.send(key=key, value=report)

    async def on_start(self):
        """Call when LiveCheck application starts."""
        await super().on_start()
        self._install_bus_agent()
        self._install_test_execution_agent()

    async def on_started(self):
        """Call when LiveCheck application is fully started."""
        await super().on_started()
        for case in self.cases.values():
            await self.add_runtime_dependency(case)

    def _install_bus_agent(self) -> Union[bool, aiohttp.web.Application]:
        return self.agent(channel=self.bus, concurrency=self.bus_concurrency)(self._populate_signals)

    def _install_test_execution_agent(self) -> Union[str, typing.Callable, aiohttp.web.Application]:
        return self.agent(channel=self.pending_tests, concurrency=self.test_concurrency)(self._execute_tests)

    async def _populate_signals(self, events):
        async for test_id, event in events.items():
            event.case_name = self._prepare_case_name(event.case_name)
            try:
                case = self.cases[event.case_name]
            except KeyError:
                self.log.error('Received signal %r for unregistered case %r', event, (test_id, event.case_name))
            else:
                await case.resolve_signal(test_id, event)

    async def _execute_tests(self, tests):
        async for test_id, test in tests.items():
            test.case_name = self._prepare_case_name(test.case_name)
            try:
                case = self.cases[test.case_name]
            except KeyError:
                self.log.error('Unregistered test case %r with id %r: %r', test.case_name, test_id, test)
            else:
                try:
                    await case.execute(test)
                except LiveCheckError:
                    pass

    def _prepare_case_name(self, name: str) -> str:
        if name.startswith('__main__.'):
            if not self.conf.origin:
                raise RuntimeError('LiveCheck app missing origin argument')
            return self.conf.origin + name[8:]
        return name

    @cached_property
    def bus(self):
        """Topic used for signal communication."""
        return self.topic(self.bus_topic_name, key_type=str, value_type=SignalEvent)

    @cached_property
    def pending_tests(self):
        """Topic used to keep pending test executions."""
        return self.topic(self.test_topic_name, key_type=str, value_type=TestExecution)

    @cached_property
    def reports(self):
        """Topic used to log test reports."""
        return self.topic(self.report_topic_name, key_type=str, value_type=TestReport)