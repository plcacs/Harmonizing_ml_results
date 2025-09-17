#!/usr/bin/env python
"""LiveCheck - Faust Application."""
import asyncio
from datetime import timedelta
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, cast

import faust
from faust.app.base import SCAN_CATEGORIES
from faust.sensors.base import Sensor
from faust.types import AgentT, AppT, EventT, StreamT, TP, TopicT
from mode.signals import BaseSignalT
from mode.utils.compat import want_bytes
from mode.utils.objects import annotations, cached_property, qualname
from mode.utils.times import Seconds
from faust.utils import venusian

from . import patches
from .case import Case
from .exceptions import LiveCheckError
from .locals import current_test, current_test_stack
from .models import SignalEvent, TestExecution, TestReport
from .signals import BaseSignal, Signal

__all__ = ['LiveCheck']

SCAN_CASE: str = 'livecheck.case'
_Case = Case
patches.patch_all()


class LiveCheckSensor(Sensor):
    def on_stream_event_in(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
    ) -> None:
        """Call when stream starts processing event."""
        test: Optional[TestExecution] = TestExecution.from_headers(event.headers)
        if test is not None:
            stream.current_test = test
            current_test_stack.push_without_automatic_cleanup(test)
        return None

    def on_stream_event_out(
        self,
        tp: TP,
        offset: int,
        stream: StreamT,
        event: EventT,
        state: Optional[Any] = None,
    ) -> None:
        """Call when stream is finished handling event."""
        has_active_test: Any = getattr(stream, 'current_test', None)
        if has_active_test:
            stream.current_test = None
            current_test_stack.pop()
        return None


class LiveCheck(faust.App):
    """LiveCheck application."""
    SCAN_CATEGORIES: ClassVar[List[str]] = list(SCAN_CATEGORIES) + [SCAN_CASE]
    Signal: Type[Signal] = Signal
    Case: Type[Case] = _Case
    bus_concurrency: int = 30
    test_concurrency: int = 100
    send_reports: bool = True
    test_topic_name: str = 'livecheck'
    bus_topic_name: str = 'livecheck-bus'
    report_topic_name: str = 'livecheck-report'

    @classmethod
    def for_app(
        cls: Type["LiveCheck"],
        app: AppT,
        *,
        prefix: str = 'livecheck-',
        web_port: int = 9999,
        test_topic_name: Optional[str] = None,
        bus_topic_name: Optional[str] = None,
        report_topic_name: Optional[str] = None,
        bus_concurrency: Optional[int] = None,
        test_concurrency: Optional[int] = None,
        send_reports: Optional[bool] = None,
        **kwargs: Any,
    ) -> "LiveCheck":
        """Create LiveCheck application targeting specific app.

        The target app will be used to configure the LiveCheck app.
        """
        app_id, passed_kwargs = app._default_options
        livecheck_id: str = f'{prefix}{app_id}'
        override: Dict[str, Any] = {
            'web_port': web_port,
            'test_topic_name': test_topic_name,
            'bus_topic_name': bus_topic_name,
            'report_topic_name': report_topic_name,
            'bus_concurrency': bus_concurrency,
            'test_concurrency': test_concurrency,
            'send_reports': send_reports,
            **kwargs,
        }
        options: Dict[str, Any] = {**passed_kwargs, **override}
        livecheck_app: LiveCheck = cls(livecheck_id, **options)
        livecheck_app._contribute_to_app(app)
        return livecheck_app

    def _contribute_to_app(self, app: AppT) -> None:
        from .patches.aiohttp import LiveCheckMiddleware
        web_app: Any = app.web.web_app
        web_app.middlewares.append(LiveCheckMiddleware())
        app.sensors.add(LiveCheckSensor())
        app.livecheck = self

    def __init__(
        self,
        id: str,
        *,
        test_topic_name: Optional[str] = None,
        bus_topic_name: Optional[str] = None,
        report_topic_name: Optional[str] = None,
        bus_concurrency: Optional[int] = None,
        test_concurrency: Optional[int] = None,
        send_reports: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
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
        self.cases: Dict[str, Case] = {}
        self._resolved_signals: Dict[str, Any] = {}
        patches.patch_all()
        self._apply_monkeypatches()
        self._connect_signals()

    @property
    def current_test(self) -> Optional[TestExecution]:
        """Return the current test context (if any)."""
        return current_test()

    @cached_property
    def _can_resolve(self) -> asyncio.Event:
        return asyncio.Event()

    def _apply_monkeypatches(self) -> None:
        patches.patch_all()

    def _connect_signals(self) -> None:
        AppT.on_produce_message.connect(self.on_produce_attach_test_headers)

    def on_produce_attach_test_headers(
        self,
        sender: Any,
        key: Any = None,
        value: Any = None,
        partition: Any = None,
        timestamp: Any = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
        signal: Any = None,
        **kwargs: Any,
    ) -> None:
        """Attach test headers to Kafka produce requests."""
        test: Optional[TestExecution] = current_test()
        if test is not None:
            if headers is None:
                raise TypeError('Produce request missing headers list')
            headers.extend([(k, want_bytes(v)) for k, v in test.as_headers().items()])

    def case(
        self,
        *,
        name: Optional[str] = None,
        probability: Optional[float] = None,
        warn_stalled_after: timedelta = timedelta(minutes=30),
        active: Optional[bool] = None,
        test_expires: Optional[Any] = None,
        frequency: Optional[Any] = None,
        max_history: Optional[int] = None,
        max_consecutive_failures: Optional[int] = None,
        url_timeout_total: Optional[Seconds] = None,
        url_timeout_connect: Optional[Seconds] = None,
        url_error_retries: Optional[int] = None,
        url_error_delay_min: Optional[Seconds] = None,
        url_error_delay_backoff: Optional[Seconds] = None,
        url_error_delay_max: Optional[Seconds] = None,
        base: Type[Case] = Case,
    ) -> Callable[[Type[Any]], Any]:
        """Decorate class to be used as a test case.

        Returns:
            :class:`faust.livecheck.Case`.
        """
        base_case: Type[Case] = base

        def _inner(cls: Type[Any]) -> Any:
            case_cls: Type[Any] = type(
                cls.__name__,
                (cls, base_case),
                {'__module__': cls.__module__, 'app': self},
            )
            signal_types: Dict[str, Any] = dict(self._extract_signals(case_cls, base_case))
            signals: List[Any] = []
            for i, (attr_name, attr_type) in enumerate(signal_types.items()):
                signal_obj: Any = getattr(case_cls, attr_name, None)
                if signal_obj is None:
                    signal_obj = attr_type(name=attr_name, index=i + 1)
                    setattr(case_cls, attr_name, signal_obj)
                    signals.append(signal_obj)
                else:
                    signal_obj.index = i + 1
            case_obj: Case = self.add_case(
                case_cls(
                    app=self,
                    name=self._prepare_case_name(name or qualname(cls)),
                    active=active,
                    probability=probability,
                    warn_stalled_after=warn_stalled_after,
                    signals=signals,
                    test_expires=test_expires,
                    frequency=frequency,
                    max_history=max_history,
                    max_consecutive_failures=max_consecutive_failures,
                    url_timeout_total=url_timeout_total,
                    url_timeout_connect=url_timeout_connect,
                    url_error_retries=url_error_retries,
                    url_error_delay_min=url_error_delay_min,
                    url_error_delay_backoff=url_error_delay_backoff,
                    url_error_delay_max=url_error_delay_max,
                )
            )
            venusian.attach(cast(Callable, case_obj), category=SCAN_CASE)
            return case_obj

        return _inner

    def _extract_signals(
        self,
        case_cls: Type[Any],
        base_case: Type[Any],
    ) -> Iterable[Tuple[str, Any]]:
        fields, defaults = annotations(
            case_cls, stop=base_case, skip_classvar=True, localns={case_cls.__name__: case_cls}
        )
        for attr_name, attr_type in fields.items():
            actual_type: Any = getattr(attr_type, '__origin__', attr_type)
            if actual_type is None:
                actual_type = attr_type
            try:
                if issubclass(actual_type, BaseSignal):
                    yield (attr_name, attr_type)
            except TypeError:
                pass

    def add_case(self, case: Case) -> Case:
        """Add and register new test case."""
        self.cases[case.name] = case
        return case

    async def post_report(self, report: TestReport) -> None:
        """Publish test report to reporting topic."""
        key: Optional[str] = None
        if report.test is not None:
            key = report.test.id
        await self.reports.send(key=key, value=report)

    async def on_start(self) -> None:
        """Call when LiveCheck application starts."""
        await super().on_start()
        self._install_bus_agent()
        self._install_test_execution_agent()

    async def on_started(self) -> None:
        """Call when LiveCheck application is fully started."""
        await super().on_started()
        for case in self.cases.values():
            await self.add_runtime_dependency(case)

    def _install_bus_agent(self) -> AgentT:
        return self.agent(channel=self.bus, concurrency=self.bus_concurrency)(self._populate_signals)

    def _install_test_execution_agent(self) -> AgentT:
        return self.agent(channel=self.pending_tests, concurrency=self.test_concurrency)(self._execute_tests)

    async def _populate_signals(self, events: StreamT) -> None:
        async for test_id, event in events.items():
            event.case_name = self._prepare_case_name(event.case_name)
            try:
                case_obj: Case = self.cases[event.case_name]
            except KeyError:
                self.log.error('Received signal %r for unregistered case %r', event, (test_id, event.case_name))
            else:
                await case_obj.resolve_signal(test_id, event)

    async def _execute_tests(self, tests: StreamT) -> None:
        async for test_id, test in tests.items():
            test.case_name = self._prepare_case_name(test.case_name)
            try:
                case_obj: Case = self.cases[test.case_name]
            except KeyError:
                self.log.error('Unregistered test case %r with id %r: %r', test.case_name, test_id, test)
            else:
                try:
                    await case_obj.execute(test)
                except LiveCheckError:
                    pass

    def _prepare_case_name(self, name: str) -> str:
        if name.startswith('__main__.'):
            if not self.conf.origin:
                raise RuntimeError('LiveCheck app missing origin argument')
            return self.conf.origin + name[8:]
        return name

    @cached_property
    def bus(self) -> TopicT[SignalEvent]:
        """Topic used for signal communication."""
        return self.topic(self.bus_topic_name, key_type=str, value_type=SignalEvent)

    @cached_property
    def pending_tests(self) -> TopicT[TestExecution]:
        """Topic used to keep pending test executions."""
        return self.topic(self.test_topic_name, key_type=str, value_type=TestExecution)

    @cached_property
    def reports(self) -> TopicT[TestReport]:
        """Topic used to log test reports."""
        return self.topic(self.report_topic_name, key_type=str, value_type=TestReport)