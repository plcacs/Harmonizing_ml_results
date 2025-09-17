"""LiveCheck Signals - Test communication and synchronization."""
import asyncio
import typing
from time import monotonic
from typing import Any, Dict, Generic, Tuple, Type, TypeVar, cast, Optional
from mode import Seconds, want_seconds
from faust.models import maybe_model
from .exceptions import TestTimeout
from .locals import current_test_stack
from .models import SignalEvent

if typing.TYPE_CHECKING:
    from .case import Case as _Case
else:

    class _Case:
        ...


__all__ = ['BaseSignal', 'Signal']
VT = TypeVar('VT')


class BaseSignal(Generic[VT]):
    """Generic base class for signals."""

    def __init__(self, name: str = '', case: Optional[_Case] = None, index: int = -1) -> None:
        self.name: str = name
        self.case: _Case = cast(_Case, case)
        self.index: int = index

    async def send(self, value: Any = None, *, key: Any = None, force: bool = False) -> None:
        """Notify test that this signal is now complete."""
        raise NotImplementedError()

    async def wait(self, *, key: Any = None, timeout: Optional[Seconds] = None) -> VT:
        """Wait for signal to be completed."""
        raise NotImplementedError()

    async def resolve(self, key: Any, event: SignalEvent) -> None:
        """Resolve signal with value."""
        self._set_current_value(key, event)
        self._wakeup_resolvers()

    def __set_name__(self, owner: Any, name: str) -> None:
        if not self.name:
            self.name = name

    def _wakeup_resolvers(self) -> None:
        self.case.app._can_resolve.set()

    async def _wait_for_resolved(self, *, timeout: Optional[Seconds] = None) -> None:
        app = self.case.app
        app._can_resolve.clear()
        await app.wait(app._can_resolve, timeout=timeout)

    def _get_current_value(self, key: Any) -> SignalEvent:
        return self.case.app._resolved_signals[self._index_key(key)]

    def _index_key(self, key: Any) -> Tuple[str, str, Any]:
        return (self.name, self.case.name, key)

    def _set_current_value(self, key: Any, event: SignalEvent) -> None:
        self.case.app._resolved_signals[self._index_key(key)] = event

    def clone(self, **kwargs: Any) -> 'BaseSignal[VT]':
        """Clone this signal using keyword arguments."""
        return type(self)(**{**self._asdict(), **kwargs})

    def _asdict(self, **kwargs: Any) -> Dict[str, Any]:
        return {'name': self.name, 'case': self.case, 'index': self.index}

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.name}>'


class Signal(BaseSignal[VT]):
    """Signal for test case using Kafka.

    Used to wait for something to happen elsewhere.
    """

    async def send(self, value: Any = None, *, key: Any = None, force: bool = False) -> None:
        """Notify test that this signal is now complete."""
        current_test = current_test_stack.top
        if current_test is None:
            if not force:
                return
            assert key
        else:
            key = key if key is not None else current_test.id
        await self.case.app.bus.send(
            key=key,
            value=SignalEvent(
                signal_name=self.name,
                case_name=self.case.name,
                key=key,
                value=value
            )
        )

    async def wait(self, *, key: Any = None, timeout: Optional[Seconds] = None) -> VT:
        """Wait for signal to be completed."""
        runner = self.case.current_execution
        if runner is None:
            raise RuntimeError('No test executing.')
        test = runner.test
        assert test
        k: Any = test.id if key is None else key
        timeout_s: Seconds = want_seconds(timeout)
        await runner.on_signal_wait(self, timeout=timeout_s)
        time_start: float = monotonic()
        event: SignalEvent = await self._wait_for_message_by_key(key=k, timeout=timeout_s)
        time_end: float = monotonic()
        await runner.on_signal_received(self, time_start=time_start, time_end=time_end)
        self._verify_event(event, k, self.name, self.case.name)
        return cast(VT, maybe_model(event.value))

    def _verify_event(self, ev: SignalEvent, key: Any, name: str, case: str) -> None:
        assert ev.key == key, f'{ev.key!r} == {key!r}'
        assert ev.signal_name == name, f'{ev.signal_name!r} == {name!r}'
        assert ev.case_name == case, f'{ev.case_name!r} == {case!r}'

    async def _wait_for_message_by_key(self, key: Any, *, timeout: Optional[Seconds] = None, max_interval: float = 2.0) -> SignalEvent:
        app = self.case.app
        time_start: float = monotonic()
        remaining: Optional[float] = timeout
        try:
            return self._get_current_value(key)
        except KeyError:
            pass
        while not app.should_stop:
            if remaining is not None:
                remaining = remaining - (monotonic() - time_start)
            try:
                if remaining is not None and remaining <= 0.0:
                    try:
                        return self._get_current_value(key)
                    except KeyError:
                        raise asyncio.TimeoutError() from None
                max_wait: Optional[float] = None
                if remaining is not None:
                    max_wait = min(remaining, max_interval)
                await self._wait_for_resolved(timeout=max_wait)
            except asyncio.TimeoutError:
                msg: str = f'Timed out waiting for signal {self.name} ({timeout})'
                raise TestTimeout(msg) from None
            if app.should_stop:
                break
            try:
                val: SignalEvent = self._get_current_value(key)
                return val
            except KeyError:
                pass
        raise asyncio.CancelledError()