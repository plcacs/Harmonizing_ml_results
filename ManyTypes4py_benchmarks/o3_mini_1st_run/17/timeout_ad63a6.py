from __future__ import annotations
import asyncio
import enum
from types import TracebackType
from typing import Any, Optional, List, Dict, Union
from .async_ import run_callback_threadsafe

ZONE_GLOBAL: str = 'global'


class _State(enum.Enum):
    """States of a task."""
    INIT = 'INIT'
    ACTIVE = 'ACTIVE'
    TIMEOUT = 'TIMEOUT'
    EXIT = 'EXIT'


class TimeoutManager:  # Forward declaration for type hints in contexts.
    def __init__(self) -> None:
        """Initialize TimeoutManager."""
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._zones: Dict[str, _ZoneTimeoutManager] = {}
        self._globals: List[_GlobalTaskContext] = []
        self._freezes: List[_GlobalFreezeContext] = []

    @property
    def zones_done(self) -> bool:
        """Return True if all zones are finished."""
        return not bool(self._zones)

    @property
    def freezes_done(self) -> bool:
        """Return True if all freezes are finished."""
        return not self._freezes

    @property
    def zones(self) -> Dict[str, _ZoneTimeoutManager]:
        """Return all Zones."""
        return self._zones

    @property
    def global_tasks(self) -> List[_GlobalTaskContext]:
        """Return all global Tasks."""
        return self._globals

    @property
    def global_freezes(self) -> List[_GlobalFreezeContext]:
        """Return all global Freezes."""
        return self._freezes

    def drop_zone(self, zone_name: str) -> None:
        """Drop a zone out of scope."""
        self._zones.pop(zone_name, None)
        if self._zones:
            return
        for task in self._globals:
            task.zones_done_signal()

    def async_timeout(self, timeout: float, zone_name: str = ZONE_GLOBAL, cool_down: float = 0) -> Union[_GlobalTaskContext, _ZoneTaskContext]:
        """Timeout based on a zone.

        For using as Async Context Manager.
        """
        current_task: Optional[asyncio.Task[Any]] = asyncio.current_task()
        assert current_task is not None
        if zone_name == ZONE_GLOBAL:
            return _GlobalTaskContext(self, current_task, timeout, cool_down)
        if zone_name in self._zones:
            zone = self._zones[zone_name]
        else:
            self._zones[zone_name] = zone = _ZoneTimeoutManager(self, zone_name)
        return _ZoneTaskContext(zone, current_task, timeout)

    def async_freeze(self, zone_name: str = ZONE_GLOBAL) -> Union[_GlobalFreezeContext, _ZoneFreezeContext]:
        """Freeze all timer until job is done.

        For using as Async Context Manager.
        """
        if zone_name == ZONE_GLOBAL:
            return _GlobalFreezeContext(self)
        if zone_name in self._zones:
            zone = self._zones[zone_name]
        else:
            self._zones[zone_name] = zone = _ZoneTimeoutManager(self, zone_name)
        return _ZoneFreezeContext(zone)

    def freeze(self, zone_name: str = ZONE_GLOBAL) -> _GlobalFreezeContext | _ZoneFreezeContext:
        """Freeze all timer until job is done.

        For using as Context Manager.
        """
        return run_callback_threadsafe(self._loop, self.async_freeze, zone_name).result()


class _GlobalFreezeContext:
    """Context manager that freezes the global timeout."""

    def __init__(self, manager: TimeoutManager) -> None:
        """Initialize internal timeout context manager."""
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._manager: TimeoutManager = manager

    async def __aenter__(self) -> _GlobalFreezeContext:
        self._enter()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._exit()
        return None

    def __enter__(self) -> _GlobalFreezeContext:
        self._loop.call_soon_threadsafe(self._enter)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._loop.call_soon_threadsafe(self._exit)
        return None

    def _enter(self) -> None:
        """Run freeze."""
        if self._manager.freezes_done:
            for task in self._manager.global_tasks:
                task.pause()
            for zone in self._manager.zones.values():
                if not zone.freezes_done:
                    continue
                zone.pause()
        self._manager.global_freezes.append(self)

    def _exit(self) -> None:
        """Finish freeze."""
        self._manager.global_freezes.remove(self)
        if not self._manager.freezes_done:
            return
        for task in self._manager.global_tasks:
            task.reset()
        for zone in self._manager.zones.values():
            if not zone.freezes_done:
                continue
            zone.reset()


class _ZoneFreezeContext:
    """Context manager that freezes a zone timeout."""

    def __init__(self, zone: _ZoneTimeoutManager) -> None:
        """Initialize internal timeout context manager."""
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._zone: _ZoneTimeoutManager = zone

    async def __aenter__(self) -> _ZoneFreezeContext:
        self._enter()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._exit()
        return None

    def __enter__(self) -> _ZoneFreezeContext:
        self._loop.call_soon_threadsafe(self._enter)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._loop.call_soon_threadsafe(self._exit)
        return None

    def _enter(self) -> None:
        """Run freeze."""
        if self._zone.freezes_done:
            self._zone.pause()
        self._zone.enter_freeze(self)

    def _exit(self) -> None:
        """Finish freeze."""
        self._zone.exit_freeze(self)
        if not self._zone.freezes_done:
            return
        self._zone.reset()


class _GlobalTaskContext:
    """Context manager that tracks a global task."""

    def __init__(self, manager: TimeoutManager, task: asyncio.Task[Any], timeout: float, cool_down: float) -> None:
        """Initialize internal timeout context manager."""
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._manager: TimeoutManager = manager
        self._task: asyncio.Task[Any] = task
        self._time_left: float = timeout
        self._expiration_time: Optional[float] = None
        self._timeout_handler: Optional[asyncio.TimerHandle] = None
        self._on_wait_task: Optional[asyncio.Task[None]] = None
        self._wait_zone: asyncio.Event = asyncio.Event()
        self._state: _State = _State.INIT
        self._cool_down: float = cool_down
        self._cancelling: int = 0

    async def __aenter__(self) -> _GlobalTaskContext:
        self._manager.global_tasks.append(self)
        self._start_timer()
        self._state = _State.ACTIVE
        self._cancelling = self._task.get_coro().__name__.count("cancel")  # placeholder for self._task.cancelling()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._stop_timer()
        self._manager.global_tasks.remove(self)
        if exc_type is asyncio.CancelledError and self.state is _State.TIMEOUT:
            if self._task.cancelled():  # placeholder for self._task.uncancel() > self._cancelling
                return None
            raise TimeoutError
        self._state = _State.EXIT
        self._wait_zone.set()
        return None

    @property
    def state(self) -> _State:
        """Return state of the Global task."""
        return self._state

    def zones_done_signal(self) -> None:
        """Signal that all zones are done."""
        self._wait_zone.set()

    def _start_timer(self) -> None:
        """Start timeout handler."""
        if self._timeout_handler:
            return
        self._expiration_time = self._loop.time() + self._time_left
        self._timeout_handler = self._loop.call_at(self._expiration_time, self._on_timeout)

    def _stop_timer(self) -> None:
        """Stop zone timer."""
        if self._timeout_handler is None:
            return
        self._timeout_handler.cancel()
        self._timeout_handler = None
        assert self._expiration_time is not None
        self._time_left = self._expiration_time - self._loop.time()

    def _on_timeout(self) -> None:
        """Process timeout."""
        self._state = _State.TIMEOUT
        self._timeout_handler = None
        if not self._manager.zones_done:
            self._on_wait_task = asyncio.create_task(self._on_wait())
        else:
            self._cancel_task()

    def _cancel_task(self) -> None:
        """Cancel own task."""
        if self._task.done():
            return
        self._task.cancel()

    def pause(self) -> None:
        """Pause timers while it freeze."""
        self._stop_timer()

    def reset(self) -> None:
        """Reset timer after freeze."""
        self._start_timer()

    async def _on_wait(self) -> None:
        """Wait until zones are done."""
        await self._wait_zone.wait()
        await asyncio.sleep(self._cool_down)
        self._on_wait_task = None
        if self.state != _State.TIMEOUT:
            return
        self._cancel_task()


class _ZoneTaskContext:
    """Context manager that tracks an active task for a zone."""

    def __init__(self, zone: _ZoneTimeoutManager, task: asyncio.Task[Any], timeout: float) -> None:
        """Initialize internal timeout context manager."""
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._zone: _ZoneTimeoutManager = zone
        self._task: asyncio.Task[Any] = task
        self._state: _State = _State.INIT
        self._time_left: float = timeout
        self._expiration_time: Optional[float] = None
        self._timeout_handler: Optional[asyncio.TimerHandle] = None
        self._cancelling: int = 0

    @property
    def state(self) -> _State:
        """Return state of the Zone task."""
        return self._state

    async def __aenter__(self) -> _ZoneTaskContext:
        self._zone.enter_task(self)
        self._state = _State.ACTIVE
        if self._zone.freezes_done:
            self._start_timer()
        self._cancelling = self._task.get_coro().__name__.count("cancel")  # placeholder for self._task.cancelling()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._zone.exit_task(self)
        self._stop_timer()
        if exc_type is asyncio.CancelledError and self.state is _State.TIMEOUT:
            if self._task.cancelled():  # placeholder for self._task.uncancel() > self._cancelling
                return None
            raise TimeoutError
        self._state = _State.EXIT
        return None

    def _start_timer(self) -> None:
        """Start timeout handler."""
        if self._timeout_handler:
            return
        self._expiration_time = self._loop.time() + self._time_left
        self._timeout_handler = self._loop.call_at(self._expiration_time, self._on_timeout)

    def _stop_timer(self) -> None:
        """Stop zone timer."""
        if self._timeout_handler is None:
            return
        self._timeout_handler.cancel()
        self._timeout_handler = None
        assert self._expiration_time is not None
        self._time_left = self._expiration_time - self._loop.time()

    def _on_timeout(self) -> None:
        """Process timeout."""
        self._state = _State.TIMEOUT
        self._timeout_handler = None
        if self._task.done():
            return
        self._task.cancel()

    def pause(self) -> None:
        """Pause timers while it freeze."""
        self._stop_timer()

    def reset(self) -> None:
        """Reset timer after freeze."""
        self._start_timer()


class _ZoneTimeoutManager:
    """Manage the timeouts for a zone."""

    def __init__(self, manager: TimeoutManager, zone: str) -> None:
        """Initialize internal timeout context manager."""
        self._manager: TimeoutManager = manager
        self._zone: str = zone
        self._tasks: List[_ZoneTaskContext] = []
        self._freezes: List[_ZoneFreezeContext] = []

    def __repr__(self) -> str:
        """Representation of a zone."""
        return f'<{self.name}: {len(self._tasks)} / {len(self._freezes)}>'

    @property
    def name(self) -> str:
        """Return Zone name."""
        return self._zone

    @property
    def active(self) -> bool:
        """Return True if zone is active."""
        return len(self._tasks) > 0 or len(self._freezes) > 0

    @property
    def freezes_done(self) -> bool:
        """Return True if all freeze are done."""
        return len(self._freezes) == 0 and self._manager.freezes_done

    def enter_task(self, task: _ZoneTaskContext) -> None:
        """Start into new Task."""
        self._tasks.append(task)

    def exit_task(self, task: _ZoneTaskContext) -> None:
        """Exit a running Task."""
        self._tasks.remove(task)
        if not self.active:
            self._manager.drop_zone(self.name)

    def enter_freeze(self, freeze: _ZoneFreezeContext) -> None:
        """Start into new freeze."""
        self._freezes.append(freeze)

    def exit_freeze(self, freeze: _ZoneFreezeContext) -> None:
        """Exit a running Freeze."""
        self._freezes.remove(freeze)
        if not self.active:
            self._manager.drop_zone(self.name)

    def pause(self) -> None:
        """Stop timers while it freeze."""
        if not self.active:
            return
        for task in self._tasks:
            task.pause()

    def reset(self) -> None:
        """Reset timer after freeze."""
        if not self.active:
            return
        for task in self._tasks:
            task.reset()