"""Advanced timeout handling.

Set of helper classes to handle timeouts of tasks with advanced options
like zones and freezing of timeouts.
"""
from __future__ import annotations
import asyncio
import enum
from types import TracebackType
from typing import Any, Self
from .async_ import run_callback_threadsafe
ZONE_GLOBAL = 'global'

class _State(enum.Enum):
    """States of a task."""
    INIT = 'INIT'
    ACTIVE = 'ACTIVE'
    TIMEOUT = 'TIMEOUT'
    EXIT = 'EXIT'

class _GlobalFreezeContext:
    """Context manager that freezes the global timeout."""

    def __init__(self, manager):
        """Initialize internal timeout context manager."""
        self._loop = asyncio.get_running_loop()
        self._manager = manager

    async def __aenter__(self):
        self._enter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit()
        return None

    def __enter__(self):
        self._loop.call_soon_threadsafe(self._enter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.call_soon_threadsafe(self._exit)
        return None

    def _enter(self):
        """Run freeze."""
        if self._manager.freezes_done:
            for task in self._manager.global_tasks:
                task.pause()
            for zone in self._manager.zones.values():
                if not zone.freezes_done:
                    continue
                zone.pause()
        self._manager.global_freezes.append(self)

    def _exit(self):
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

    def __init__(self, zone):
        """Initialize internal timeout context manager."""
        self._loop = asyncio.get_running_loop()
        self._zone = zone

    async def __aenter__(self):
        self._enter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit()
        return None

    def __enter__(self):
        self._loop.call_soon_threadsafe(self._enter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.call_soon_threadsafe(self._exit)
        return None

    def _enter(self):
        """Run freeze."""
        if self._zone.freezes_done:
            self._zone.pause()
        self._zone.enter_freeze(self)

    def _exit(self):
        """Finish freeze."""
        self._zone.exit_freeze(self)
        if not self._zone.freezes_done:
            return
        self._zone.reset()

class _GlobalTaskContext:
    """Context manager that tracks a global task."""

    def __init__(self, manager, task, timeout, cool_down):
        """Initialize internal timeout context manager."""
        self._loop = asyncio.get_running_loop()
        self._manager = manager
        self._task = task
        self._time_left = timeout
        self._expiration_time = None
        self._timeout_handler = None
        self._on_wait_task = None
        self._wait_zone = asyncio.Event()
        self._state = _State.INIT
        self._cool_down = cool_down
        self._cancelling = 0

    async def __aenter__(self):
        self._manager.global_tasks.append(self)
        self._start_timer()
        self._state = _State.ACTIVE
        self._cancelling = self._task.cancelling()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stop_timer()
        self._manager.global_tasks.remove(self)
        if exc_type is asyncio.CancelledError and self.state is _State.TIMEOUT:
            if self._task.uncancel() > self._cancelling:
                return None
            raise TimeoutError
        self._state = _State.EXIT
        self._wait_zone.set()
        return None

    @property
    def state(self):
        """Return state of the Global task."""
        return self._state

    def zones_done_signal(self):
        """Signal that all zones are done."""
        self._wait_zone.set()

    def _start_timer(self):
        """Start timeout handler."""
        if self._timeout_handler:
            return
        self._expiration_time = self._loop.time() + self._time_left
        self._timeout_handler = self._loop.call_at(self._expiration_time, self._on_timeout)

    def _stop_timer(self):
        """Stop zone timer."""
        if self._timeout_handler is None:
            return
        self._timeout_handler.cancel()
        self._timeout_handler = None
        assert self._expiration_time
        self._time_left = self._expiration_time - self._loop.time()

    def _on_timeout(self):
        """Process timeout."""
        self._state = _State.TIMEOUT
        self._timeout_handler = None
        if not self._manager.zones_done:
            self._on_wait_task = asyncio.create_task(self._on_wait())
        else:
            self._cancel_task()

    def _cancel_task(self):
        """Cancel own task."""
        if self._task.done():
            return
        self._task.cancel('Global task timeout')

    def pause(self):
        """Pause timers while it freeze."""
        self._stop_timer()

    def reset(self):
        """Reset timer after freeze."""
        self._start_timer()

    async def _on_wait(self):
        """Wait until zones are done."""
        await self._wait_zone.wait()
        await asyncio.sleep(self._cool_down)
        self._on_wait_task = None
        if self.state != _State.TIMEOUT:
            return
        self._cancel_task()

class _ZoneTaskContext:
    """Context manager that tracks an active task for a zone."""

    def __init__(self, zone, task, timeout):
        """Initialize internal timeout context manager."""
        self._loop = asyncio.get_running_loop()
        self._zone = zone
        self._task = task
        self._state = _State.INIT
        self._time_left = timeout
        self._expiration_time = None
        self._timeout_handler = None
        self._cancelling = 0

    @property
    def state(self):
        """Return state of the Zone task."""
        return self._state

    async def __aenter__(self):
        self._zone.enter_task(self)
        self._state = _State.ACTIVE
        if self._zone.freezes_done:
            self._start_timer()
        self._cancelling = self._task.cancelling()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._zone.exit_task(self)
        self._stop_timer()
        if exc_type is asyncio.CancelledError and self.state is _State.TIMEOUT:
            if self._task.uncancel() > self._cancelling:
                return None
            raise TimeoutError
        self._state = _State.EXIT
        return None

    def _start_timer(self):
        """Start timeout handler."""
        if self._timeout_handler:
            return
        self._expiration_time = self._loop.time() + self._time_left
        self._timeout_handler = self._loop.call_at(self._expiration_time, self._on_timeout)

    def _stop_timer(self):
        """Stop zone timer."""
        if self._timeout_handler is None:
            return
        self._timeout_handler.cancel()
        self._timeout_handler = None
        assert self._expiration_time
        self._time_left = self._expiration_time - self._loop.time()

    def _on_timeout(self):
        """Process timeout."""
        self._state = _State.TIMEOUT
        self._timeout_handler = None
        if self._task.done():
            return
        self._task.cancel('Zone timeout')

    def pause(self):
        """Pause timers while it freeze."""
        self._stop_timer()

    def reset(self):
        """Reset timer after freeze."""
        self._start_timer()

class _ZoneTimeoutManager:
    """Manage the timeouts for a zone."""

    def __init__(self, manager, zone):
        """Initialize internal timeout context manager."""
        self._manager = manager
        self._zone = zone
        self._tasks = []
        self._freezes = []

    def __repr__(self):
        """Representation of a zone."""
        return f'<{self.name}: {len(self._tasks)} / {len(self._freezes)}>'

    @property
    def name(self):
        """Return Zone name."""
        return self._zone

    @property
    def active(self):
        """Return True if zone is active."""
        return len(self._tasks) > 0 or len(self._freezes) > 0

    @property
    def freezes_done(self):
        """Return True if all freeze are done."""
        return len(self._freezes) == 0 and self._manager.freezes_done

    def enter_task(self, task):
        """Start into new Task."""
        self._tasks.append(task)

    def exit_task(self, task):
        """Exit a running Task."""
        self._tasks.remove(task)
        if not self.active:
            self._manager.drop_zone(self.name)

    def enter_freeze(self, freeze):
        """Start into new freeze."""
        self._freezes.append(freeze)

    def exit_freeze(self, freeze):
        """Exit a running Freeze."""
        self._freezes.remove(freeze)
        if not self.active:
            self._manager.drop_zone(self.name)

    def pause(self):
        """Stop timers while it freeze."""
        if not self.active:
            return
        for task in self._tasks:
            task.pause()

    def reset(self):
        """Reset timer after freeze."""
        if not self.active:
            return
        for task in self._tasks:
            task.reset()

class TimeoutManager:
    """Class to manage timeouts over different zones.

    Manages both global and zone based timeouts.
    """

    def __init__(self):
        """Initialize TimeoutManager."""
        self._loop = asyncio.get_running_loop()
        self._zones = {}
        self._globals = []
        self._freezes = []

    @property
    def zones_done(self):
        """Return True if all zones are finished."""
        return not bool(self._zones)

    @property
    def freezes_done(self):
        """Return True if all freezes are finished."""
        return not self._freezes

    @property
    def zones(self):
        """Return all Zones."""
        return self._zones

    @property
    def global_tasks(self):
        """Return all global Tasks."""
        return self._globals

    @property
    def global_freezes(self):
        """Return all global Freezes."""
        return self._freezes

    def drop_zone(self, zone_name):
        """Drop a zone out of scope."""
        self._zones.pop(zone_name, None)
        if self._zones:
            return
        for task in self._globals:
            task.zones_done_signal()

    def async_timeout(self, timeout, zone_name=ZONE_GLOBAL, cool_down=0):
        """Timeout based on a zone.

        For using as Async Context Manager.
        """
        current_task = asyncio.current_task()
        assert current_task
        if zone_name == ZONE_GLOBAL:
            return _GlobalTaskContext(self, current_task, timeout, cool_down)
        if zone_name in self.zones:
            zone = self.zones[zone_name]
        else:
            self.zones[zone_name] = zone = _ZoneTimeoutManager(self, zone_name)
        return _ZoneTaskContext(zone, current_task, timeout)

    def async_freeze(self, zone_name=ZONE_GLOBAL):
        """Freeze all timer until job is done.

        For using as Async Context Manager.
        """
        if zone_name == ZONE_GLOBAL:
            return _GlobalFreezeContext(self)
        if zone_name in self.zones:
            zone = self.zones[zone_name]
        else:
            self.zones[zone_name] = zone = _ZoneTimeoutManager(self, zone_name)
        return _ZoneFreezeContext(zone)

    def freeze(self, zone_name=ZONE_GLOBAL):
        """Freeze all timer until job is done.

        For using as Context Manager.
        """
        return run_callback_threadsafe(self._loop, self.async_freeze, zone_name).result()