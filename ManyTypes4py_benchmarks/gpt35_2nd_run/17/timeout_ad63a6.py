from __future__ import annotations
import asyncio
import enum
from types import TracebackType
from typing import Any

ZONE_GLOBAL: str = 'global'

class _State(enum.Enum):
    INIT: str = 'INIT'
    ACTIVE: str = 'ACTIVE'
    TIMEOUT: str = 'TIMEOUT'
    EXIT: str = 'EXIT'

class _GlobalFreezeContext:
    def __init__(self, manager: TimeoutManager) -> None:
    async def __aenter__(self) -> _GlobalFreezeContext:
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
    def __enter__(self) -> _GlobalFreezeContext:
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
    def _enter(self) -> None
    def _exit(self) -> None

class _ZoneFreezeContext:
    def __init__(self, zone: _ZoneTimeoutManager) -> None:
    async def __aenter__(self) -> _ZoneFreezeContext:
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
    def __enter__(self) -> _ZoneFreezeContext:
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
    def _enter(self) -> None
    def _exit(self) -> None

class _GlobalTaskContext:
    def __init__(self, manager: TimeoutManager, task: Any, timeout: float, cool_down: float) -> None:
    async def __aenter__(self) -> _GlobalTaskContext:
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
    @property
    def state(self) -> _State:
    def zones_done_signal(self) -> None
    def _start_timer(self) -> None
    def _stop_timer(self) -> None
    def _on_timeout(self) -> None
    def _cancel_task(self) -> None
    def pause(self) -> None
    def reset(self) -> None
    async def _on_wait(self) -> None

class _ZoneTaskContext:
    def __init__(self, zone: _ZoneTimeoutManager, task: Any, timeout: float) -> None:
    @property
    def state(self) -> _State
    async def __aenter__(self) -> _ZoneTaskContext
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
    def _start_timer(self) -> None
    def _stop_timer(self) -> None
    def _on_timeout(self) -> None
    def pause(self) -> None
    def reset(self) -> None

class _ZoneTimeoutManager:
    def __init__(self, manager: TimeoutManager, zone: str) -> None:
    def __repr__(self) -> str
    @property
    def name(self) -> str
    @property
    def active(self) -> bool
    @property
    def freezes_done(self) -> bool
    def enter_task(self, task: _ZoneTaskContext) -> None
    def exit_task(self, task: _ZoneTaskContext) -> None
    def enter_freeze(self, freeze: _ZoneFreezeContext) -> None
    def exit_freeze(self, freeze: _ZoneFreezeContext) -> None
    def pause(self) -> None
    def reset(self) -> None

class TimeoutManager:
    def __init__(self) -> None
    @property
    def zones_done(self) -> bool
    @property
    def freezes_done(self) -> bool
    @property
    def zones(self) -> dict[str, _ZoneTimeoutManager]
    @property
    def global_tasks(self) -> list[_GlobalTaskContext]
    @property
    def global_freezes(self) -> list[_GlobalFreezeContext]
    def drop_zone(self, zone_name: str) -> None
    def async_timeout(self, timeout: float, zone_name: str = ZONE_GLOBAL, cool_down: float = 0) -> _GlobalTaskContext
    def async_freeze(self, zone_name: str = ZONE_GLOBAL) -> _GlobalFreezeContext
    def freeze(self, zone_name: str = ZONE_GLOBAL) -> _GlobalFreezeContext
