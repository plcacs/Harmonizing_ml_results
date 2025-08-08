from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timedelta
import logging
from math import ceil
from typing import Any
from aiopvapi.helpers.constants import ATTR_NAME, CLOSED_POSITION, MAX_POSITION, MIN_POSITION, MOTION_STOP
from aiopvapi.resources.shade import BaseShade, ShadePosition
from homeassistant.components.cover import ATTR_POSITION, ATTR_TILT_POSITION, CoverDeviceClass, CoverEntity, CoverEntityFeature
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from .const import STATE_ATTRIBUTE_ROOM_NAME
from .coordinator import PowerviewShadeUpdateCoordinator
from .entity import ShadeEntity
from .model import PowerviewConfigEntry, PowerviewDeviceInfo

_LOGGER: logging.Logger
TRANSITION_COMPLETE_DURATION: int
PARALLEL_UPDATES: int
RESYNC_DELAY: int
SCAN_INTERVAL: timedelta

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class PowerViewShadeBase(ShadeEntity, CoverEntity):
    ...

class PowerViewShade(PowerViewShadeBase):
    ...

class PowerViewShadeWithTiltBase(PowerViewShadeBase):
    ...

class PowerViewShadeWithTiltOnClosed(PowerViewShadeWithTiltBase):
    ...

class PowerViewShadeWithTiltAnywhere(PowerViewShadeWithTiltBase):
    ...

class PowerViewShadeTiltOnly(PowerViewShadeWithTiltBase):
    ...

class PowerViewShadeTopDown(PowerViewShadeBase):
    ...

class PowerViewShadeDualRailBase(PowerViewShadeBase):
    ...

class PowerViewShadeTDBUBottom(PowerViewShadeDualRailBase):
    ...

class PowerViewShadeTDBUTop(PowerViewShadeDualRailBase):
    ...

class PowerViewShadeDualOverlappedBase(PowerViewShadeBase):
    ...

class PowerViewShadeDualOverlappedCombined(PowerViewShadeDualOverlappedBase):
    ...

class PowerViewShadeDualOverlappedFront(PowerViewShadeDualOverlappedBase):
    ...

class PowerViewShadeDualOverlappedRear(PowerViewShadeDualOverlappedBase):
    ...

class PowerViewShadeDualOverlappedCombinedTilt(PowerViewShadeDualOverlappedCombined):
    ...

TYPE_TO_CLASSES: dict[int, tuple[type, ...]]

def create_powerview_shade_entity(coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: BaseShade, name_before_refresh: str) -> list[CoverEntity]:
    ...
