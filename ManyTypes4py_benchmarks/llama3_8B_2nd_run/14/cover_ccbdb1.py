from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timedelta
import logging
from math import ceil
from typing import Any

async def async_setup_entry(hass: HomeAssistant, entry: AddConfigEntryEntitiesCallback, async_add_entities: Callable[[Iterable[CoverEntity]], None]) -> None:
    ...

class PowerViewShadeBase(ShadeEntity, CoverEntity):
    """Representation of a powerview shade."""
    _attr_device_class: CoverDeviceClass
    _attr_supported_features: CoverEntityFeature

    def __init__(self, coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: Shade, name: str) -> None:
        ...

class PowerViewShade(PowerViewShadeBase):
    """Represent a standard shade."""
    _attr_name: str | None

class PowerViewShadeWithTiltBase(PowerViewShadeBase):
    """Representation for PowerView shades with tilt capabilities."""
    _attr_name: str | None

    def __init__(self, coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: Shade, name: str) -> None:
        ...

class PowerViewShadeWithTiltOnClosed(PowerViewShadeWithTiltBase):
    """Representation of a PowerView shade with tilt when closed capabilities."""
    _attr_name: str | None

class PowerViewShadeWithTiltAnywhere(PowerViewShadeWithTiltBase):
    """Representation of a PowerView shade with tilt anywhere capabilities."""
    ...

class PowerViewShadeTiltOnly(PowerViewShadeWithTiltBase):
    """Representation of a shade with tilt only capability, no move."""
    ...

class PowerViewShadeTopDown(PowerViewShadeBase):
    """Representation of a shade that lowers from the roof to the floor."""
    _attr_name: str | None

class PowerViewShadeDualRailBase(PowerViewShadeBase):
    """Represent a shade that has top/down bottom/up capabilities."""
    ...

class PowerViewShadeDualOverlappedBase(PowerViewShadeBase):
    """Represent a shade that has a front sheer and rear opaque panel."""
    ...

class PowerViewShadeDualOverlappedCombined(PowerViewShadeDualOverlappedBase):
    """Represent a shade that has a front sheer and rear opaque panel."""
    _attr_translation_key: str

    def __init__(self, coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: Shade, name: str) -> None:
        ...

class PowerViewShadeDualOverlappedCombinedTilt(PowerViewShadeDualOverlappedCombined):
    """Represent a shade that has a front sheer and rear opaque panel."""
    ...

TYPE_TO_CLASSES = {0: (PowerViewShade,), 1: (PowerViewShadeWithTiltOnClosed,), 2: (PowerViewShadeWithTiltAnywhere,), 3: (PowerViewShade,), 4: (PowerViewShadeWithTiltAnywhere,), 5: (PowerViewShadeTiltOnly,), 6: (PowerViewShadeTopDown,), 7: (PowerViewShadeTDBUTop, PowerViewShadeTDBUBottom), 8: (PowerViewShadeDualOverlappedCombined, PowerViewShadeDualOverlappedFront, PowerViewShadeDualOverlappedRear), 9: (PowerViewShadeDualOverlappedCombinedTilt, PowerViewShadeDualOverlappedFront, PowerViewShadeDualOverlappedRear), 10: (PowerViewShadeDualOverlappedCombinedTilt, PowerViewShadeDualOverlappedFront, PowerViewShadeDualOverlappedRear), 11: (PowerViewShadeDualOverlappedCombined, PowerViewShadeDualOverlappedFront, PowerViewShadeDualOverlappedRear)}

def create_powerview_shade_entity(coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: Shade, name_before_refresh: str) -> list[PowerViewShadeBase]:
    ...

async def async_added_to_hass(self: PowerViewShadeBase) -> None:
    ...

async def async_will_remove_from_hass(self: PowerViewShadeBase) -> None:
    ...
