from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, kw_only=True)
class XiaomiMiioSwitchDescription(SwitchEntityDescription):
    """A class that describes switch entities."""
    available_with_device_off: bool = True

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the switch from a config entry."""
    ...

async def async_setup_coordinated_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the coordinated switch from a config entry."""
    ...

async def async_setup_other_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the other type switch from a config entry."""
    ...

class XiaomiGenericCoordinatedSwitch(XiaomiCoordinatedMiioEntity, SwitchEntity):
    """Representation of a Xiaomi Plug Generic."""
    ...

class XiaomiGatewaySwitch(XiaomiGatewayDevice, SwitchEntity):
    """Representation of a XiaomiGatewaySwitch."""
    ...

class XiaomiPlugGenericSwitch(XiaomiMiioEntity, SwitchEntity):
    """Representation of a Xiaomi Plug Generic."""
    ...

class XiaomiPowerStripSwitch(XiaomiPlugGenericSwitch):
    """Representation of a Xiaomi Power Strip."""
    ...

class ChuangMiPlugSwitch(XiaomiPlugGenericSwitch):
    """Representation of a Chuang Mi Plug V1 and V3."""
    ...

class XiaomiAirConditioningCompanionSwitch(XiaomiPlugGenericSwitch):
    """Representation of a Xiaomi AirConditioning Companion."""
    ...
