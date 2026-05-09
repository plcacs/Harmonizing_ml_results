"""Support for Xiaomi Mi Air Purifier and Xiaomi Mi Air Humidifier."""
from __future__ import annotations
from abc import abstractmethod
import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union
import voluptuous as vol
from homeassistant.components.fan import FanEntity, FanEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN, SERVICE_RESET_FILTER, SERVICE_SET_EXTRA_FEATURES
from .entity import XiaomiCoordinatedMiioEntity
from .typing import ServiceMethodDetails

_LOGGER = logging.getLogger(__name__)
DATA_KEY = 'fan.xiaomi_miio'

class XiaomiGenericDevice(XiaomiCoordinatedMiioEntity, FanEntity):
    """Representation of a generic Xiaomi device."""
    _attr_name: str

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the generic Xiaomi device."""
        ...

    @property
    @abstractmethod
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def preset_modes(self) -> List[str]:
        """Get the list of available preset modes."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the percentage based speed of the fan."""
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the device."""
        ...

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        ...

    async def async_turn_on(self, percentage: Optional[int] = None, preset_mode: Optional[str] = None, **kwargs: Any) -> None:
        """Turn the device on."""
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        ...

class XiaomiGenericAirPurifier(XiaomiGenericDevice):
    """Representation of a generic AirPurifier device."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the generic AirPurifier device."""
        ...

    @property
    def speed_count(self) -> int:
        """Return the number of speeds of the fan supported."""
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        """Get the active preset mode."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

class XiaomiAirPurifier(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Purifier."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the plug switch."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    async def async_set_extra_features(self, features: int = 1) -> None:
        """Set the extra features."""
        ...

    async def async_reset_filter(self) -> None:
        """Reset the filter lifetime and usage."""
        ...

class XiaomiAirPurifierMiot(XiaomiAirPurifier):
    """Representation of a Xiaomi Air Purifier (MiOT protocol)."""
    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

class XiaomiAirPurifierMB4(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Purifier MB4."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize Air Purifier MB4."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

class XiaomiAirFresh(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Fresh."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the miio device."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    async def async_set_extra_features(self, features: int = 1) -> None:
        """Set the extra features."""
        ...

    async def async_reset_filter(self) -> None:
        """Reset the filter lifetime and usage."""
        ...

class XiaomiAirFreshA1(XiaomiGenericAirPurifier):
    """Representation of a Xiaomi Air Fresh A1."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the miio device."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current percentage based speed."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

class XiaomiAirFreshT2017(XiaomiAirFreshA1):
    """Representation of a Xiaomi Air Fresh T2017."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the miio device."""
        ...

class XiaomiGenericFan(XiaomiGenericDevice):
    """Representation of a generic Xiaomi Fan."""
    _attr_translation_key: str

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the fan."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def percentage(self) -> Optional[int]:
        """Return the current speed as a percentage."""
        ...

    @property
    def oscillating(self) -> Optional[bool]:
        """Return whether or not the fan is currently oscillating."""
        ...

    async def async_oscillate(self, oscillating: bool) -> None:
        """Set oscillation."""
        ...

    async def async_set_direction(self, direction: str) -> None:
        """Set the direction of the fan."""
        ...

class XiaomiFan(XiaomiGenericFan):
    """Representation of a Xiaomi Fan."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the fan."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        """Get the active preset mode."""
        ...

    @property
    def preset_modes(self) -> List[str]:
        """Get the list of available preset modes."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

class XiaomiFanP5(XiaomiGenericFan):
    """Representation of a Xiaomi Fan P5."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize the fan."""
        ...

    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

class XiaomiFanMiot(XiaomiGenericFan):
    """Representation of a Xiaomi Fan Miot."""
    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the preset mode of the fan."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

class XiaomiFanZA5(XiaomiFanMiot):
    """Representation of a Xiaomi Fan ZA5."""
    @property
    def operation_mode_class(self) -> Any:
        """Hold operation mode class."""
        ...

class XiaomiFan1C(XiaomiFanMiot):
    """Representation of a Xiaomi Fan 1C (Standing Fan 2 Lite)."""
    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any):
        """Initialize MIOT fan with speed count."""
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        """Set the percentage of the fan."""
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Fan from a config entry."""
    ...

class ServiceMethodDetails:
    """Service method details."""
    method: str
    schema: Optional[vol.Schema]

SERVICE_TO_METHOD: Dict[str, ServiceMethodDetails]