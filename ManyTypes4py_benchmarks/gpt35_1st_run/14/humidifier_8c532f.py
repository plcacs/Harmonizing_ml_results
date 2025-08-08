from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import CommandClass
from zwave_js_server.const.command_class.humidity_control import HUMIDITY_CONTROL_SETPOINT_PROPERTY, HumidityControlMode, HumidityControlSetpointType
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.value import Value as ZwaveValue
from homeassistant.components.humidifier import DEFAULT_MAX_HUMIDITY, DEFAULT_MIN_HUMIDITY, DOMAIN as HUMIDIFIER_DOMAIN, HumidifierDeviceClass, HumidifierEntity, HumidifierEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DATA_CLIENT, DOMAIN
from .discovery import ZwaveDiscoveryInfo
from .entity import ZWaveBaseEntity

@dataclass(frozen=True, kw_only=True)
class ZwaveHumidifierEntityDescription(HumidifierEntityDescription):
    """A class that describes the humidifier or dehumidifier entity."""

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class ZWaveHumidifier(ZWaveBaseEntity, HumidifierEntity):
    """Representation of a Z-Wave Humidifier or Dehumidifier."""
    
    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo, description: ZwaveHumidifierEntityDescription) -> None:

    @property
    def is_on(self) -> Optional[bool]:

    def _supports_inverse_mode(self) -> bool:

    async def async_turn_on(self, **kwargs) -> None:

    async def async_turn_off(self, **kwargs) -> None:

    @property
    def target_humidity(self) -> Optional[int]:

    async def async_set_humidity(self, humidity: int) -> None:

    @property
    def min_humidity(self) -> int:

    @property
    def max_humidity(self) -> int:
