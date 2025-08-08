from __future__ import annotations
from dataclasses import dataclass
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription
from homeassistant.const import CONF_DEVICE_ID, CONF_NAME, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import DISPATCHER_KAITERRA, DOMAIN

@dataclass(frozen=True, kw_only=True)
class KaiterraSensorEntityDescription(SensorEntityDescription):
    """Class describing Renault sensor entities."""
    SENSORS: list[KaiterraSensorEntityDescription]

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class KaiterraSensor(SensorEntity):
    """Implementation of a Kaittera sensor."""
    _attr_should_poll: bool = False

    def __init__(self, api: Any, name: str, device_id: str, description: KaiterraSensorEntityDescription) -> None:

    @property
    def _sensor(self) -> dict:

    @property
    def available(self) -> bool:

    @property
    def native_value(self) -> Any:

    @property
    def native_unit_of_measurement(self) -> UnitOfTemperature:

    async def async_added_to_hass(self) -> None:
