"""Support for Satel Integra zone states- represented as binary sensors."""
from __future__ import annotations
from typing import Any, Dict, Optional

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import (
    CONF_OUTPUTS,
    CONF_ZONE_NAME,
    CONF_ZONE_TYPE,
    CONF_ZONES,
    DATA_SATEL,
    SIGNAL_OUTPUTS_UPDATED,
    SIGNAL_ZONES_UPDATED,
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Satel Integra binary sensor devices."""
    if not discovery_info:
        return
    configured_zones: Dict[str, Any] = discovery_info[CONF_ZONES]
    controller: Any = hass.data[DATA_SATEL]
    devices: list[SatelIntegraBinarySensor] = []
    for zone_num, device_config_data in configured_zones.items():
        zone_type: BinarySensorDeviceClass = device_config_data[CONF_ZONE_TYPE]
        zone_name: str = device_config_data[CONF_ZONE_NAME]
        device = SatelIntegraBinarySensor(
            controller,
            zone_num,
            zone_name,
            zone_type,
            CONF_ZONES,
            SIGNAL_ZONES_UPDATED,
        )
        devices.append(device)
    configured_outputs: Dict[str, Any] = discovery_info[CONF_OUTPUTS]
    for zone_num, device_config_data in configured_outputs.items():
        zone_type: BinarySensorDeviceClass = device_config_data[CONF_ZONE_TYPE]
        zone_name: str = device_config_data[CONF_ZONE_NAME]
        device = SatelIntegraBinarySensor(
            controller,
            zone_num,
            zone_name,
            zone_type,
            CONF_OUTPUTS,
            SIGNAL_OUTPUTS_UPDATED,
        )
        devices.append(device)
    async_add_entities(devices)


class SatelIntegraBinarySensor(BinarySensorEntity):
    """Representation of a Satel Integra binary sensor."""

    _attr_should_poll: bool = False

    def __init__(
        self,
        controller: Any,
        device_number: str,
        device_name: str,
        zone_type: BinarySensorDeviceClass,
        sensor_type: str,
        react_to_signal: str,
    ) -> None:
        """Initialize the binary sensor."""
        self._device_number: str = device_number
        self._attr_unique_id: str = f'satel_{sensor_type}_{device_number}'
        self._name: str = device_name
        self._zone_type: BinarySensorDeviceClass = zone_type
        self._state: int = 0
        self._react_to_signal: str = react_to_signal
        self._satel: Any = controller

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        if self._react_to_signal == SIGNAL_OUTPUTS_UPDATED:
            if self._device_number in self._satel.violated_outputs:
                self._state = 1
            else:
                self._state = 0
        elif self._device_number in self._satel.violated_zones:
            self._state = 1
        else:
            self._state = 0
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, self._react_to_signal, self._devices_updated
            )
        )

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    @property
    def icon(self) -> Optional[str]:
        """Icon for device by its type."""
        if self._zone_type is BinarySensorDeviceClass.SMOKE:
            return 'mdi:fire'
        return None

    @property
    def is_on(self) -> bool:
        """Return true if sensor is on."""
        return self._state == 1

    @property
    def device_class(self) -> Optional[BinarySensorDeviceClass]:
        """Return the class of this sensor, from DEVICE_CLASSES."""
        return self._zone_type

    @callback
    def _devices_updated(self, zones: Dict[str, int]) -> None:
        """Update the zone's state, if needed."""
        if self._device_number in zones and self._state != zones[self._device_number]:
            self._state = zones[self._device_number]
            self.async_write_ha_state()
