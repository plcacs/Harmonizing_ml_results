"""Interfaces with Egardia/Woonveilig alarm control panel."""
from __future__ import annotations
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import ATTR_DISCOVER_DEVICES, EGARDIA_DEVICE

EGARDIA_TYPE_TO_DEVICE_CLASS: dict[str, BinarySensorDeviceClass] = {
    'IR Sensor': BinarySensorDeviceClass.MOTION,
    'Door Contact': BinarySensorDeviceClass.OPENING,
    'IR': BinarySensorDeviceClass.MOTION
}

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Initialize the platform."""
    if discovery_info is None or discovery_info[ATTR_DISCOVER_DEVICES] is None:
        return
    disc_info = discovery_info[ATTR_DISCOVER_DEVICES]
    async_add_entities(
        (
            EgardiaBinarySensor(
                sensor_id=disc_info[sensor]['id'],
                name=disc_info[sensor]['name'],
                egardia_system=hass.data[EGARDIA_DEVICE],
                device_class=EGARDIA_TYPE_TO_DEVICE_CLASS.get(disc_info[sensor]['type'], None)
            )
            for sensor in disc_info
        ),
        True
    )

class EgardiaBinarySensor(BinarySensorEntity):
    """Represents a sensor based on an Egardia sensor (IR, Door Contact)."""

    def __init__(
        self,
        sensor_id: str,
        name: str,
        egardia_system: object,
        device_class: BinarySensorDeviceClass | None
    ) -> None:
        """Initialize the sensor device."""
        self._id: str = sensor_id
        self._name: str = name
        self._state: str | None = None
        self._device_class: BinarySensorDeviceClass | None = device_class
        self._egardia_system: object = egardia_system

    def update(self) -> None:
        """Update the status."""
        egardia_input = self._egardia_system.getsensorstate(self._id)
        self._state = STATE_ON if egardia_input else STATE_OFF

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Whether the device is switched on."""
        return self._state == STATE_ON

    @property
    def device_class(self) -> BinarySensorDeviceClass | None:
        """Return the device class."""
        return self._device_class
