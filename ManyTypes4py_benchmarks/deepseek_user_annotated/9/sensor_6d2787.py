"""Support for Kaiterra Temperature and Humidity Sensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.const import CONF_DEVICE_ID, CONF_NAME, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import DISPATCHER_KAITERRA, DOMAIN


@dataclass(frozen=True, kw_only=True)
class KaiterraSensorEntityDescription(SensorEntityDescription):
    """Class describing Renault sensor entities."""

    suffix: str


SENSORS: list[KaiterraSensorEntityDescription] = [
    KaiterraSensorEntityDescription(
        suffix="Temperature",
        key="rtemp",
        device_class=SensorDeviceClass.TEMPERATURE,
    ),
    KaiterraSensorEntityDescription(
        suffix="Humidity",
        key="rhumid",
        device_class=SensorDeviceClass.HUMIDITY,
    ),
]


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the kaiterra temperature and humidity sensor."""
    if discovery_info is None:
        return

    api = hass.data[DOMAIN]
    name = cast(str, discovery_info[CONF_NAME])
    device_id = cast(str, discovery_info[CONF_DEVICE_ID])

    async_add_entities(
        [KaiterraSensor(api, name, device_id, description) for description in SENSORS]
    )


class KaiterraSensor(SensorEntity):
    """Implementation of a Kaittera sensor."""

    _attr_should_poll: bool = False
    _api: Any
    _device_id: str
    _attr_name: str | None = None
    _attr_unique_id: str | None = None

    def __init__(
        self, api: Any, name: str, device_id: str, description: KaiterraSensorEntityDescription
    ) -> None:
        """Initialize the sensor."""
        self._api = api
        self._device_id = device_id
        self.entity_description = description
        self._attr_name = f"{name} {description.suffix}"
        self._attr_unique_id = f"{device_id}_{description.suffix.lower()}"

    @property
    def _sensor(self) -> dict[str, Any]:
        """Return the sensor data."""
        return self._api.data.get(self._device_id, {}).get(
            self.entity_description.key, {}
        )

    @property
    def available(self) -> bool:
        """Return the availability of the sensor."""
        return self._api.data.get(self._device_id) is not None

    @property
    def native_value(self) -> Any:
        """Return the state."""
        return self._sensor.get("value")

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit the value is expressed in."""
        if not self._sensor.get("units"):
            return None

        value = self._sensor["units"].value

        if value == "F":
            return UnitOfTemperature.FAHRENHEIT
        if value == "C":
            return UnitOfTemperature.CELSIUS
        return cast(str, value)

    async def async_added_to_hass(self) -> None:
        """Register callback."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, DISPATCHER_KAITERRA, self.async_write_ha_state
            )
        )
