from typing import Any, Optional
from homeassistant.helpers import entity
from homeassistant.helpers.device_registry import DeviceInfo
from ..const import CONF_ALLOW_UNREACHABLE, DEFAULT_ALLOW_UNREACHABLE, DOMAIN as HUE_DOMAIN

class GenericHueDevice(entity.Entity):
    """Representation of a Hue device."""

    def __init__(self, sensor: Any, name: str, bridge: Any, primary_sensor: Optional[Any] = None) -> None:
        """Initialize the sensor."""
        self.sensor: Any = sensor
        self._name: str = name
        self._primary_sensor: Optional[Any] = primary_sensor
        self.bridge: Any = bridge
        self.allow_unreachable: bool = bridge.config_entry.options.get(
            CONF_ALLOW_UNREACHABLE, DEFAULT_ALLOW_UNREACHABLE
        )

    @property
    def primary_sensor(self) -> Any:
        """Return the primary sensor entity of the physical device."""
        return self._primary_sensor or self.sensor

    @property
    def device_id(self) -> str:
        """Return the ID of the physical device this sensor is part of."""
        return self.unique_id[:23]

    @property
    def unique_id(self) -> str:
        """Return the ID of this Hue sensor."""
        return self.sensor.uniqueid

    @property
    def name(self) -> str:
        """Return a friendly name for the sensor."""
        return self._name

    @property
    def swupdatestate(self) -> Optional[Any]:
        """Return detail of available software updates for this device."""
        return self.primary_sensor.raw.get("swupdate", {}).get("state")

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info.

        Links individual entities together in the hass device registry.
        """
        return DeviceInfo(
            identifiers={(HUE_DOMAIN, self.device_id)},
            manufacturer=self.primary_sensor.manufacturername,
            model=self.primary_sensor.productname or self.primary_sensor.modelid,
            name=self.primary_sensor.name,
            sw_version=self.primary_sensor.swversion,
            via_device=(HUE_DOMAIN, self.bridge.api.config.bridgeid),
        )