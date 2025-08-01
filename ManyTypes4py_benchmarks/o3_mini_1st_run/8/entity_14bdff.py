"""Entity for Firmata devices."""
from __future__ import annotations
from typing import Any, Set, Tuple
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo
from .board import FirmataPinType
from .const import DOMAIN, FIRMATA_MANUFACTURER
from .pin import FirmataBoardPin

class FirmataEntity:
    """Representation of a Firmata entity."""

    def __init__(self, api: Any) -> None:
        """Initialize the entity."""
        self._api = api

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            connections=set(),
            identifiers={(DOMAIN, self._api.board.name)},
            manufacturer=FIRMATA_MANUFACTURER,
            name=self._api.board.name,
            sw_version=self._api.board.firmware_version,
        )

class FirmataPinEntity(FirmataEntity):
    """Representation of a Firmata pin entity."""
    _attr_should_poll = False

    def __init__(self, api: Any, config_entry: ConfigEntry, name: str, pin: Any) -> None:
        """Initialize the pin entity."""
        super().__init__(api)
        self._name: str = name
        location: Tuple[Any, str, Any] = (config_entry.entry_id, 'pin', pin)
        self._unique_id: str = '_'.join((str(i) for i in location))

    @property
    def name(self) -> str:
        """Get the name of the pin."""
        return self._name

    @property
    def unique_id(self) -> str:
        """Return a unique identifier for this device."""
        return self._unique_id