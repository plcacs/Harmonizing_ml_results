from __future__ import annotations
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo
from .board import FirmataPinType
from .const import DOMAIN, FIRMATA_MANUFACTURER
from .pin import FirmataBoardPin

class FirmataEntity:
    def __init__(self, api: Any) -> None:
        self._api = api

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(connections=set(), identifiers={(DOMAIN, self._api.board.name)}, manufacturer=FIRMATA_MANUFACTURER, name=self._api.board.name, sw_version=self._api.board.firmware_version)

class FirmataPinEntity(FirmataEntity):
    _attr_should_poll: bool = False

    def __init__(self, api: Any, config_entry: ConfigEntry, name: str, pin: int) -> None:
        super().__init__(api)
        self._name = name
        location: tuple[str, str, int] = (config_entry.entry_id, 'pin', pin)
        self._unique_id = '_'.join((str(i) for i in location))

    @property
    def name(self) -> str:
        return self._name

    @property
    def unique_id(self) -> str:
        return self._unique_id
