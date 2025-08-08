from homeassistant.components.humidifier import HumidifierEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class XiaomiGenericHumidifier(XiaomiCoordinatedMiioEntity, HumidifierEntity):
    ...

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def mode(self) -> Any:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    def translate_humidity(self, humidity: float) -> float:
        ...

class XiaomiAirHumidifier(XiaomiGenericHumidifier, HumidifierEntity):
    ...

    def __init__(self, device: Any, entry: ConfigEntry, unique_id: str, coordinator: Any) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def mode(self) -> str:
        ...

    @property
    def target_humidity(self) -> float:
        ...

    async def async_set_humidity(self, humidity: float) -> None:
        ...

    async def async_set_mode(self, mode: str) -> None:
        ...

class XiaomiAirHumidifierMiot(XiaomiAirHumidifier):
    ...

    @property
    def mode(self) -> str:
        ...

    @property
    def target_humidity(self) -> float:
        ...

    async def async_set_humidity(self, humidity: float) -> None:
        ...

    async def async_set_mode(self, mode: str) -> None:
        ...

class XiaomiAirHumidifierMjjsq(XiaomiAirHumidifier):
    ...

    @property
    def mode(self) -> str:
        ...

    @property
    def target_humidity(self) -> float:
        ...

    async def async_set_humidity(self, humidity: float) -> None:
        ...

    async def async_set_mode(self, mode: str) -> None:
        ...
