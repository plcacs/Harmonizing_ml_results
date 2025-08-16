from typing import Any, Dict, List, Optional
from homeassistant.components.air_quality import AirQualityEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

class AirMonitorB1(XiaomiMiioEntity, AirQualityEntity):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        ...

    async def async_update(self) -> None:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def available(self) -> Optional[bool]:
        ...

    @property
    def air_quality_index(self) -> Optional[int]:
        ...

    @property
    def carbon_dioxide(self) -> Optional[float]:
        ...

    @property
    def carbon_dioxide_equivalent(self) -> Optional[float]:
        ...

    @property
    def particulate_matter_2_5(self) -> Optional[float]:
        ...

    @property
    def total_volatile_organic_compounds(self) -> Optional[float]:
        ...

    @property
    def temperature(self) -> Optional[float]:
        ...

    @property
    def humidity(self) -> Optional[float]:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

class AirMonitorS1(AirMonitorB1):
    async def async_update(self) -> None:
        ...

class AirMonitorV1(AirMonitorB1):
    async def async_update(self) -> None:
        ...

    @property
    def unit_of_measurement(self) -> Optional[str]:
        ...

class AirMonitorCGDN1(XiaomiMiioEntity, AirQualityEntity):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        ...

    async def async_update(self) -> None:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def available(self) -> Optional[bool]:
        ...

    @property
    def carbon_dioxide(self) -> Optional[float]:
        ...

    @property
    def particulate_matter_2_5(self) -> Optional[float]:
        ...

    @property
    def particulate_matter_10(self) -> Optional[float]:
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...
