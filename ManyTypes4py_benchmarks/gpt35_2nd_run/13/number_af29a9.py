from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import logging
from typing import List, Optional
from pykoplenti import SettingsData
from homeassistant.components.number import NumberDeviceClass, NumberEntity, NumberEntityDescription, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import SettingDataUpdateCoordinator
from .helper import PlenticoreDataFormatter

@dataclass(frozen=True, kw_only=True)
class PlenticoreNumberEntityDescription(NumberEntityDescription):
    """Describes a Plenticore number entity."""

NUMBER_SETTINGS_DATA: List[PlenticoreNumberEntityDescription]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Add Kostal Plenticore Number entities."""

class PlenticoreDataNumber(CoordinatorEntity[SettingDataUpdateCoordinator], NumberEntity):
    """Representation of a Kostal Plenticore Number entity."""

    def __init__(self, coordinator: SettingDataUpdateCoordinator, entry_id: str, platform_name: str, device_info: DeviceInfo, description: PlenticoreNumberEntityDescription, setting_data: SettingsData) -> None:

    @property
    def module_id(self) -> str:

    @property
    def data_id(self) -> str:

    @property
    def available(self) -> bool:

    async def async_added_to_hass(self) -> None:

    async def async_will_remove_from_hass(self) -> None:

    @property
    def native_value(self) -> Optional[float]:

    async def async_set_native_value(self, value: float) -> None:
