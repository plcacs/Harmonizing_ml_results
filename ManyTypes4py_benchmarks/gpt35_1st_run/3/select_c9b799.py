from __future__ import annotations
from typing import Any, Dict, List, Optional

from aiohomekit.model.characteristics import Characteristic
from aiohomekit.model.characteristics.const import TargetAirPurifierStateValues, TemperatureDisplayUnits
from homeassistant.components.select import SelectEntity, SelectEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity

class HomeKitSelectEntityDescription(SelectEntityDescription):
    name: Optional[str] = None

class BaseHomeKitSelect(CharacteristicEntity, SelectEntity):
    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def current_option(self) -> Optional[str]:
        ...

    async def async_select_option(self, option: str) -> None:
        ...

class HomeKitSelect(BaseHomeKitSelect):
    def __init__(self, conn: HKDevice, info: Dict[str, Any], char: Characteristic, description: HomeKitSelectEntityDescription) -> None:
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def current_option(self) -> Optional[str]:
        ...

    async def async_select_option(self, option: str) -> None:
        ...

class EcobeeModeSelect(BaseHomeKitSelect):
    @property
    def name(self) -> str:
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def current_option(self) -> Optional[str]:
        ...

    async def async_select_option(self, option: str) -> None:
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        ...
