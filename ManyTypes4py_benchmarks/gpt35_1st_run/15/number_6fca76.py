from aiohomekit.model.characteristics import Characteristic
from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from .connection import HKDevice
from .entity import CharacteristicEntity
from typing import Any, Dict, List, Optional

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HomeKitNumber(CharacteristicEntity, NumberEntity):
    def __init__(self, conn: HKDevice, info: Dict[str, Any], char: Characteristic, description: NumberEntityDescription) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    def get_characteristic_types(self) -> List[str]:
        ...

    @property
    def native_min_value(self) -> float:
        ...

    @property
    def native_max_value(self) -> float:
        ...

    @property
    def native_step(self) -> float:
        ...

    @property
    def native_value(self) -> Optional[float]:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...
