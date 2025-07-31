from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes
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


@dataclass(frozen=True, kw_only=True)
class HomeKitSelectEntityDescription(SelectEntityDescription):
    name: Optional[str] = None


SELECT_ENTITIES: Dict[str, HomeKitSelectEntityDescription] = {
    CharacteristicsTypes.TEMPERATURE_UNITS: HomeKitSelectEntityDescription(
        key='temperature_display_units',
        translation_key='temperature_display_units',
        name='Temperature Display Units',
        entity_category=EntityCategory.CONFIG,
        choices={'celsius': TemperatureDisplayUnits.CELSIUS, 'fahrenheit': TemperatureDisplayUnits.FAHRENHEIT}
    ),
    CharacteristicsTypes.AIR_PURIFIER_STATE_TARGET: HomeKitSelectEntityDescription(
        key='air_purifier_state_target',
        translation_key='air_purifier_state_target',
        name='Air Purifier Mode',
        entity_category=EntityCategory.CONFIG,
        choices={'automatic': TargetAirPurifierStateValues.AUTOMATIC, 'manual': TargetAirPurifierStateValues.MANUAL}
    )
}
_ECOBEE_MODE_TO_TEXT: Dict[int, str] = {0: 'home', 1: 'sleep', 2: 'away'}
_ECOBEE_MODE_TO_NUMBERS: Dict[str, int] = {v: k for k, v in _ECOBEE_MODE_TO_TEXT.items()}


class BaseHomeKitSelect(CharacteristicEntity, SelectEntity):
    pass


class HomeKitSelect(BaseHomeKitSelect):
    def __init__(
        self,
        conn: HKDevice,
        info: Dict[str, Any],
        char: Characteristic,
        description: HomeKitSelectEntityDescription,
    ) -> None:
        self.entity_description: HomeKitSelectEntityDescription = description
        self._choice_to_enum: Dict[str, Any] = self.entity_description.choices  # choices mapping str to enum value
        self._enum_to_choice: Dict[Any, str] = {v: k for k, v in self.entity_description.choices.items()}
        self._attr_options: List[str] = list(self.entity_description.choices.keys())
        super().__init__(conn, info, char)

    def get_characteristic_types(self) -> List[str]:
        return [self._char.type]

    @property
    def name(self) -> str:
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return self.entity_description.name  # type: ignore

    @property
    def current_option(self) -> Optional[str]:
        return self._enum_to_choice.get(self._char.value)

    async def async_select_option(self, option: str) -> None:
        await self.async_put_characteristics({self._char.type: self._choice_to_enum[option]})


class EcobeeModeSelect(BaseHomeKitSelect):
    _attr_options: List[str] = ['home', 'sleep', 'away']
    _attr_translation_key: str = 'ecobee_mode'

    @property
    def name(self) -> str:
        if (name := super().name):
            return f'{name} Current Mode'
        return 'Current Mode'

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.VENDOR_ECOBEE_CURRENT_MODE]

    @property
    def current_option(self) -> Optional[str]:
        return _ECOBEE_MODE_TO_TEXT.get(self._char.value)

    async def async_select_option(self, option: str) -> None:
        option_int: int = _ECOBEE_MODE_TO_NUMBERS[option]
        await self.async_put_characteristics({CharacteristicsTypes.VENDOR_ECOBEE_SET_HOLD_SCHEDULE: option_int})


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    hkid: str = config_entry.data['AccessoryPairingID']
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        entities: List[SelectEntity] = []
        info: Dict[str, Any] = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        description: Optional[HomeKitSelectEntityDescription] = SELECT_ENTITIES.get(char.type)
        if description:
            entities.append(HomeKitSelect(conn, info, char, description))
        elif char.type == CharacteristicsTypes.VENDOR_ECOBEE_CURRENT_MODE:
            entities.append(EcobeeModeSelect(conn, info, char))
        if not entities:
            return False
        for entity in entities:
            conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SELECT)
        async_add_entities(entities)
        return True

    conn.add_char_factory(async_add_characteristic)