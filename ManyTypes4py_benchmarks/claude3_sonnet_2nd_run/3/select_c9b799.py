"""Support for Homekit select entities."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, List, Mapping, Optional, Union, cast

from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes
from aiohomekit.model.characteristics.const import TargetAirPurifierStateValues, TemperatureDisplayUnits
from homeassistant.components.select import SelectEntity, SelectEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType

from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity

@dataclass(frozen=True, kw_only=True)
class HomeKitSelectEntityDescription(SelectEntityDescription):
    """A generic description of a select entity backed by a single characteristic."""
    name: Optional[str] = None
    choices: Dict[str, int] = None

SELECT_ENTITIES: Final[Dict[str, HomeKitSelectEntityDescription]] = {
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

_ECOBEE_MODE_TO_TEXT: Final[Dict[int, str]] = {0: 'home', 1: 'sleep', 2: 'away'}
_ECOBEE_MODE_TO_NUMBERS: Final[Dict[str, int]] = {v: k for k, v in _ECOBEE_MODE_TO_TEXT.items()}

class BaseHomeKitSelect(CharacteristicEntity, SelectEntity):
    """Base entity for select entities backed by a single characteristics."""

class HomeKitSelect(BaseHomeKitSelect):
    """Representation of a select control on a homekit accessory."""

    def __init__(
        self, 
        conn: HKDevice, 
        info: Dict[str, int], 
        char: Characteristic, 
        description: HomeKitSelectEntityDescription
    ) -> None:
        """Initialise a HomeKit select control."""
        self.entity_description: HomeKitSelectEntityDescription = description
        self._choice_to_enum: Dict[str, int] = self.entity_description.choices
        self._enum_to_choice: Dict[int, str] = {v: k for k, v in self.entity_description.choices.items()}
        self._attr_options: List[str] = list(self.entity_description.choices.keys())
        super().__init__(conn, info, char)

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [self._char.type]

    @property
    def name(self) -> Optional[str]:
        """Return the name of the device if any."""
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return self.entity_description.name

    @property
    def current_option(self) -> Optional[str]:
        """Return the current selected option."""
        return self._enum_to_choice.get(self._char.value)

    async def async_select_option(self, option: str) -> None:
        """Set the current option."""
        await self.async_put_characteristics({self._char.type: self._choice_to_enum[option]})

class EcobeeModeSelect(BaseHomeKitSelect):
    """Represents a ecobee mode select entity."""
    _attr_options: List[str] = ['home', 'sleep', 'away']
    _attr_translation_key: str = 'ecobee_mode'

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        if (name := super().name):
            return f'{name} Current Mode'
        return 'Current Mode'

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [CharacteristicsTypes.VENDOR_ECOBEE_CURRENT_MODE]

    @property
    def current_option(self) -> Optional[str]:
        """Return the current selected option."""
        return _ECOBEE_MODE_TO_TEXT.get(self._char.value)

    async def async_select_option(self, option: str) -> None:
        """Set the current mode."""
        option_int: int = _ECOBEE_MODE_TO_NUMBERS[option]
        await self.async_put_characteristics({CharacteristicsTypes.VENDOR_ECOBEE_SET_HOLD_SCHEDULE: option_int})

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Homekit select entities."""
    hkid: str = config_entry.data['AccessoryPairingID']
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        entities: List[BaseHomeKitSelect] = []
        info: Dict[str, int] = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        if (description := SELECT_ENTITIES.get(char.type)):
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
