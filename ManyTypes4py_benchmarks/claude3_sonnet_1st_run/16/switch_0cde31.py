"""Support for Homekit switches."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Set, Union

from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes, InUseValues, IsConfiguredValues
from aiohomekit.model.services import Service, ServicesTypes

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType

from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity, HomeKitEntity

OUTLET_IN_USE: Final[str] = 'outlet_in_use'
ATTR_IN_USE: Final[str] = 'in_use'
ATTR_IS_CONFIGURED: Final[str] = 'is_configured'
ATTR_REMAINING_DURATION: Final[str] = 'remaining_duration'


@dataclass(frozen=True)
class DeclarativeSwitchEntityDescription(SwitchEntityDescription):
    """Describes Homekit button."""
    true_value: Any = True
    false_value: Any = False


SWITCH_ENTITIES: Dict[str, DeclarativeSwitchEntityDescription] = {
    CharacteristicsTypes.VENDOR_AQARA_PAIRING_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AQARA_PAIRING_MODE,
        name='Pairing Mode',
        translation_key='pairing_mode',
        entity_category=EntityCategory.CONFIG
    ),
    CharacteristicsTypes.VENDOR_AQARA_E1_PAIRING_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AQARA_E1_PAIRING_MODE,
        name='Pairing Mode',
        translation_key='pairing_mode',
        entity_category=EntityCategory.CONFIG
    ),
    CharacteristicsTypes.LOCK_PHYSICAL_CONTROLS: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.LOCK_PHYSICAL_CONTROLS,
        name='Lock Physical Controls',
        translation_key='lock_physical_controls',
        entity_category=EntityCategory.CONFIG
    ),
    CharacteristicsTypes.MUTE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.MUTE,
        name='Mute',
        translation_key='mute',
        entity_category=EntityCategory.CONFIG
    ),
    CharacteristicsTypes.VENDOR_AIRVERSA_SLEEP_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AIRVERSA_SLEEP_MODE,
        name='Sleep Mode',
        translation_key='sleep_mode',
        entity_category=EntityCategory.CONFIG
    )
}


class HomeKitSwitch(HomeKitEntity, SwitchEntity):
    """Representation of a Homekit switch."""

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [CharacteristicsTypes.ON, CharacteristicsTypes.OUTLET_IN_USE]

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self.service.value(CharacteristicsTypes.ON)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the specified switch on."""
        await self.async_put_characteristics({CharacteristicsTypes.ON: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the specified switch off."""
        await self.async_put_characteristics({CharacteristicsTypes.ON: False})

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the optional state attributes."""
        outlet_in_use = self.service.value(CharacteristicsTypes.OUTLET_IN_USE)
        if outlet_in_use is not None:
            return {OUTLET_IN_USE: outlet_in_use}
        return None


class HomeKitFaucet(HomeKitEntity, SwitchEntity):
    """Representation of a Homekit faucet."""

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [CharacteristicsTypes.ACTIVE]

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self.service.value(CharacteristicsTypes.ACTIVE)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the specified faucet on."""
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the specified faucet off."""
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: False})


class HomeKitValve(HomeKitEntity, SwitchEntity):
    """Represents a valve in an irrigation system."""
    _attr_translation_key: str = 'valve'

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [
            CharacteristicsTypes.ACTIVE,
            CharacteristicsTypes.IN_USE,
            CharacteristicsTypes.IS_CONFIGURED,
            CharacteristicsTypes.REMAINING_DURATION
        ]

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the specified valve on."""
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the specified valve off."""
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: False})

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self.service.value(CharacteristicsTypes.ACTIVE)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the optional state attributes."""
        attrs: Dict[str, Any] = {}
        in_use = self.service.value(CharacteristicsTypes.IN_USE)
        if in_use is not None:
            attrs[ATTR_IN_USE] = in_use == InUseValues.IN_USE
        is_configured = self.service.value(CharacteristicsTypes.IS_CONFIGURED)
        if is_configured is not None:
            attrs[ATTR_IS_CONFIGURED] = is_configured == IsConfiguredValues.CONFIGURED
        remaining = self.service.value(CharacteristicsTypes.REMAINING_DURATION)
        if remaining is not None:
            attrs[ATTR_REMAINING_DURATION] = remaining
        return attrs


class DeclarativeCharacteristicSwitch(CharacteristicEntity, SwitchEntity):
    """Representation of a Homekit switch backed by a single characteristic."""

    entity_description: DeclarativeSwitchEntityDescription

    def __init__(
        self,
        conn: HKDevice,
        info: Dict[str, Any],
        char: Characteristic,
        description: DeclarativeSwitchEntityDescription
    ) -> None:
        """Initialise a HomeKit switch."""
        self.entity_description = description
        super().__init__(conn, info, char)

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return f'{self.entity_description.name}'

    def get_characteristic_types(self) -> List[str]:
        """Define the homekit characteristics the entity cares about."""
        return [self._char.type]

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._char.value == self.entity_description.true_value

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the specified switch on."""
        await self.async_put_characteristics({self._char.type: self.entity_description.true_value})

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the specified switch off."""
        await self.async_put_characteristics({self._char.type: self.entity_description.false_value})


ENTITY_TYPES: Dict[str, type[HomeKitEntity]] = {
    ServicesTypes.SWITCH: HomeKitSwitch,
    ServicesTypes.OUTLET: HomeKitSwitch,
    ServicesTypes.FAUCET: HomeKitFaucet,
    ServicesTypes.VALVE: HomeKitValve
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Homekit switches."""
    hkid: str = config_entry.data['AccessoryPairingID']
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_service(service: Service) -> bool:
        if not (entity_class := ENTITY_TYPES.get(service.type)):
            return False
        info: Dict[str, Any] = {'aid': service.accessory.aid, 'iid': service.iid}
        entity = entity_class(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SWITCH)
        async_add_entities([entity])
        return True

    conn.add_listener(async_add_service)

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        if not (description := SWITCH_ENTITIES.get(char.type)):
            return False
        info: Dict[str, Any] = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        entity = DeclarativeCharacteristicSwitch(conn, info, char, description)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SWITCH)
        async_add_entities([entity])
        return True

    conn.add_char_factory(async_add_characteristic)
