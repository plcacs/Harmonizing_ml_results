from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aiohomekit.model.characteristics import (
    Characteristic,
    CharacteristicsTypes,
    InUseValues,
    IsConfiguredValues,
)
from aiohomekit.model.services import Service, ServicesTypes
from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType

from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity, HomeKitEntity

OUTLET_IN_USE: str = 'outlet_in_use'
ATTR_IN_USE: str = 'in_use'
ATTR_IS_CONFIGURED: str = 'is_configured'
ATTR_REMAINING_DURATION: str = 'remaining_duration'


@dataclass(frozen=True)
class DeclarativeSwitchEntityDescription(SwitchEntityDescription):
    true_value: bool = True
    false_value: bool = False


SWITCH_ENTITIES: Dict[str, DeclarativeSwitchEntityDescription] = {
    CharacteristicsTypes.VENDOR_AQARA_PAIRING_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AQARA_PAIRING_MODE,
        name='Pairing Mode',
        translation_key='pairing_mode',
        entity_category=EntityCategory.CONFIG,
    ),
    CharacteristicsTypes.VENDOR_AQARA_E1_PAIRING_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AQARA_E1_PAIRING_MODE,
        name='Pairing Mode',
        translation_key='pairing_mode',
        entity_category=EntityCategory.CONFIG,
    ),
    CharacteristicsTypes.LOCK_PHYSICAL_CONTROLS: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.LOCK_PHYSICAL_CONTROLS,
        name='Lock Physical Controls',
        translation_key='lock_physical_controls',
        entity_category=EntityCategory.CONFIG,
    ),
    CharacteristicsTypes.MUTE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.MUTE,
        name='Mute',
        translation_key='mute',
        entity_category=EntityCategory.CONFIG,
    ),
    CharacteristicsTypes.VENDOR_AIRVERSA_SLEEP_MODE: DeclarativeSwitchEntityDescription(
        key=CharacteristicsTypes.VENDOR_AIRVERSA_SLEEP_MODE,
        name='Sleep Mode',
        translation_key='sleep_mode',
        entity_category=EntityCategory.CONFIG,
    ),
}


class HomeKitSwitch(HomeKitEntity, SwitchEntity):
    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.ON, CharacteristicsTypes.OUTLET_IN_USE]

    @property
    def is_on(self) -> bool:
        return self.service.value(CharacteristicsTypes.ON)

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ON: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ON: False})

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        outlet_in_use = self.service.value(CharacteristicsTypes.OUTLET_IN_USE)
        if outlet_in_use is not None:
            return {OUTLET_IN_USE: outlet_in_use}
        return None


class HomeKitFaucet(HomeKitEntity, SwitchEntity):
    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.ACTIVE]

    @property
    def is_on(self) -> bool:
        return self.service.value(CharacteristicsTypes.ACTIVE)

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: False})


class HomeKitValve(HomeKitEntity, SwitchEntity):
    _attr_translation_key: str = 'valve'

    def get_characteristic_types(self) -> List[str]:
        return [
            CharacteristicsTypes.ACTIVE,
            CharacteristicsTypes.IN_USE,
            CharacteristicsTypes.IS_CONFIGURED,
            CharacteristicsTypes.REMAINING_DURATION,
        ]

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: True})

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({CharacteristicsTypes.ACTIVE: False})

    @property
    def is_on(self) -> bool:
        return self.service.value(CharacteristicsTypes.ACTIVE)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
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
    def __init__(
        self,
        conn: HKDevice,
        info: Dict[str, Any],
        char: Characteristic,
        description: DeclarativeSwitchEntityDescription,
    ) -> None:
        self.entity_description: DeclarativeSwitchEntityDescription = description
        super().__init__(conn, info, char)

    @property
    def name(self) -> str:
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return f'{self.entity_description.name}'

    def get_characteristic_types(self) -> List[str]:
        return [self._char.type]

    @property
    def is_on(self) -> bool:
        return self._char.value == self.entity_description.true_value

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({self._char.type: self.entity_description.true_value})

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.async_put_characteristics({self._char.type: self.entity_description.false_value})


ENTITY_TYPES: Dict[str, type[HomeKitEntity]] = {
    ServicesTypes.SWITCH: HomeKitSwitch,
    ServicesTypes.OUTLET: HomeKitSwitch,
    ServicesTypes.FAUCET: HomeKitFaucet,
    ServicesTypes.VALVE: HomeKitValve,
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    hkid: str = config_entry.data["AccessoryPairingID"]
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_service(service: Service) -> bool:
        entity_class: Optional[type[HomeKitEntity]] = ENTITY_TYPES.get(service.type)
        if not entity_class:
            return False
        info: Dict[str, Any] = {"aid": service.accessory.aid, "iid": service.iid}
        entity: HomeKitEntity = entity_class(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SWITCH)
        async_add_entities([entity])
        return True

    conn.add_listener(async_add_service)

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        description: Optional[DeclarativeSwitchEntityDescription] = SWITCH_ENTITIES.get(char.type)
        if not description:
            return False
        info: Dict[str, Any] = {"aid": char.service.accessory.aid, "iid": char.service.iid}
        entity = DeclarativeCharacteristicSwitch(conn, info, char, description)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SWITCH)
        async_add_entities([entity])
        return True

    conn.add_char_factory(async_add_characteristic)