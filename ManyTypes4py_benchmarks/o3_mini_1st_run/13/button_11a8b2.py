from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Type

from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes
from homeassistant.components.button import ButtonDeviceClass, ButtonEntity, ButtonEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType

from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HomeKitButtonEntityDescription(ButtonEntityDescription):
    """Describes HomeKit button."""
    probe: Any = None
    write_value: Any = None


BUTTON_ENTITIES: Dict[str, HomeKitButtonEntityDescription] = {
    CharacteristicsTypes.VENDOR_HAA_SETUP: HomeKitButtonEntityDescription(
        key=CharacteristicsTypes.VENDOR_HAA_SETUP,
        name='Setup',
        translation_key='setup',
        entity_category=EntityCategory.CONFIG,
        write_value='#HAA@trcmd'
    ),
    CharacteristicsTypes.VENDOR_HAA_UPDATE: HomeKitButtonEntityDescription(
        key=CharacteristicsTypes.VENDOR_HAA_UPDATE,
        name='Update',
        device_class=ButtonDeviceClass.UPDATE,
        entity_category=EntityCategory.CONFIG,
        write_value='#HAA@trcmd'
    ),
    CharacteristicsTypes.IDENTIFY: HomeKitButtonEntityDescription(
        key=CharacteristicsTypes.IDENTIFY,
        name='Identify',
        device_class=ButtonDeviceClass.IDENTIFY,
        entity_category=EntityCategory.DIAGNOSTIC,
        write_value=True
    )
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Homekit buttons."""
    hkid: str = config_entry.data['AccessoryPairingID']
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        entities: List[ButtonEntity] = []
        info: Dict[str, Any] = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        if (description := BUTTON_ENTITIES.get(char.type)):
            entities.append(HomeKitButton(conn, info, char, description))
        elif (entity_type := BUTTON_ENTITY_CLASSES.get(char.type)):
            entities.append(entity_type(conn, info, char))
        elif char.type == CharacteristicsTypes.THREAD_CONTROL_POINT:
            if not conn.is_unprovisioned_thread_device:
                return False
            entities.append(HomeKitProvisionPreferredThreadCredentials(conn, info, char))
        else:
            return False
        for entity in entities:
            conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.BUTTON)
        async_add_entities(entities)
        return True

    conn.add_char_factory(async_add_characteristic)


class BaseHomeKitButton(CharacteristicEntity, ButtonEntity):
    """Base class for all HomeKit buttons."""


class HomeKitButton(BaseHomeKitButton):
    """Representation of a Button control on a HomeKit accessory."""

    def __init__(
        self, conn: HKDevice, info: Dict[str, Any], char: Characteristic, description: HomeKitButtonEntityDescription
    ) -> None:
        """Initialise a HomeKit button control."""
        self.entity_description: HomeKitButtonEntityDescription = description
        super().__init__(conn, info, char)

    def get_characteristic_types(self) -> List[str]:
        """Define the HomeKit characteristics the entity is tracking."""
        return [self._char.type]

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return f'{self.entity_description.name}'

    async def async_press(self) -> None:
        """Press the button."""
        key: str = self.entity_description.key
        val: Any = self.entity_description.write_value
        await self.async_put_characteristics({key: val})


class HomeKitEcobeeClearHoldButton(BaseHomeKitButton):
    """Representation of a Button control for Ecobee clear hold request."""

    def get_characteristic_types(self) -> List[str]:
        """Define the HomeKit characteristics the entity is tracking."""
        return []

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        prefix: str = ''
        if (name := super().name):
            prefix = name
        return f'{prefix} Clear Hold'

    async def async_press(self) -> None:
        """Press the button."""
        key: str = self._char.type
        for val in (False, True):
            await self.async_put_characteristics({key: val})


class HomeKitProvisionPreferredThreadCredentials(BaseHomeKitButton):
    """A button users can press to migrate their HomeKit BLE device to Thread."""
    _attr_entity_category: EntityCategory = EntityCategory.CONFIG

    def get_characteristic_types(self) -> List[str]:
        """Define the HomeKit characteristics the entity is tracking."""
        return []

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        prefix: str = ''
        if (name := super().name):
            prefix = name
        return f'{prefix} Provision Preferred Thread Credentials'

    async def async_press(self) -> None:
        """Press the button."""
        await self._accessory.async_thread_provision()


BUTTON_ENTITY_CLASSES: Dict[str, Type[BaseHomeKitButton]] = {
    CharacteristicsTypes.VENDOR_ECOBEE_CLEAR_HOLD: HomeKitEcobeeClearHoldButton
}