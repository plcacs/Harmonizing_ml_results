"""Representation of Z-Wave binary sensors."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import CommandClass
from zwave_js_server.const.command_class.lock import DOOR_STATUS_PROPERTY
from zwave_js_server.const.command_class.notification import CC_SPECIFIC_NOTIFICATION_TYPE
from zwave_js_server.model.driver import Driver
from homeassistant.components.binary_sensor import (
    DOMAIN as BINARY_SENSOR_DOMAIN,
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DATA_CLIENT, DOMAIN
from .discovery import ZwaveDiscoveryInfo
from .entity import ZWaveBaseEntity

PARALLEL_UPDATES: int = 0

NOTIFICATION_SMOKE_ALARM: str = '1'
NOTIFICATION_CARBON_MONOOXIDE: str = '2'
NOTIFICATION_CARBON_DIOXIDE: str = '3'
NOTIFICATION_HEAT: str = '4'
NOTIFICATION_WATER: str = '5'
NOTIFICATION_ACCESS_CONTROL: str = '6'
NOTIFICATION_HOME_SECURITY: str = '7'
NOTIFICATION_POWER_MANAGEMENT: str = '8'
NOTIFICATION_SYSTEM: str = '9'
NOTIFICATION_EMERGENCY: str = '10'
NOTIFICATION_CLOCK: str = '11'
NOTIFICATION_APPLIANCE: str = '12'
NOTIFICATION_HOME_HEALTH: str = '13'
NOTIFICATION_SIREN: str = '14'
NOTIFICATION_WATER_VALVE: str = '15'
NOTIFICATION_WEATHER: str = '16'
NOTIFICATION_IRRIGATION: str = '17'
NOTIFICATION_GAS: str = '18'


@dataclass(frozen=True)
class NotificationZWaveJSEntityDescription(BinarySensorEntityDescription):
    """Represent a Z-Wave JS binary sensor entity description."""
    off_state: str = '0'
    states: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True, kw_only=True)
class PropertyZWaveJSEntityDescription(BinarySensorEntityDescription):
    """Represent the entity description for property name sensors."""


NOTIFICATION_SENSOR_MAPPINGS: Tuple[NotificationZWaveJSEntityDescription, ...] = (
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_SMOKE_ALARM,
        states=('1', '2'),
        device_class=BinarySensorDeviceClass.SMOKE,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_SMOKE_ALARM,
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_CARBON_MONOOXIDE,
        states=('1', '2'),
        device_class=BinarySensorDeviceClass.CO,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_CARBON_MONOOXIDE,
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_CARBON_DIOXIDE,
        states=('1', '2'),
        device_class=BinarySensorDeviceClass.GAS,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_CARBON_DIOXIDE,
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HEAT,
        states=('1', '2', '5', '6'),
        device_class=BinarySensorDeviceClass.HEAT,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HEAT,
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_WATER,
        states=('1', '2', '3', '4'),
        device_class=BinarySensorDeviceClass.MOISTURE,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_WATER,
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_ACCESS_CONTROL,
        states=('1', '2', '3', '4'),
        device_class=BinarySensorDeviceClass.LOCK,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_ACCESS_CONTROL,
        states=('11',),
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_ACCESS_CONTROL,
        off_state='23',
        states=('22', '23'),
        device_class=BinarySensorDeviceClass.DOOR,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HOME_SECURITY,
        states=('1', '2'),
        device_class=BinarySensorDeviceClass.SAFETY,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HOME_SECURITY,
        states=('3', '4', '9'),
        device_class=BinarySensorDeviceClass.TAMPER,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HOME_SECURITY,
        states=('5', '6'),
        device_class=BinarySensorDeviceClass.SAFETY,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_HOME_SECURITY,
        states=('7', '8'),
        device_class=BinarySensorDeviceClass.MOTION,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_POWER_MANAGEMENT,
        off_state='2',
        states=('2', '3'),
        device_class=BinarySensorDeviceClass.PLUG,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_POWER_MANAGEMENT,
        states=('6', '7', '8', '9'),
        device_class=BinarySensorDeviceClass.SAFETY,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_POWER_MANAGEMENT,
        states=('10', '11', '17'),
        device_class=BinarySensorDeviceClass.BATTERY,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_SYSTEM,
        states=('1', '2', '3', '4', '6', '7'),
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_EMERGENCY,
        states=('1', '2', '3'),
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_SIREN,
        states=('1',),
        device_class=BinarySensorDeviceClass.SOUND,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_GAS,
        states=('1', '2', '3', '4'),
        device_class=BinarySensorDeviceClass.GAS,
    ),
    NotificationZWaveJSEntityDescription(
        key=NOTIFICATION_GAS,
        states=('6',),
        device_class=BinarySensorDeviceClass.PROBLEM,
    ),
)

PROPERTY_SENSOR_MAPPINGS: Dict[str, PropertyZWaveJSEntityDescription] = {
    DOOR_STATUS_PROPERTY: PropertyZWaveJSEntityDescription(
        key=DOOR_STATUS_PROPERTY,
        on_states=('open',),
        device_class=BinarySensorDeviceClass.DOOR,
    )
}

BOOLEAN_SENSOR_MAPPINGS: Dict[int, BinarySensorEntityDescription] = {
    CommandClass.BATTERY.value: BinarySensorEntityDescription(
        key=str(CommandClass.BATTERY.value),
        device_class=BinarySensorDeviceClass.BATTERY,
        entity_category=EntityCategory.DIAGNOSTIC,
    )
}


@callback
def is_valid_notification_binary_sensor(info: ZwaveDiscoveryInfo) -> bool:
    """Return if the notification CC Value is valid as binary sensor."""
    if not info.primary_value.metadata.states:
        return False
    return len(info.primary_value.metadata.states) > 1


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Z-Wave binary sensor from config entry."""
    client: ZwaveClient = config_entry.runtime_data[DATA_CLIENT]

    @callback
    def async_add_binary_sensor(info: ZwaveDiscoveryInfo) -> None:
        """Add Z-Wave Binary Sensor."""
        driver: Optional[Driver] = client.driver
        if driver is None:
            return
        entities: List[BinarySensorEntity] = []
        if info.platform_hint == 'notification':
            if not is_valid_notification_binary_sensor(info):
                return
            for state_key in info.primary_value.metadata.states:
                if state_key == '0':
                    continue
                notification_description: Optional[NotificationZWaveJSEntityDescription] = None
                for description in NOTIFICATION_SENSOR_MAPPINGS:
                    if (
                        int(description.key) == info.primary_value.metadata.cc_specific.get(CC_SPECIFIC_NOTIFICATION_TYPE, 0)
                        and (not description.states or state_key in description.states)
                    ):
                        notification_description = description
                        break
                if notification_description and notification_description.off_state == state_key:
                    continue
                entities.append(
                    ZWaveNotificationBinarySensor(
                        config_entry=config_entry,
                        driver=driver,
                        info=info,
                        state_key=state_key,
                        description=notification_description,
                    )
                )
        elif (
            info.platform_hint == 'property'
            and info.primary_value.property_name
            and (property_description := PROPERTY_SENSOR_MAPPINGS.get(info.primary_value.property_name))
        ):
            entities.append(
                ZWavePropertyBinarySensor(
                    config_entry=config_entry,
                    driver=driver,
                    info=info,
                    description=property_description,
                )
            )
        elif info.platform_hint == 'config_parameter':
            entities.append(
                ZWaveConfigParameterBinarySensor(
                    config_entry=config_entry,
                    driver=driver,
                    info=info,
                )
            )
        else:
            entities.append(
                ZWaveBooleanBinarySensor(
                    config_entry=config_entry,
                    driver=driver,
                    info=info,
                )
            )
        async_add_entities(entities)

    config_entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            f'{DOMAIN}_{config_entry.entry_id}_add_{BINARY_SENSOR_DOMAIN}',
            async_add_binary_sensor,
        )
    )


class ZWaveBooleanBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    """Representation of a Z-Wave binary_sensor."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
    ) -> None:
        """Initialize a ZWaveBooleanBinarySensor entity."""
        super().__init__(config_entry, driver, info)
        self._attr_name: str = self.generate_name(include_value_name=True)
        description: Optional[BinarySensorEntityDescription] = BOOLEAN_SENSOR_MAPPINGS.get(
            self.info.primary_value.command_class
        )
        if description:
            self.entity_description = description

    @property
    def is_on(self) -> Optional[bool]:
        """Return if the sensor is on or off."""
        if self.info.primary_value.value is None:
            return None
        return bool(self.info.primary_value.value)


class ZWaveNotificationBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    """Representation of a Z-Wave binary_sensor from Notification CommandClass."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        state_key: str,
        description: Optional[NotificationZWaveJSEntityDescription] = None,
    ) -> None:
        """Initialize a ZWaveNotificationBinarySensor entity."""
        super().__init__(config_entry, driver, info)
        self.state_key: str = state_key
        if description:
            self.entity_description = description
        alternate_value_name: Optional[str] = self.info.primary_value.metadata.states.get(state_key)
        self._attr_name: str = self.generate_name(alternate_value_name=alternate_value_name)
        self._attr_unique_id: str = f'{self._attr_unique_id}.{self.state_key}'

    @property
    def is_on(self) -> Optional[bool]:
        """Return if the sensor is on or off."""
        if self.info.primary_value.value is None:
            return None
        return int(self.info.primary_value.value) == int(self.state_key)


class ZWavePropertyBinarySensor(ZWaveBaseEntity, BinarySensorEntity):
    """Representation of a Z-Wave binary_sensor from a property."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        description: PropertyZWaveJSEntityDescription,
    ) -> None:
        """Initialize a ZWavePropertyBinarySensor entity."""
        super().__init__(config_entry, driver, info)
        self.entity_description: PropertyZWaveJSEntityDescription = description
        self._attr_name: str = self.generate_name(include_value_name=True)

    @property
    def is_on(self) -> Optional[bool]:
        """Return if the sensor is on or off."""
        if self.info.primary_value.value is None:
            return None
        return self.info.primary_value.value in self.entity_description.on_states


class ZWaveConfigParameterBinarySensor(ZWaveBooleanBinarySensor):
    """Representation of a Z-Wave config parameter binary sensor."""
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
    ) -> None:
        """Initialize a ZWaveConfigParameterBinarySensor entity."""
        super().__init__(config_entry, driver, info)
        property_key_name: Optional[str] = self.info.primary_value.property_key_name
        additional_info: Optional[List[str]] = [property_key_name] if property_key_name else None
        self._attr_name: str = self.generate_name(
            alternate_value_name=self.info.primary_value.property_name,
            additional_info=additional_info,
        )
