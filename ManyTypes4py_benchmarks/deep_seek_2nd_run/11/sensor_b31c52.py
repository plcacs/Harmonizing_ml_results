"""Support for Homekit sensors."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Type, TypedDict, Union

from aiohomekit.model import Accessory, Transport
from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes
from aiohomekit.model.characteristics.const import CurrentAirPurifierStateValues, ThreadNodeCapabilities, ThreadStatus
from aiohomekit.model.services import Service, ServicesTypes
from homeassistant.components.bluetooth import async_ble_device_from_address, async_last_service_info
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, CONCENTRATION_PARTS_PER_MILLION, LIGHT_LUX, PERCENTAGE, SIGNAL_STRENGTH_DECIBELS_MILLIWATT, EntityCategory, Platform, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfPower, UnitOfPressure, UnitOfSoundPressure, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity, HomeKitEntity
from .utils import folded_name

class ServiceInfo(TypedDict):
    aid: int
    iid: int

@dataclass(frozen=True)
class HomeKitSensorEntityDescription(SensorEntityDescription):
    """Describes Homekit sensor."""
    probe: Optional[Callable[[Characteristic], bool]] = None
    format: Optional[Callable[[Characteristic], str]] = None
    enum: Optional[dict[IntEnum, str]] = None

def thread_node_capability_to_str(char: Characteristic) -> str:
    """Return the thread device type as a string."""
    val = ThreadNodeCapabilities(char.value)
    if val & ThreadNodeCapabilities.BORDER_ROUTER_CAPABLE:
        return 'border_router_capable'
    if val & ThreadNodeCapabilities.ROUTER_ELIGIBLE:
        return 'router_eligible'
    if val & ThreadNodeCapabilities.FULL:
        return 'full'
    if val & ThreadNodeCapabilities.MINIMAL:
        return 'minimal'
    if val & ThreadNodeCapabilities.SLEEPY:
        return 'sleepy'
    return 'none'

def thread_status_to_str(char: Characteristic) -> str:
    """Return the thread status as a string."""
    val = ThreadStatus(char.value)
    if val & ThreadStatus.BORDER_ROUTER:
        return 'border_router'
    if val & ThreadStatus.LEADER:
        return 'leader'
    if val & ThreadStatus.ROUTER:
        return 'router'
    if val & ThreadStatus.CHILD:
        return 'child'
    if val & ThreadStatus.JOINING:
        return 'joining'
    if val & ThreadStatus.DETACHED:
        return 'detached'
    return 'disabled'

SIMPLE_SENSOR: dict[str, HomeKitSensorEntityDescription] = {
    CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_WATT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_WATT,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT
    ),
    # ... (rest of the SIMPLE_SENSOR dictionary remains the same)
}

class HomeKitSensor(HomeKitEntity, SensorEntity):
    """Representation of a HomeKit sensor."""
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the device."""
        full_name = super().name
        default_name = self.default_name
        if default_name and full_name and (folded_name(default_name) not in folded_name(full_name)):
            return f'{full_name} {default_name}'
        return full_name or ""

class HomeKitHumiditySensor(HomeKitSensor):
    """Representation of a Homekit humidity sensor."""
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.HUMIDITY
    _attr_native_unit_of_measurement: str = PERCENTAGE

    def get_characteristic_types(self) -> list[str]:
        """Define the homekit characteristics the entity is tracking."""
        return [CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT]

    @property
    def default_name(self) -> str:
        """Return the default name of the device."""
        return 'Humidity'

    @property
    def native_value(self) -> Optional[float]:
        """Return the current humidity."""
        return self.service.value(CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT)

# ... (other sensor classes with similar type annotations)

class SimpleSensor(CharacteristicEntity, SensorEntity):
    """A simple sensor for a single characteristic."""
    entity_description: HomeKitSensorEntityDescription

    def __init__(self, conn: HKDevice, info: ServiceInfo, char: Characteristic, description: HomeKitSensorEntityDescription) -> None:
        """Initialise a secondary HomeKit characteristic sensor."""
        self.entity_description = description
        if self.entity_description.enum:
            self._attr_options = list(self.entity_description.enum.values())
        super().__init__(conn, info, char)

    def get_characteristic_types(self) -> list[str]:
        """Define the homekit characteristics the entity is tracking."""
        return [self._char.type]

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return f'{self.entity_description.name}'

    @property
    def native_value(self) -> Any:
        """Return the current sensor value."""
        if self.entity_description.enum:
            return self.entity_description.enum[self._char.value]
        if self.entity_description.format:
            return self.entity_description.format(self._char)
        return self._char.value

ENTITY_TYPES: dict[str, Type[HomeKitSensor]] = {
    ServicesTypes.HUMIDITY_SENSOR: HomeKitHumiditySensor,
    # ... (rest of ENTITY_TYPES dictionary)
}

REQUIRED_CHAR_BY_TYPE: dict[str, str] = {
    ServicesTypes.BATTERY_SERVICE: CharacteristicsTypes.BATTERY_LEVEL
}

class RSSISensor(HomeKitEntity, SensorEntity):
    """HomeKit Controller RSSI sensor."""
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.SIGNAL_STRENGTH
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default: bool = False
    _attr_has_entity_name: bool = True
    _attr_native_unit_of_measurement: str = SIGNAL_STRENGTH_DECIBELS_MILLIWATT
    _attr_should_poll: bool = False

    def __init__(self, accessory: Accessory, devinfo: ServiceInfo) -> None:
        """Initialise a HomeKit Controller RSSI sensor."""
        super().__init__(accessory, devinfo)
        self._attr_unique_id = f'{accessory.unique_id}_rssi'

    def get_characteristic_types(self) -> list[str]:
        """Define the homekit characteristics the entity cares about."""
        return []

    @property
    def available(self) -> bool:
        """Return if the bluetooth device is available."""
        address = self._accessory.pairing_data['AccessoryAddress']
        return async_ble_device_from_address(self.hass, address) is not None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return 'Signal strength'

    @property
    def old_unique_id(self) -> str:
        """Return the old ID of this device."""
        serial = self.accessory_info.value(CharacteristicsTypes.SERIAL_NUMBER)
        return f'homekit-{serial}-rssi'

    @property
    def native_value(self) -> Optional[int]:
        """Return the current rssi value."""
        address = self._accessory.pairing_data['AccessoryAddress']
        last_service_info = async_last_service_info(self.hass, address)
        return last_service_info.rssi if last_service_info else None

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Homekit sensors."""
    hkid = config_entry.data['AccessoryPairingID']
    conn = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_service(service: Service) -> bool:
        if not (entity_class := ENTITY_TYPES.get(service.type)):
            return False
        if (required_char := REQUIRED_CHAR_BY_TYPE.get(service.type)) and (not service.has(required_char)):
            return False
        info: ServiceInfo = {'aid': service.accessory.aid, 'iid': service.iid}
        entity = entity_class(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True
    conn.add_listener(async_add_service)

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        if not (description := SIMPLE_SENSOR.get(char.type)):
            return False
        if description.probe and (not description.probe(char)):
            return False
        info: ServiceInfo = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        entity = SimpleSensor(conn, info, char, description)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True
    conn.add_char_factory(async_add_characteristic)

    @callback
    def async_add_accessory(accessory: Accessory) -> bool:
        if conn.pairing.transport != Transport.BLE:
            return False
        accessory_info = accessory.services.first(service_type=ServicesTypes.ACCESSORY_INFORMATION)
        assert accessory_info
        info: ServiceInfo = {'aid': accessory.aid, 'iid': accessory_info.iid}
        entity = RSSISensor(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True
    conn.add_accessory_factory(async_add_accessory)
