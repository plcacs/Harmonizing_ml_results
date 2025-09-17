from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Mapping, Dict, List
from aiohomekit.model import Accessory, Transport
from aiohomekit.model.characteristics import Characteristic, CharacteristicsTypes
from aiohomekit.model.characteristics.const import (
    CurrentAirPurifierStateValues,
    ThreadNodeCapabilities,
    ThreadStatus,
)
from aiohomekit.model.services import Service, ServicesTypes
from homeassistant.components.bluetooth import async_ble_device_from_address, async_last_service_info
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    CONCENTRATION_PARTS_PER_MILLION,
    LIGHT_LUX,
    PERCENTAGE,
    SIGNAL_STRENGTH_DECIBELS_MILLIWATT,
    EntityCategory,
    Platform,
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfPressure,
    UnitOfSoundPressure,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from . import KNOWN_DEVICES
from .connection import HKDevice
from .entity import CharacteristicEntity, HomeKitEntity
from .utils import folded_name


@dataclass(frozen=True)
class HomeKitSensorEntityDescription(SensorEntityDescription):
    probe: Optional[Callable[[Characteristic], bool]] = None
    format: Optional[Callable[[Characteristic], Any]] = None
    enum: Optional[Mapping[Any, str]] = None


def thread_node_capability_to_str(char: Characteristic) -> str:
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


SIMPLE_SENSOR: Mapping[str, HomeKitSensorEntityDescription] = {
    CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_WATT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_WATT,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
    ),
    CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_AMPS: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_AMPS,
        name='Current',
        device_class=SensorDeviceClass.CURRENT,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfElectricCurrent.AMPERE,
    ),
    CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_AMPS_20: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_AMPS_20,
        name='Current',
        device_class=SensorDeviceClass.CURRENT,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfElectricCurrent.AMPERE,
    ),
    CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_KW_HOUR: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_CONNECTSENSE_ENERGY_KW_HOUR,
        name='Energy kWh',
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
    ),
    CharacteristicsTypes.VENDOR_EVE_ENERGY_WATT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_ENERGY_WATT,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
    ),
    CharacteristicsTypes.VENDOR_EVE_ENERGY_KW_HOUR: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_ENERGY_KW_HOUR,
        name='Energy kWh',
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
    ),
    CharacteristicsTypes.VENDOR_EVE_ENERGY_VOLTAGE: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_ENERGY_VOLTAGE,
        name='Volts',
        device_class=SensorDeviceClass.VOLTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfElectricPotential.VOLT,
    ),
    CharacteristicsTypes.VENDOR_EVE_ENERGY_AMPERE: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_ENERGY_AMPERE,
        name='Amps',
        device_class=SensorDeviceClass.CURRENT,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfElectricCurrent.AMPERE,
    ),
    CharacteristicsTypes.VENDOR_KOOGEEK_REALTIME_ENERGY: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_KOOGEEK_REALTIME_ENERGY,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
    ),
    CharacteristicsTypes.VENDOR_KOOGEEK_REALTIME_ENERGY_2: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_KOOGEEK_REALTIME_ENERGY_2,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
    ),
    CharacteristicsTypes.VENDOR_EVE_DEGREE_AIR_PRESSURE: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_DEGREE_AIR_PRESSURE,
        name='Air Pressure',
        device_class=SensorDeviceClass.PRESSURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPressure.HPA,
    ),
    CharacteristicsTypes.VENDOR_VOCOLINC_OUTLET_ENERGY: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_VOCOLINC_OUTLET_ENERGY,
        name='Power',
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
    ),
    CharacteristicsTypes.TEMPERATURE_CURRENT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.TEMPERATURE_CURRENT,
        name='Current Temperature',
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        probe=lambda char: char.service.type != ServicesTypes.TEMPERATURE_SENSOR,
    ),
    CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT,
        name='Current Humidity',
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        probe=lambda char: char.service.type != ServicesTypes.HUMIDITY_SENSOR,
    ),
    CharacteristicsTypes.AIR_QUALITY: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.AIR_QUALITY,
        name='Air Quality',
        device_class=SensorDeviceClass.AQI,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    CharacteristicsTypes.DENSITY_PM25: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_PM25,
        name='PM2.5 Density',
        device_class=SensorDeviceClass.PM25,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.DENSITY_PM10: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_PM10,
        name='PM10 Density',
        device_class=SensorDeviceClass.PM10,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.DENSITY_OZONE: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_OZONE,
        name='Ozone Density',
        device_class=SensorDeviceClass.OZONE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.DENSITY_NO2: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_NO2,
        name='Nitrogen Dioxide Density',
        device_class=SensorDeviceClass.NITROGEN_DIOXIDE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.DENSITY_SO2: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_SO2,
        name='Sulphur Dioxide Density',
        device_class=SensorDeviceClass.SULPHUR_DIOXIDE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.DENSITY_VOC: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.DENSITY_VOC,
        name='Volatile Organic Compound Density',
        device_class=SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    CharacteristicsTypes.THREAD_NODE_CAPABILITIES: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.THREAD_NODE_CAPABILITIES,
        name='Thread Capabilities',
        entity_category=EntityCategory.DIAGNOSTIC,
        format=thread_node_capability_to_str,
        device_class=SensorDeviceClass.ENUM,
        options=['border_router_capable', 'full', 'minimal', 'none', 'router_eligible', 'sleepy'],
        translation_key='thread_node_capabilities',
    ),
    CharacteristicsTypes.THREAD_STATUS: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.THREAD_STATUS,
        name='Thread Status',
        entity_category=EntityCategory.DIAGNOSTIC,
        format=thread_status_to_str,
        device_class=SensorDeviceClass.ENUM,
        options=['border_router', 'child', 'detached', 'disabled', 'joining', 'leader', 'router'],
        translation_key='thread_status',
    ),
    CharacteristicsTypes.AIR_PURIFIER_STATE_CURRENT: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.AIR_PURIFIER_STATE_CURRENT,
        name='Air Purifier Status',
        entity_category=EntityCategory.DIAGNOSTIC,
        device_class=SensorDeviceClass.ENUM,
        enum={
            CurrentAirPurifierStateValues.INACTIVE: 'inactive',
            CurrentAirPurifierStateValues.IDLE: 'idle',
            CurrentAirPurifierStateValues.ACTIVE: 'purifying',
        },
        translation_key='air_purifier_state_current',
    ),
    CharacteristicsTypes.VENDOR_NETATMO_NOISE: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_NETATMO_NOISE,
        name='Noise',
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfSoundPressure.DECIBEL,
        device_class=SensorDeviceClass.SOUND_PRESSURE,
    ),
    CharacteristicsTypes.FILTER_LIFE_LEVEL: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.FILTER_LIFE_LEVEL,
        name='Filter lifetime',
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
    ),
    CharacteristicsTypes.VENDOR_EVE_THERMO_VALVE_POSITION: HomeKitSensorEntityDescription(
        key=CharacteristicsTypes.VENDOR_EVE_THERMO_VALVE_POSITION,
        name='Valve position',
        translation_key='valve_position',
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
    ),
}


class HomeKitSensor(HomeKitEntity, SensorEntity):
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        full_name: str = super().name
        default_name: Optional[str] = self.default_name
        if default_name and full_name and (folded_name(default_name) not in folded_name(full_name)):
            return f'{full_name} {default_name}'
        return full_name


class HomeKitHumiditySensor(HomeKitSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.HUMIDITY
    _attr_native_unit_of_measurement: str = PERCENTAGE

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT]

    @property
    def default_name(self) -> str:
        return 'Humidity'

    @property
    def native_value(self) -> Any:
        return self.service.value(CharacteristicsTypes.RELATIVE_HUMIDITY_CURRENT)


class HomeKitTemperatureSensor(HomeKitSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.TEMPERATURE_CURRENT]

    @property
    def default_name(self) -> str:
        return 'Temperature'

    @property
    def native_value(self) -> Any:
        return self.service.value(CharacteristicsTypes.TEMPERATURE_CURRENT)


class HomeKitLightSensor(HomeKitSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ILLUMINANCE
    _attr_native_unit_of_measurement: str = LIGHT_LUX

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.LIGHT_LEVEL_CURRENT]

    @property
    def default_name(self) -> str:
        return 'Light Level'

    @property
    def native_value(self) -> Any:
        return self.service.value(CharacteristicsTypes.LIGHT_LEVEL_CURRENT)


class HomeKitCarbonDioxideSensor(HomeKitSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.CO2
    _attr_native_unit_of_measurement: str = CONCENTRATION_PARTS_PER_MILLION

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.CARBON_DIOXIDE_LEVEL]

    @property
    def default_name(self) -> str:
        return 'Carbon Dioxide'

    @property
    def native_value(self) -> Any:
        return self.service.value(CharacteristicsTypes.CARBON_DIOXIDE_LEVEL)


class HomeKitBatterySensor(HomeKitSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.BATTERY
    _attr_native_unit_of_measurement: str = PERCENTAGE
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC

    def get_characteristic_types(self) -> List[str]:
        return [CharacteristicsTypes.BATTERY_LEVEL, CharacteristicsTypes.STATUS_LO_BATT, CharacteristicsTypes.CHARGING_STATE]

    @property
    def default_name(self) -> str:
        return 'Battery'

    @property
    def icon(self) -> str:
        native_value: Optional[Any] = self.native_value
        if not self.available or native_value is None:
            return 'mdi:battery-unknown'
        icon: str = 'mdi:battery'
        is_charging: bool = self.is_charging
        if is_charging and native_value > 10:
            percentage: int = int(round(native_value / 20 - 0.01)) * 20
            icon += f'-charging-{percentage}'
        elif is_charging:
            icon += '-outline'
        elif self.is_low_battery:
            icon += '-alert'
        elif native_value < 95:
            percentage = max(int(round(native_value / 10 - 0.01)) * 10, 10)
            icon += f'-{percentage}'
        return icon

    @property
    def is_low_battery(self) -> bool:
        return self.service.value(CharacteristicsTypes.STATUS_LO_BATT) == 1

    @property
    def is_charging(self) -> bool:
        return self.service.value(CharacteristicsTypes.CHARGING_STATE) == 1

    @property
    def native_value(self) -> Any:
        return self.service.value(CharacteristicsTypes.BATTERY_LEVEL)


class SimpleSensor(CharacteristicEntity, SensorEntity):
    def __init__(
        self,
        conn: HKDevice,
        info: Dict[str, Any],
        char: Characteristic,
        description: HomeKitSensorEntityDescription,
    ) -> None:
        self.entity_description: HomeKitSensorEntityDescription = description
        if self.entity_description.enum:
            self._attr_options = list(self.entity_description.enum.values())
        super().__init__(conn, info, char)

    def get_characteristic_types(self) -> List[str]:
        return [self._char.type]

    @property
    def name(self) -> str:
        if (name := self.accessory.name):
            return f'{name} {self.entity_description.name}'
        return f'{self.entity_description.name}'

    @property
    def native_value(self) -> Any:
        if self.entity_description.enum:
            return self.entity_description.enum[self._char.value]
        if self.entity_description.format:
            return self.entity_description.format(self._char)
        return self._char.value


ENTITY_TYPES: Mapping[str, Any] = {
    ServicesTypes.HUMIDITY_SENSOR: HomeKitHumiditySensor,
    ServicesTypes.TEMPERATURE_SENSOR: HomeKitTemperatureSensor,
    ServicesTypes.LIGHT_SENSOR: HomeKitLightSensor,
    ServicesTypes.CARBON_DIOXIDE_SENSOR: HomeKitCarbonDioxideSensor,
    ServicesTypes.BATTERY_SERVICE: HomeKitBatterySensor,
}
REQUIRED_CHAR_BY_TYPE: Mapping[str, str] = {ServicesTypes.BATTERY_SERVICE: CharacteristicsTypes.BATTERY_LEVEL}


class RSSISensor(HomeKitEntity, SensorEntity):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.SIGNAL_STRENGTH
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default: bool = False
    _attr_has_entity_name: bool = True
    _attr_native_unit_of_measurement: str = SIGNAL_STRENGTH_DECIBELS_MILLIWATT
    _attr_should_poll: bool = False

    def __init__(self, accessory: Accessory, devinfo: Dict[str, Any]) -> None:
        super().__init__(accessory, devinfo)
        self._attr_unique_id = f'{accessory.unique_id}_rssi'

    def get_characteristic_types(self) -> List[str]:
        return []

    @property
    def available(self) -> bool:
        address: str = self._accessory.pairing_data['AccessoryAddress']
        return async_ble_device_from_address(self.hass, address) is not None

    @property
    def name(self) -> str:
        return 'Signal strength'

    @property
    def old_unique_id(self) -> str:
        serial: Any = self.accessory_info.value(CharacteristicsTypes.SERIAL_NUMBER)
        return f'homekit-{serial}-rssi'

    @property
    def native_value(self) -> Optional[int]:
        address: str = self._accessory.pairing_data['AccessoryAddress']
        last_service_info = async_last_service_info(self.hass, address)
        return last_service_info.rssi if last_service_info else None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    hkid: str = config_entry.data['AccessoryPairingID']
    conn: HKDevice = hass.data[KNOWN_DEVICES][hkid]

    @callback
    def async_add_service(service: Service) -> bool:
        entity_class: Optional[Any] = ENTITY_TYPES.get(service.type)
        if not entity_class:
            return False
        required_char: Optional[str] = REQUIRED_CHAR_BY_TYPE.get(service.type)
        if required_char and (not service.has(required_char)):
            return False
        info: Dict[str, Any] = {'aid': service.accessory.aid, 'iid': service.iid}
        entity = entity_class(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True

    conn.add_listener(async_add_service)

    @callback
    def async_add_characteristic(char: Characteristic) -> bool:
        description: Optional[HomeKitSensorEntityDescription] = SIMPLE_SENSOR.get(char.type)
        if not description:
            return False
        if description.probe and (not description.probe(char)):
            return False
        info: Dict[str, Any] = {'aid': char.service.accessory.aid, 'iid': char.service.iid}
        entity = SimpleSensor(conn, info, char, description)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True

    conn.add_char_factory(async_add_characteristic)

    @callback
    def async_add_accessory(accessory: Accessory) -> bool:
        if conn.pairing.transport != Transport.BLE:
            return False
        accessory_info: Optional[Any] = accessory.services.first(service_type=ServicesTypes.ACCESSORY_INFORMATION)
        assert accessory_info
        info: Dict[str, Any] = {'aid': accessory.aid, 'iid': accessory_info.iid}
        entity = RSSISensor(conn, info)
        conn.async_migrate_unique_id(entity.old_unique_id, entity.unique_id, Platform.SENSOR)
        async_add_entities([entity])
        return True

    conn.add_accessory_factory(async_add_accessory)