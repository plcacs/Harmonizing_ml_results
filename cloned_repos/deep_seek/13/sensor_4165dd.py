"""Sensor for Shelly."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, cast, Any, TypedDict, Optional, Union, Type, TypeVar, Dict, Tuple, List

from aioshelly.block_device import Block
from aioshelly.const import RPC_GENERATIONS
from homeassistant.components.sensor import (
    DOMAIN as SENSOR_PLATFORM, 
    RestoreSensor, 
    SensorDeviceClass, 
    SensorEntity, 
    SensorEntityDescription, 
    SensorExtraStoredData, 
    SensorStateClass
)
from homeassistant.const import (
    CONCENTRATION_PARTS_PER_MILLION, 
    DEGREE, 
    LIGHT_LUX, 
    PERCENTAGE, 
    SIGNAL_STRENGTH_DECIBELS_MILLIWATT, 
    EntityCategory, 
    UnitOfApparentPower, 
    UnitOfElectricCurrent, 
    UnitOfElectricPotential, 
    UnitOfEnergy, 
    UnitOfFrequency, 
    UnitOfPower, 
    UnitOfTemperature
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import RegistryEntry
from homeassistant.helpers.typing import StateType
from .const import CONF_SLEEP_PERIOD, ROLE_TO_DEVICE_CLASS_MAP, SHAIR_MAX_WORK_HOURS
from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry, ShellyRpcCoordinator
from .entity import (
    BlockEntityDescription, 
    RestEntityDescription, 
    RpcEntityDescription, 
    ShellyBlockAttributeEntity, 
    ShellyRestAttributeEntity, 
    ShellyRpcAttributeEntity, 
    ShellySleepingBlockAttributeEntity, 
    ShellySleepingRpcAttributeEntity, 
    async_setup_entry_attribute_entities, 
    async_setup_entry_rest, 
    async_setup_entry_rpc
)
from .utils import (
    async_remove_orphaned_entities, 
    get_device_entry_gen, 
    get_device_uptime, 
    get_virtual_component_ids, 
    is_rpc_wifi_stations_disabled
)

T = TypeVar('T')

@dataclass(frozen=True, kw_only=True)
class BlockSensorDescription(BlockEntityDescription, SensorEntityDescription):
    """Class to describe a BLOCK sensor."""

@dataclass(frozen=True, kw_only=True)
class RpcSensorDescription(RpcEntityDescription, SensorEntityDescription):
    """Class to describe a RPC sensor."""
    device_class_fn: Optional[Callable[[Dict[str, Any]], Optional[SensorDeviceClass]]] = None

@dataclass(frozen=True, kw_only=True)
class RestSensorDescription(RestEntityDescription, SensorEntityDescription):
    """Class to describe a REST sensor."""

class RpcSensor(ShellyRpcAttributeEntity, SensorEntity):
    """Represent a RPC sensor."""

    def __init__(
        self, 
        coordinator: ShellyRpcCoordinator, 
        key: str, 
        attribute: str, 
        description: RpcSensorDescription
    ) -> None:
        """Initialize select."""
        super().__init__(coordinator, key, attribute, description)
        if self.option_map:
            self._attr_options = list(self.option_map.values())
        if description.device_class_fn is not None:
            if (device_class := description.device_class_fn(coordinator.device.config[key])):
                self._attr_device_class = device_class

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        attribute_value = self.attribute_value
        if not self.option_map:
            return attribute_value
        if not isinstance(attribute_value, str):
            return None
        return self.option_map[attribute_value]

class RpcBluTrvSensor(RpcSensor):
    """Represent a RPC BluTrv sensor."""

    def __init__(
        self, 
        coordinator: ShellyRpcCoordinator, 
        key: str, 
        attribute: str, 
        description: RpcSensorDescription
    ) -> None:
        """Initialize."""
        super().__init__(coordinator, key, attribute, description)
        ble_addr = coordinator.device.config[key]['addr']
        self._attr_device_info = DeviceInfo(connections={(CONNECTION_BLUETOOTH, ble_addr)})

SENSORS: Dict[Tuple[str, str], BlockSensorDescription] = {
    ('device', 'battery'): BlockSensorDescription(
        key='device|battery', 
        name='Battery', 
        native_unit_of_measurement=PERCENTAGE, 
        device_class=SensorDeviceClass.BATTERY, 
        state_class=SensorStateClass.MEASUREMENT, 
        removal_condition=lambda settings, _: settings.get('external_power') == 1, 
        available=lambda block: cast(int, block.battery) != -1, 
        entity_category=EntityCategory.DIAGNOSTIC
    ),
    # ... (rest of the SENSORS dictionary remains the same)
}

REST_SENSORS: Dict[str, RestSensorDescription] = {
    'rssi': RestSensorDescription(
        key='rssi', 
        name='RSSI', 
        native_unit_of_measurement=SIGNAL_STRENGTH_DECIBELS_MILLIWATT, 
        value=lambda status, _: status['wifi_sta']['rssi'], 
        device_class=SensorDeviceClass.SIGNAL_STRENGTH, 
        state_class=SensorStateClass.MEASUREMENT, 
        entity_registry_enabled_default=False, 
        entity_category=EntityCategory.DIAGNOSTIC
    ),
    # ... (rest of the REST_SENSORS dictionary remains the same)
}

RPC_SENSORS: Dict[str, RpcSensorDescription] = {
    'power': RpcSensorDescription(
        key='switch', 
        sub_key='apower', 
        name='Power', 
        native_unit_of_measurement=UnitOfPower.WATT, 
        device_class=SensorDeviceClass.POWER, 
        state_class=SensorStateClass.MEASUREMENT
    ),
    # ... (rest of the RPC_SENSORS dictionary remains the same)
}

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up sensors for device."""
    if get_device_entry_gen(config_entry) in RPC_GENERATIONS:
        if config_entry.data[CONF_SLEEP_PERIOD]:
            async_setup_entry_rpc(hass, config_entry, async_add_entities, RPC_SENSORS, RpcSleepingSensor)
        else:
            coordinator = config_entry.runtime_data.rpc
            assert coordinator
            async_setup_entry_rpc(hass, config_entry, async_add_entities, RPC_SENSORS, RpcSensor)
            async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SENSOR_PLATFORM, coordinator.device.status)
            virtual_component_ids = get_virtual_component_ids(coordinator.device.config, SENSOR_PLATFORM)
            for component in ('enum', 'number', 'text'):
                async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SENSOR_PLATFORM, virtual_component_ids, component)
        return
    
    if config_entry.data[CONF_SLEEP_PERIOD]:
        async_setup_entry_attribute_entities(hass, config_entry, async_add_entities, SENSORS, BlockSleepingSensor)
    else:
        async_setup_entry_attribute_entities(hass, config_entry, async_add_entities, SENSORS, BlockSensor)
        async_setup_entry_rest(hass, config_entry, async_add_entities, REST_SENSORS, RestSensor)

class BlockSensor(ShellyBlockAttributeEntity, SensorEntity):
    """Represent a block sensor."""

    def __init__(
        self, 
        coordinator: ShellyBlockCoordinator, 
        block: Block, 
        attribute: str, 
        description: BlockSensorDescription
    ) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, block, attribute, description)
        self._attr_native_unit_of_measurement = description.native_unit_of_measurement

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        return self.attribute_value

class RestSensor(ShellyRestAttributeEntity, SensorEntity):
    """Represent a REST sensor."""

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        return self.attribute_value

class BlockSleepingSensor(ShellySleepingBlockAttributeEntity, RestoreSensor):
    """Represent a block sleeping sensor."""

    def __init__(
        self, 
        coordinator: ShellyBlockCoordinator, 
        block: Block, 
        attribute: str, 
        description: BlockSensorDescription, 
        entry: Optional[RegistryEntry] = None
    ) -> None:
        """Initialize the sleeping sensor."""
        super().__init__(coordinator, block, attribute, description, entry)
        self.restored_data: Optional[SensorExtraStoredData] = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self.restored_data = await self.async_get_last_sensor_data()

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        if self.block is not None:
            return self.attribute_value
        if self.restored_data is None:
            return None
        return cast(StateType, self.restored_data.native_value)

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement of the sensor, if any."""
        if self.block is not None:
            return self.entity_description.native_unit_of_measurement
        if self.restored_data is None:
            return None
        return self.restored_data.native_unit_of_measurement

class RpcSleepingSensor(ShellySleepingRpcAttributeEntity, RestoreSensor):
    """Represent a RPC sleeping sensor."""

    def __init__(
        self, 
        coordinator: ShellyRpcCoordinator, 
        key: str, 
        attribute: str, 
        description: RpcSensorDescription, 
        entry: Optional[RegistryEntry] = None
    ) -> None:
        """Initialize the sleeping sensor."""
        super().__init__(coordinator, key, attribute, description, entry)
        self.restored_data: Optional[SensorExtraStoredData] = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self.restored_data = await self.async_get_last_sensor_data()

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        if self.coordinator.device.initialized:
            return self.attribute_value
        if self.restored_data is None:
            return None
        return cast(StateType, self.restored_data.native_value)

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement of the sensor, if any."""
        return self.entity_description.native_unit_of_measurement
