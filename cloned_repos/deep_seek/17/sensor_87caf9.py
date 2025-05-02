"""Representation of Z-Wave sensors."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union, cast
import voluptuous as vol
from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import CommandClass
from zwave_js_server.const.command_class.meter import RESET_METER_OPTION_TARGET_VALUE, RESET_METER_OPTION_TYPE
from zwave_js_server.exceptions import BaseZwaveJSServerError
from zwave_js_server.model.controller import Controller
from zwave_js_server.model.controller.statistics import ControllerStatistics
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.node import Node as ZwaveNode
from zwave_js_server.model.node.statistics import NodeStatistics
from zwave_js_server.util.command_class.meter import get_meter_type
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONCENTRATION_PARTS_PER_MILLION, LIGHT_LUX, PERCENTAGE, SIGNAL_STRENGTH_DECIBELS_MILLIWATT, UV_INDEX, EntityCategory, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfPower, UnitOfPressure, UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_platform
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import UNDEFINED, StateType
from .binary_sensor import is_valid_notification_binary_sensor
from .const import ATTR_METER_TYPE, ATTR_METER_TYPE_NAME, ATTR_VALUE, DATA_CLIENT, DOMAIN, ENTITY_DESC_KEY_BATTERY, ENTITY_DESC_KEY_CO, ENTITY_DESC_KEY_CO2, ENTITY_DESC_KEY_CURRENT, ENTITY_DESC_KEY_ENERGY_MEASUREMENT, ENTITY_DESC_KEY_ENERGY_PRODUCTION_POWER, ENTITY_DESC_KEY_ENERGY_PRODUCTION_TIME, ENTITY_DESC_KEY_ENERGY_PRODUCTION_TODAY, ENTITY_DESC_KEY_ENERGY_PRODUCTION_TOTAL, ENTITY_DESC_KEY_ENERGY_TOTAL_INCREASING, ENTITY_DESC_KEY_HUMIDITY, ENTITY_DESC_KEY_ILLUMINANCE, ENTITY_DESC_KEY_MEASUREMENT, ENTITY_DESC_KEY_POWER, ENTITY_DESC_KEY_POWER_FACTOR, ENTITY_DESC_KEY_PRESSURE, ENTITY_DESC_KEY_SIGNAL_STRENGTH, ENTITY_DESC_KEY_TARGET_TEMPERATURE, ENTITY_DESC_KEY_TEMPERATURE, ENTITY_DESC_KEY_TOTAL_INCREASING, ENTITY_DESC_KEY_UV_INDEX, ENTITY_DESC_KEY_VOLTAGE, LOGGER, SERVICE_RESET_METER
from .discovery import ZwaveDiscoveryInfo
from .discovery_data_template import NumericSensorDataTemplate, NumericSensorDataTemplateData
from .entity import ZWaveBaseEntity
from .helpers import get_device_info, get_valueless_base_unique_id
from .migrate import async_migrate_statistics_sensors

PARALLEL_UPDATES = 0

ENTITY_DESCRIPTION_KEY_DEVICE_CLASS_MAP: dict[tuple[str, str], SensorEntityDescription] = {
    (ENTITY_DESC_KEY_BATTERY, PERCENTAGE): SensorEntityDescription(
        key=ENTITY_DESC_KEY_BATTERY,
        device_class=SensorDeviceClass.BATTERY,
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE
    ),
    # ... (rest of the ENTITY_DESCRIPTION_KEY_DEVICE_CLASS_MAP entries remain the same)
}

ENTITY_DESCRIPTION_KEY_MAP: dict[str, SensorEntityDescription] = {
    ENTITY_DESC_KEY_CO: SensorEntityDescription(
        key=ENTITY_DESC_KEY_CO,
        state_class=SensorStateClass.MEASUREMENT
    ),
    # ... (rest of the ENTITY_DESCRIPTION_KEY_MAP entries remain the same)
}

def convert_nested_attr(statistics: Any, key: str) -> Any:
    """Convert a string that represents a nested attr to a value."""
    data = statistics
    for _key in key.split('.'):
        if data is None:
            return None
        data = getattr(data, _key)
    return data

@dataclass(frozen=True, kw_only=True)
class ZWaveJSStatisticsSensorEntityDescription(SensorEntityDescription):
    """Class to represent a Z-Wave JS statistics sensor entity description."""
    convert: Callable[[Any, str], Any] = getattr
    entity_registry_enabled_default: bool = False

ENTITY_DESCRIPTION_CONTROLLER_STATISTICS_LIST: list[ZWaveJSStatisticsSensorEntityDescription] = [
    # ... (all controller statistics entries remain the same)
]

CONTROLLER_STATISTICS_KEY_MAP: dict[str, str] = {
    # ... (all controller statistics key mappings remain the same)
}

ENTITY_DESCRIPTION_NODE_STATISTICS_LIST: list[ZWaveJSStatisticsSensorEntityDescription] = [
    # ... (all node statistics entries remain the same)
]

NODE_STATISTICS_KEY_MAP: dict[str, str] = {
    # ... (all node statistics key mappings remain the same)
}

def get_entity_description(data: NumericSensorDataTemplateData) -> SensorEntityDescription:
    """Return the entity description for the given data."""
    data_description_key = data.entity_description_key or ''
    data_unit = data.unit_of_measurement or ''
    return ENTITY_DESCRIPTION_KEY_DEVICE_CLASS_MAP.get(
        (data_description_key, data_unit),
        ENTITY_DESCRIPTION_KEY_MAP.get(
            data_description_key,
            SensorEntityDescription(
                key='base_sensor',
                native_unit_of_measurement=data.unit_of_measurement
            )
        )
    )

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up Z-Wave sensor from config entry."""
    client: ZwaveClient = config_entry.runtime_data[DATA_CLIENT]
    driver: Driver = client.driver
    assert driver is not None

    @callback
    def async_add_sensor(info: ZwaveDiscoveryInfo) -> None:
        """Add Z-Wave Sensor."""
        entities: list[ZwaveSensor] = []
        if info.platform_data:
            data = info.platform_data
        else:
            data = NumericSensorDataTemplateData()
        entity_description = get_entity_description(data)
        if info.platform_hint == 'numeric_sensor':
            entities.append(ZWaveNumericSensor(config_entry, driver, info, entity_description, data.unit_of_measurement))
        elif info.platform_hint == 'notification':
            if is_valid_notification_binary_sensor(info):
                return
            entities.append(ZWaveListSensor(config_entry, driver, info, entity_description))
        elif info.platform_hint == 'config_parameter':
            entities.append(ZWaveConfigParameterSensor(config_entry, driver, info, entity_description))
        elif info.platform_hint == 'meter':
            entities.append(ZWaveMeterSensor(config_entry, driver, info, entity_description))
        else:
            entities.append(ZwaveSensor(config_entry, driver, info, entity_description))
        async_add_entities(entities)

    @callback
    def async_add_controller_status_sensor() -> None:
        """Add controller status sensor."""
        async_add_entities([ZWaveControllerStatusSensor(config_entry, driver)])

    @callback
    def async_add_node_status_sensor(node: ZwaveNode) -> None:
        """Add node status sensor."""
        async_add_entities([ZWaveNodeStatusSensor(config_entry, driver, node)])

    @callback
    def async_add_statistics_sensors(node: ZwaveNode) -> None:
        """Add statistics sensors."""
        async_migrate_statistics_sensors(hass, driver, node, CONTROLLER_STATISTICS_KEY_MAP if driver.controller.own_node == node else NODE_STATISTICS_KEY_MAP)
        async_add_entities([
            ZWaveStatisticsSensor(
                config_entry,
                driver,
                driver.controller if driver.controller.own_node == node else node,
                entity_description
            ) for entity_description in (
                ENTITY_DESCRIPTION_CONTROLLER_STATISTICS_LIST if driver.controller.own_node == node
                else ENTITY_DESCRIPTION_NODE_STATISTICS_LIST
            )
        ])

    config_entry.async_on_unload(async_dispatcher_connect(hass, f'{DOMAIN}_{config_entry.entry_id}_add_{SENSOR_DOMAIN}', async_add_sensor))
    config_entry.async_on_unload(async_dispatcher_connect(hass, f'{DOMAIN}_{config_entry.entry_id}_add_controller_status_sensor', async_add_controller_status_sensor))
    config_entry.async_on_unload(async_dispatcher_connect(hass, f'{DOMAIN}_{config_entry.entry_id}_add_node_status_sensor', async_add_node_status_sensor))
    config_entry.async_on_unload(async_dispatcher_connect(hass, f'{DOMAIN}_{config_entry.entry_id}_add_statistics_sensors', async_add_statistics_sensors))
    
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_RESET_METER,
        {
            vol.Optional(ATTR_METER_TYPE): vol.Coerce(int),
            vol.Optional(ATTR_VALUE): vol.Coerce(int)
        },
        'async_reset_meter'
    )

class ZwaveSensor(ZWaveBaseEntity, SensorEntity):
    """Basic Representation of a Z-Wave sensor."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        entity_description: SensorEntityDescription,
        unit_of_measurement: Optional[str] = None
    ) -> None:
        """Initialize a ZWaveSensorBase entity."""
        self.entity_description = entity_description
        super().__init__(config_entry, driver, info)
        self._attr_native_unit_of_measurement = unit_of_measurement
        self._attr_force_update = True
        if not entity_description.name or entity_description.name is UNDEFINED:
            self._attr_name = self.generate_name(include_value_name=True)

    @property
    def native_value(self) -> StateType:
        """Return state of the sensor."""
        key = str(self.info.primary_value.value)
        if key not in self.info.primary_value.metadata.states:
            return self.info.primary_value.value
        return str(self.info.primary_value.metadata.states[key])

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return unit of measurement the value is expressed in."""
        if (unit := super().native_unit_of_measurement) is not None:
            return unit
        if self.info.primary_value.metadata.unit is None:
            return None
        return str(self.info.primary_value.metadata.unit)

class ZWaveNumericSensor(ZwaveSensor):
    """Representation of a Z-Wave Numeric sensor."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        entity_description: SensorEntityDescription,
        unit_of_measurement: Optional[str] = None
    ) -> None:
        """Initialize a ZWaveBasicSensor entity."""
        super().__init__(config_entry, driver, info, entity_description, unit_of_measurement)
        if self.info.primary_value.command_class == CommandClass.BASIC:
            self._attr_name = self.generate_name(include_value_name=True, alternate_value_name='Basic')

    @callback
    def on_value_update(self) -> None:
        """Handle scale changes for this value on value updated event."""
        data = NumericSensorDataTemplate().resolve_data(self.info.primary_value)
        self.entity_description = get_entity_description(data)
        self._attr_native_unit_of_measurement = data.unit_of_measurement

    @property
    def native_value(self) -> float:
        """Return state of the sensor."""
        if self.info.primary_value.value is None:
            return 0
        return float(self.info.primary_value.value)

class ZWaveMeterSensor(ZWaveNumericSensor):
    """Representation of a Z-Wave Meter CC sensor."""

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        meter_type = get_meter_type(self.info.primary_value)
        return {
            ATTR_METER_TYPE: meter_type.value,
            ATTR_METER_TYPE_NAME: meter_type.name
        }

    async def async_reset_meter(self, meter_type: Optional[int] = None, value: Optional[int] = None) -> None:
        """Reset meter(s) on device."""
        node = self.info.node
        endpoint = self.info.primary_value.endpoint or 0
        options = {}
        if meter_type is not None:
            options[RESET_METER_OPTION_TYPE] = meter_type
        if value is not None:
            options[RESET_METER_OPTION_TARGET_VALUE] = value
        args = [options] if options else []
        try:
            await node.endpoints[endpoint].async_invoke_cc_api(
                CommandClass.METER,
                'reset',
                *args,
                wait_for_result=False
            )
        except BaseZwaveJSServerError as err:
            raise HomeAssistantError(f'Failed to reset meters on node {node} endpoint {endpoint}: {err}') from err
        LOGGER.debug('Meters on node %s endpoint %s reset with the following options: %s', node, endpoint, options)

class ZWaveListSensor(ZwaveSensor):
    """Representation of a Z-Wave Numeric sensor with multiple states."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        entity_description: SensorEntityDescription,
        unit_of_measurement: Optional[str] = None
    ) -> None:
        """Initialize a ZWaveListSensor entity."""
        super().__init__(config_entry, driver, info, entity_description, unit_of_measurement)
        self._attr_name = self.generate_name(
            alternate_value_name=self.info.primary_value.property_name,
            additional_info=[self.info.primary_value.property_key_name]
        )
        if self.info.primary_value.metadata.states:
            self._attr_device_class = SensorDeviceClass.ENUM
            self._attr_options = list(info.primary_value.metadata.states.values())

    @property
    def extra_state_attributes(self) -> Optional[dict[str, Any]]:
        """Return the device specific state attributes."""
        if (value := self.info.primary_value.value) is None:
            return None
        return {ATTR_VALUE: value}

class ZWaveConfigParameterSensor(ZWaveListSensor):
    """Representation of a Z-Wave config parameter sensor."""
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        config_entry: ConfigEntry,
        driver: Driver,
        info: ZwaveDiscoveryInfo,
        entity_description: SensorEntityDescription,
        unit_of_measurement: Optional[str] = None
    ) -> None:
        """Initialize a ZWaveConfigParameterSensor entity."""
        super().__init__(config_entry, driver, info, entity_description, unit_of_measurement)
        property_key_name = self.info.primary_value.property_key_name
        self._attr_name = self.generate_name(
            alternate_value_name=self.info.primary_value.property_name,
            additional_info=[property_key_name] if property_key_name else None
        )

    @property
    def extra_state_attributes(self) -> Optional[dict[str, Any]]:
        """Return the device specific state attributes."""
        if (value := self.info.primary_value.value) is None:
            return None
        return {ATTR_VALUE: value}

class ZWaveNodeStatusSensor(SensorEntity):
    """Representation of a node status sensor."""
    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True
    _attr_translation_key = 'node_status'

    def __init__(self, config_entry: ConfigEntry, driver: Driver, node: ZwaveNode) -> None:
        """Initialize a generic Z-Wave device entity."""
        self.config_entry = config_entry
        self.node = node
        self._base_unique_id = get_valueless_base_unique_id(driver, node)
        self._attr_unique_id = f'{self._base_unique_id}.node_status'
        self._attr_device_info = get_device_info(driver, node)

    async def async_poll_value(self, _: bool) -> None:
        """Poll a value."""
        LOGGER.error("There is no value to refresh for this entity so the zwave_js.refresh_value service won't work for it")

    @callback
    def _status_changed(self, _: dict[str, Any]) -> None:
        """Call when status event is received."""
        self._attr_native_value = self.node.status.name.lower()
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Call when entity is added."""
        for evt in ('wake up', 'sleep', 'dead', 'alive'):
            self.async_on_remove(self.node.on(evt, self._status_changed))
        self.async_on_remove(async_dispatcher_connect(
            self.hass,
            f'{DOMAIN}_{self.unique_id}_poll_value',
            self.async_poll_value
        ))
        self.async_on_remove(async_dispatcher_connect(
            self.hass,
            f'{DOMAIN}_{self._base_unique_id}_remove_entity',
            self.async_remove
        ))
        self._attr_native_value = self.node.status.name.lower()
        self.async_write_ha_state()

class ZWaveControllerStatusSensor(SensorEntity):
    """Representation of a controller status sensor."""
