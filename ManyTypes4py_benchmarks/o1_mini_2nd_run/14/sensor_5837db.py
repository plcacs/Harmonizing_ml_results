"""Support for ISY sensors."""
from __future__ import annotations
from typing import Any, cast, Dict, List, Optional, Tuple, Union
from pyisy.constants import (
    ATTR_ACTION,
    ATTR_CONTROL,
    COMMAND_FRIENDLY_NAME,
    ISY_VALUE_UNKNOWN,
    NC_NODE_ENABLED,
    PROP_BATTERY_LEVEL,
    PROP_COMMS_ERROR,
    PROP_ENERGY_MODE,
    PROP_HEAT_COOL_STATE,
    PROP_HUMIDITY,
    PROP_ON_LEVEL,
    PROP_RAMP_RATE,
    PROP_STATUS,
    PROP_TEMPERATURE,
    TAG_ADDRESS,
)
from pyisy.helpers import EventListener, NodeProperty
from pyisy.nodes import Node, NodeChangedEvent
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, Platform, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import (
    _LOGGER,
    DOMAIN,
    UOM_DOUBLE_TEMP,
    UOM_FRIENDLY_NAME,
    UOM_INDEX,
    UOM_ON_OFF,
    UOM_TO_STATES,
)
from .entity import ISYNodeEntity
from .helpers import convert_isy_value_to_hass
from .models import IsyData

AUX_DISABLED_BY_DEFAULT_MATCH: List[str] = ['GV', 'DO']
AUX_DISABLED_BY_DEFAULT_EXACT: set = {
    PROP_COMMS_ERROR,
    PROP_ENERGY_MODE,
    PROP_HEAT_COOL_STATE,
    PROP_ON_LEVEL,
    PROP_RAMP_RATE,
    PROP_STATUS,
}
ISY_CONTROL_TO_DEVICE_CLASS: Dict[str, SensorDeviceClass] = {
    PROP_BATTERY_LEVEL: SensorDeviceClass.BATTERY,
    PROP_HUMIDITY: SensorDeviceClass.HUMIDITY,
    PROP_TEMPERATURE: SensorDeviceClass.TEMPERATURE,
    'BARPRES': SensorDeviceClass.ATMOSPHERIC_PRESSURE,
    'CC': SensorDeviceClass.CURRENT,
    'CO2LVL': SensorDeviceClass.CO2,
    'CPW': SensorDeviceClass.POWER,
    'CV': SensorDeviceClass.VOLTAGE,
    'DEWPT': SensorDeviceClass.TEMPERATURE,
    'DISTANC': SensorDeviceClass.DISTANCE,
    'ETO': SensorDeviceClass.PRECIPITATION_INTENSITY,
    'FATM': SensorDeviceClass.WEIGHT,
    'FREQ': SensorDeviceClass.FREQUENCY,
    'MUSCLEM': SensorDeviceClass.WEIGHT,
    'PF': SensorDeviceClass.POWER_FACTOR,
    'PM10': SensorDeviceClass.PM10,
    'PM25': SensorDeviceClass.PM25,
    'PRECIP': SensorDeviceClass.PRECIPITATION,
    'RAINRT': SensorDeviceClass.PRECIPITATION_INTENSITY,
    'RFSS': SensorDeviceClass.SIGNAL_STRENGTH,
    'SOILH': SensorDeviceClass.MOISTURE,
    'SOILT': SensorDeviceClass.TEMPERATURE,
    'SOLRAD': SensorDeviceClass.IRRADIANCE,
    'SPEED': SensorDeviceClass.SPEED,
    'TEMPEXH': SensorDeviceClass.TEMPERATURE,
    'TEMPOUT': SensorDeviceClass.TEMPERATURE,
    'TPW': SensorDeviceClass.ENERGY,
    'WATERP': SensorDeviceClass.PRESSURE,
    'WATERT': SensorDeviceClass.TEMPERATURE,
    'WATERTB': SensorDeviceClass.TEMPERATURE,
    'WATERTD': SensorDeviceClass.TEMPERATURE,
    'WEIGHT': SensorDeviceClass.WEIGHT,
    'WINDCH': SensorDeviceClass.TEMPERATURE,
}
ISY_CONTROL_TO_STATE_CLASS: Dict[str, SensorStateClass] = {
    control: SensorStateClass.MEASUREMENT for control in ISY_CONTROL_TO_DEVICE_CLASS
}
ISY_CONTROL_TO_ENTITY_CATEGORY: Dict[str, EntityCategory] = {
    PROP_RAMP_RATE: EntityCategory.DIAGNOSTIC,
    PROP_ON_LEVEL: EntityCategory.DIAGNOSTIC,
    PROP_COMMS_ERROR: EntityCategory.DIAGNOSTIC,
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the ISY sensor platform."""
    isy_data: IsyData = hass.data[DOMAIN][entry.entry_id]
    entities: List[SensorEntity] = []
    devices: Dict[str, DeviceInfo] = isy_data.devices
    for node in isy_data.nodes[Platform.SENSOR]:
        _LOGGER.debug('Loading %s', node.name)
        entities.append(ISYSensorEntity(node, devices.get(node.primary_node)))
    aux_sensors_list: List[Tuple[Node, str]] = isy_data.aux_properties[Platform.SENSOR]
    for node, control in aux_sensors_list:
        _LOGGER.debug('Loading %s %s', node.name, COMMAND_FRIENDLY_NAME.get(control))
        enabled_default: bool = (
            control not in AUX_DISABLED_BY_DEFAULT_EXACT
            and not any(control.startswith(match) for match in AUX_DISABLED_BY_DEFAULT_MATCH)
        )
        entities.append(
            ISYAuxSensorEntity(
                node=node,
                control=control,
                enabled_default=enabled_default,
                unique_id=f'{isy_data.uid_base(node)}_{control}',
                device_info=devices.get(node.primary_node),
            )
        )
    async_add_entities(entities)


class ISYSensorEntity(ISYNodeEntity, SensorEntity):
    """Representation of an ISY sensor device."""

    @property
    def target(self) -> Optional[Node]:
        """Return target for the sensor."""
        return self._node

    @property
    def target_value(self) -> Optional[Any]:
        """Return the target value."""
        return self._node.status

    @property
    def raw_unit_of_measurement(self) -> Optional[Union[str, Dict[str, Any]]]:
        """Get the raw unit of measurement for the ISY sensor device."""
        if self.target is None:
            return None
        uom = self.target.uom
        if isinstance(uom, list):
            return UOM_FRIENDLY_NAME.get(uom[0], uom[0])
        if (isy_states := UOM_TO_STATES.get(uom)):
            return isy_states
        if uom in (UOM_ON_OFF, UOM_INDEX):
            assert isinstance(uom, str)
            return uom
        return UOM_FRIENDLY_NAME.get(uom)

    @property
    def native_value(self) -> Optional[Union[str, float, int]]:
        """Get the state of the ISY sensor device."""
        if self.target is None:
            return None
        value = self.target_value
        if value == ISY_VALUE_UNKNOWN:
            return None
        uom = self.raw_unit_of_measurement
        if isinstance(uom, dict):
            return uom.get(value, value)
        if uom in (UOM_INDEX, UOM_ON_OFF):
            return cast(str, self.target.formatted)
        if uom == UOM_INDEX and hasattr(self.target, 'formatted'):
            return cast(str, self.target.formatted)
        value_converted: Optional[float] = convert_isy_value_to_hass(value, uom, self.target.prec)
        if value_converted is None:
            return None
        if uom in (UnitOfTemperature.CELSIUS, UnitOfTemperature.FAHRENHEIT):
            value_converted = self.hass.config.units.temperature(value_converted, uom)
        assert isinstance(value_converted, (int, float))
        return value_converted

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Get the Home Assistant unit of measurement for the device."""
        raw_units = self.raw_unit_of_measurement
        if isinstance(raw_units, dict) or raw_units in (UOM_ON_OFF, UOM_INDEX):
            return None
        if raw_units in (
            UnitOfTemperature.FAHRENHEIT,
            UnitOfTemperature.CELSIUS,
            UOM_DOUBLE_TEMP,
        ):
            return self.hass.config.units.temperature_unit
        return raw_units


class ISYAuxSensorEntity(ISYSensorEntity):
    """Representation of an ISY aux sensor device."""

    def __init__(
        self,
        node: Node,
        control: str,
        enabled_default: bool,
        unique_id: str,
        device_info: Optional[DeviceInfo] = None,
    ) -> None:
        """Initialize the ISY aux sensor."""
        super().__init__(node, device_info=device_info)
        self._control: str = control
        self._attr_entity_registry_enabled_default: bool = enabled_default
        self._attr_entity_category: Optional[EntityCategory] = ISY_CONTROL_TO_ENTITY_CATEGORY.get(control)
        self._attr_device_class: Optional[SensorDeviceClass] = ISY_CONTROL_TO_DEVICE_CLASS.get(control)
        self._attr_state_class: Optional[SensorStateClass] = ISY_CONTROL_TO_STATE_CLASS.get(control)
        self._attr_unique_id: str = unique_id
        self._change_handler: Optional[EventListener] = None
        self._availability_handler: Optional[EventListener] = None
        name: str = COMMAND_FRIENDLY_NAME.get(self._control, self._control)
        self._attr_name: str = f'{node.name} {name.replace("_", " ").title()}'

    @property
    def target(self) -> Optional[NodeProperty]:
        """Return target for the sensor."""
        if self._control not in self._node.aux_properties:
            return None
        return cast(NodeProperty, self._node.aux_properties[self._control])

    @property
    def target_value(self) -> Optional[Any]:
        """Return the target value."""
        return None if self.target is None else self.target.value

    async def async_added_to_hass(self) -> None:
        """Subscribe to the node control change events.

        Overloads the default ISYNodeEntity updater to only update when
        this control is changed on the device and prevent duplicate firing
        of `isy994_control` events.
        """
        self._change_handler = self._node.control_events.subscribe(
            self.async_on_update,
            event_filter={ATTR_CONTROL: self._control},
        )
        self._availability_handler = self._node.isy.nodes.status_events.subscribe(
            self.async_on_update,
            event_filter={TAG_ADDRESS: self._node.address, ATTR_ACTION: NC_NODE_ENABLED},
        )

    @callback
    def async_on_update(self, event: NodeChangedEvent) -> None:
        """Handle a control event from the ISY Node."""
        self.async_write_ha_state()

    @property
    def available(self) -> bool:
        """Return entity availability."""
        return cast(bool, self._node.enabled)
