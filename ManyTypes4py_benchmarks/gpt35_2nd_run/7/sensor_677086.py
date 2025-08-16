from typing import Any, Dict, List
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONCENTRATION_PARTS_PER_MILLION, PERCENTAGE, UnitOfElectricPotential, UnitOfMass, UnitOfTemperature, UnitOfVolume
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .common import check_guard
from .const import COORDINATOR, DEFAULT_PH_OFFSET, DOMAIN, PUMP_TYPES
from .coordinator import OmniLogicUpdateCoordinator
from .entity import OmniLogicEntity

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    coordinator: OmniLogicUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][COORDINATOR]
    entities: List[OmnilogicSensor] = []
    for item_id, item in coordinator.data.items():
        id_len: int = len(item_id)
        item_kind: str = item_id[-2]
        entity_settings: List[Dict[str, Any]] = SENSOR_TYPES.get((id_len, item_kind), [])
        if not entity_settings:
            continue
        for entity_setting in entity_settings:
            entity_classes: Dict[str, Any] = entity_setting['entity_classes']
            for state_key, entity_class in entity_classes.items():
                if check_guard(state_key, item, entity_setting):
                    continue
                entity: OmnilogicSensor = entity_class(coordinator=coordinator, state_key=state_key, name=entity_setting['name'], kind=entity_setting['kind'], item_id=item_id, device_class=entity_setting['device_class'], icon=entity_setting['icon'], unit=entity_setting['unit'])
                entities.append(entity)
    async_add_entities(entities)

class OmnilogicSensor(OmniLogicEntity, SensorEntity):
    """Defines an Omnilogic sensor entity."""

    def __init__(self, coordinator: OmniLogicUpdateCoordinator, kind: str, name: str, device_class: SensorDeviceClass, icon: str, unit: str, item_id: str, state_key: str) -> None:
        super().__init__(coordinator=coordinator, kind=kind, name=name, item_id=item_id, icon=icon)
        backyard_id: str = item_id[:2]
        unit_type: str = coordinator.data[backyard_id].get('Unit-of-Measurement')
        self._unit_type: str = unit_type
        self._attr_device_class: SensorDeviceClass = device_class
        self._attr_native_unit_of_measurement: str = unit
        self._state_key: str = state_key

class OmniLogicTemperatureSensor(OmnilogicSensor):
    """Define an OmniLogic Temperature (Air/Water) Sensor."""

    @property
    def native_value(self) -> Any:
        sensor_data: Any = self.coordinator.data[self._item_id][self._state_key]
        hayward_state: Any = sensor_data
        hayward_unit_of_measure: UnitOfTemperature = UnitOfTemperature.FAHRENHEIT
        state: Any = sensor_data
        if self._unit_type == 'Metric':
            hayward_state = round((int(hayward_state) - 32) * 5 / 9, 1)
            hayward_unit_of_measure = UnitOfTemperature.CELSIUS
        if int(sensor_data) == -1:
            hayward_state = None
            state = None
        self._attrs['hayward_temperature'] = hayward_state
        self._attrs['hayward_unit_of_measure'] = hayward_unit_of_measure
        self._attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
        return state

class OmniLogicPumpSpeedSensor(OmnilogicSensor):
    """Define an OmniLogic Pump Speed Sensor."""

    @property
    def native_value(self) -> Any:
        pump_type: str = PUMP_TYPES[self.coordinator.data[self._item_id].get('Filter-Type', self.coordinator.data[self._item_id].get('Type', {}))]
        pump_speed: Any = self.coordinator.data[self._item_id][self._state_key]
        if pump_type == 'VARIABLE':
            self._attr_native_unit_of_measurement = PERCENTAGE
            state: Any = pump_speed
        elif pump_type == 'DUAL':
            self._attr_native_unit_of_measurement = None
            if pump_speed == 0:
                state = 'off'
            elif pump_speed == self.coordinator.data[self._item_id].get('Min-Pump-Speed'):
                state = 'low'
            elif pump_speed == self.coordinator.data[self._item_id].get('Max-Pump-Speed'):
                state = 'high'
        self._attrs['pump_type'] = pump_type
        return state

class OmniLogicSaltLevelSensor(OmnilogicSensor):
    """Define an OmniLogic Salt Level Sensor."""

    @property
    def native_value(self) -> Any:
        salt_return: Any = self.coordinator.data[self._item_id][self._state_key]
        if self._unit_type == 'Metric':
            salt_return = round(int(salt_return) / 1000, 2)
            self._attr_native_unit_of_measurement = f'{UnitOfMass.GRAMS}/{UnitOfVolume.LITERS}'
        return salt_return

class OmniLogicChlorinatorSensor(OmnilogicSensor):
    """Define an OmniLogic Chlorinator Sensor."""

    @property
    def native_value(self) -> Any:
        return self.coordinator.data[self._item_id][self._state_key]

class OmniLogicPHSensor(OmnilogicSensor):
    """Define an OmniLogic pH Sensor."""

    @property
    def native_value(self) -> Any:
        ph_state: Any = self.coordinator.data[self._item_id][self._state_key]
        if ph_state == 0:
            ph_state = None
        else:
            ph_state = float(ph_state) + float(self.coordinator.config_entry.options.get('ph_offset', DEFAULT_PH_OFFSET))
        return ph_state

class OmniLogicORPSensor(OmnilogicSensor):
    """Define an OmniLogic ORP Sensor."""

    def __init__(self, coordinator: OmniLogicUpdateCoordinator, state_key: str, name: str, kind: str, item_id: str, device_class: SensorDeviceClass, icon: str, unit: str) -> None:
        super().__init__(coordinator=coordinator, kind=kind, name=name, device_class=device_class, icon=icon, unit=unit, item_id=item_id, state_key=state_key)

    @property
    def native_value(self) -> Any:
        orp_state: Any = int(self.coordinator.data[self._item_id][self._state_key])
        if orp_state == -1:
            orp_state = None
        return orp_state
