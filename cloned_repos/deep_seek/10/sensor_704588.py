"""Helper sensor for calculating utility costs."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import logging
from typing import Any, Final, Literal, cast, TypedDict, Optional, Union
from homeassistant.components.sensor import ATTR_LAST_RESET, ATTR_STATE_CLASS, SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.components.sensor.recorder import reset_detected
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant, State, callback, split_entity_id, valid_entity_id
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util, unit_conversion
from homeassistant.util.unit_system import METRIC_SYSTEM
from .const import DOMAIN
from .data import EnergyManager, async_get_manager

SUPPORTED_STATE_CLASSES = {SensorStateClass.MEASUREMENT, SensorStateClass.TOTAL, SensorStateClass.TOTAL_INCREASING}
VALID_ENERGY_UNITS = {UnitOfEnergy.GIGA_JOULE, UnitOfEnergy.KILO_WATT_HOUR, UnitOfEnergy.MEGA_JOULE, UnitOfEnergy.MEGA_WATT_HOUR, UnitOfEnergy.WATT_HOUR}
VALID_ENERGY_UNITS_GAS = {UnitOfVolume.CENTUM_CUBIC_FEET, UnitOfVolume.CUBIC_FEET, UnitOfVolume.CUBIC_METERS, *VALID_ENERGY_UNITS}
VALID_VOLUME_UNITS_WATER = {UnitOfVolume.CENTUM_CUBIC_FEET, UnitOfVolume.CUBIC_FEET, UnitOfVolume.CUBIC_METERS, UnitOfVolume.GALLONS, UnitOfVolume.LITERS}
_LOGGER = logging.getLogger(__name__)

class EnergySourceConfig(TypedDict, total=False):
    type: str
    flow_from: list[dict[str, Any]]
    flow_to: list[dict[str, Any]]
    entity_energy_price: Optional[str]
    number_energy_price: Optional[float]
    stat_energy_key: str
    total_money_key: Optional[str]

class EnergyManagerData(TypedDict, total=False):
    energy_sources: list[EnergySourceConfig]

@dataclass(slots=True)
class SourceAdapter:
    """Adapter to allow sources and their flows to be used as sensors."""
    source_type: str
    flow_type: Optional[str]
    stat_energy_key: str
    total_money_key: str
    name_suffix: str
    entity_id_suffix: str

SOURCE_ADAPTERS: tuple[SourceAdapter, ...] = (
    SourceAdapter('grid', 'flow_from', 'stat_energy_from', 'stat_cost', 'Cost', 'cost'),
    SourceAdapter('grid', 'flow_to', 'stat_energy_to', 'stat_compensation', 'Compensation', 'compensation'),
    SourceAdapter('gas', None, 'stat_energy_from', 'stat_cost', 'Cost', 'cost'),
    SourceAdapter('water', None, 'stat_energy_from', 'stat_cost', 'Cost', 'cost')
)

class SensorManager:
    """Class to handle creation/removal of sensor data."""

    def __init__(self, manager: EnergyManager, async_add_entities: AddEntitiesCallback) -> None:
        """Initialize sensor manager."""
        self.manager: EnergyManager = manager
        self.async_add_entities: AddEntitiesCallback = async_add_entities
        self.current_entities: dict[tuple[str, Optional[str], str], EnergyCostSensor] = {}

    async def async_start(self) -> None:
        """Start."""
        self.manager.async_listen_updates(self._process_manager_data)
        if self.manager.data:
            await self._process_manager_data()

    async def _process_manager_data(self) -> None:
        """Process manager data."""
        to_add: list[EnergyCostSensor] = []
        to_remove: dict[tuple[str, Optional[str], str], EnergyCostSensor] = dict(self.current_entities)

        async def finish() -> None:
            if to_add:
                self.async_add_entities(to_add)
                await asyncio.wait((ent.add_finished for ent in to_add))
            for key, entity in to_remove.items():
                self.current_entities.pop(key)
                await entity.async_remove()

        if not self.manager.data:
            await finish()
            return

        for energy_source in self.manager.data['energy_sources']:
            for adapter in SOURCE_ADAPTERS:
                if adapter.source_type != energy_source['type']:
                    continue
                if adapter.flow_type is None:
                    self._process_sensor_data(adapter, energy_source, to_add, to_remove)
                    continue
                for flow in energy_source[adapter.flow_type]:
                    self._process_sensor_data(adapter, flow, to_add, to_remove)
        await finish()

    @callback
    def _process_sensor_data(
        self,
        adapter: SourceAdapter,
        config: dict[str, Any],
        to_add: list[EnergyCostSensor],
        to_remove: dict[tuple[str, Optional[str], str], EnergyCostSensor]
    ) -> None:
        """Process sensor data."""
        if config.get(adapter.total_money_key) is not None:
            return
        key = (adapter.source_type, adapter.flow_type, config[adapter.stat_energy_key])
        if not valid_entity_id(config[adapter.stat_energy_key]) or (config.get('entity_energy_price') is None and config.get('number_energy_price') is None):
            return
        if (current_entity := to_remove.pop(key, None)):
            current_entity.update_config(config)
            return
        self.current_entities[key] = EnergyCostSensor(adapter, config)
        to_add.append(self.current_entities[key])

def _set_result_unless_done(future: asyncio.Future[None]) -> None:
    """Set the result of a future unless it is done."""
    if not future.done():
        future.set_result(None)

class EnergyCostSensor(SensorEntity):
    """Calculate costs incurred by consuming energy.

    This is intended as a fallback for when no specific cost sensor is available for the
    utility.
    """
    _attr_entity_registry_visible_default: bool = False
    _attr_should_poll: bool = False
    _wrong_state_class_reported: bool = False
    _wrong_unit_reported: bool = False

    def __init__(self, adapter: SourceAdapter, config: dict[str, Any]) -> None:
        """Initialize the sensor."""
        super().__init__()
        self._adapter: SourceAdapter = adapter
        self.entity_id: str = f'{config[adapter.stat_energy_key]}_{adapter.entity_id_suffix}'
        self._attr_device_class: SensorDeviceClass = SensorDeviceClass.MONETARY
        self._attr_state_class: SensorStateClass = SensorStateClass.TOTAL
        self._config: dict[str, Any] = config
        self._last_energy_sensor_state: Optional[State] = None
        self.add_finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def _reset(self, energy_state: State) -> None:
        """Reset the cost sensor."""
        self._attr_native_value: float = 0.0
        self._attr_last_reset = dt_util.utcnow()
        self._last_energy_sensor_state = energy_state
        self.async_write_ha_state()

    @callback
    def _update_cost(self) -> None:
        """Update incurred costs."""
        valid_units: set[str]
        default_price_unit: Optional[str]
        
        if self._adapter.source_type == 'grid':
            valid_units = VALID_ENERGY_UNITS
            default_price_unit = UnitOfEnergy.KILO_WATT_HOUR
        elif self._adapter.source_type == 'gas':
            valid_units = VALID_ENERGY_UNITS_GAS
            default_price_unit = None
        elif self._adapter.source_type == 'water':
            valid_units = VALID_VOLUME_UNITS_WATER
            if self.hass.config.units is METRIC_SYSTEM:
                default_price_unit = UnitOfVolume.CUBIC_METERS
            else:
                default_price_unit = UnitOfVolume.GALLONS
        else:
            return

        energy_state: Optional[State] = self.hass.states.get(cast(str, self._config[self._adapter.stat_energy_key]))
        if energy_state is None:
            return

        state_class: Optional[str] = energy_state.attributes.get(ATTR_STATE_CLASS)
        if state_class not in SUPPORTED_STATE_CLASSES:
            if not self._wrong_state_class_reported:
                self._wrong_state_class_reported = True
                _LOGGER.warning('Found unexpected state_class %s for %s', state_class, energy_state.entity_id)
            return

        if state_class == SensorStateClass.MEASUREMENT and ATTR_LAST_RESET not in energy_state.attributes:
            return

        try:
            energy: float = float(energy_state.state)
        except ValueError:
            return

        energy_price: float
        energy_price_unit: Optional[str]
        
        if self._config['entity_energy_price'] is not None:
            energy_price_state: Optional[State] = self.hass.states.get(self._config['entity_energy_price'])
            if energy_price_state is None:
                return
            try:
                energy_price = float(energy_price_state.state)
            except ValueError:
                if self._last_energy_sensor_state is None:
                    self._reset(energy_state)
                return
            energy_price_unit = energy_price_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT, '').partition('/')[2]
            if energy_price_unit not in valid_units:
                energy_price_unit = default_price_unit
        else:
            energy_price = cast(float, self._config['number_energy_price'])
            energy_price_unit = default_price_unit

        if self._last_energy_sensor_state is None:
            self._reset(energy_state)
            return

        energy_unit: Optional[str] = energy_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if energy_unit is None or energy_unit not in valid_units:
            if not self._wrong_unit_reported:
                self._wrong_unit_reported = True
                _LOGGER.warning('Found unexpected unit %s for %s', energy_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT), energy_state.entity_id)
            return

        if (state_class != SensorStateClass.TOTAL_INCREASING and 
            energy_state.attributes.get(ATTR_LAST_RESET) != self._last_energy_sensor_state.attributes.get(ATTR_LAST_RESET)) or (
                state_class == SensorStateClass.TOTAL_INCREASING and 
                reset_detected(self.hass, cast(str, self._config[self._adapter.stat_energy_key]), energy, float(self._last_energy_sensor_state.state), self._last_energy_sensor_state)):
            energy_state_copy: State = copy.copy(energy_state)
            energy_state_copy.state = '0.0'
            self._reset(energy_state_copy)

        old_energy_value: float = float(self._last_energy_sensor_state.state)
        cur_value: float = cast(float, self._attr_native_value)
        converted_energy_price: float

        if energy_price_unit is None:
            converted_energy_price = energy_price
        else:
            converter: Callable[[float, str, str], float]
            if energy_unit in VALID_ENERGY_UNITS:
                converter = unit_conversion.EnergyConverter.convert
            else:
                converter = unit_conversion.VolumeConverter.convert
            converted_energy_price = converter(energy_price, energy_unit, energy_price_unit)

        self._attr_native_value = cur_value + (energy - old_energy_value) * converted_energy_price
        self._last_energy_sensor_state = energy_state

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        energy_state: Optional[State] = self.hass.states.get(self._config[self._adapter.stat_energy_key])
        name: str
        if energy_state:
            name = energy_state.name
        else:
            name = split_entity_id(self._config[self._adapter.stat_energy_key])[0].replace('_', ' ')
        self._attr_name = f'{name} {self._adapter.name_suffix}'
        self._update_cost()
        self.hass.data[DOMAIN]['cost_sensors'][self._config[self._adapter.stat_energy_key]] = self.entity_id
        self.async_on_remove(async_track_state_change_event(self.hass, cast(str, self._config[self._adapter.stat_energy_key]), self._async_state_changed_listener))
        _set_result_unless_done(self.add_finished)

    @callback
    def _async_state_changed_listener(self, *_: Any) -> None:
        """Handle child updates."""
        self._update_cost()
        self.async_write_ha_state()

    @callback
    def add_to_platform_abort(self) -> None:
        """Abort adding an entity to a platform."""
        _set_result_unless_done(self.add_finished)
        super().add_to_platform_abort()

    async def async_will_remove_from_hass(self) -> None:
        """Handle removing from hass."""
        self.hass.data[DOMAIN]['cost_sensors'].pop(self._config[self._adapter.stat_energy_key])
        await super().async_will_remove_from_hass()

    @callback
    def update_config(self, config: dict[str, Any]) -> None:
        """Update the config."""
        self._config = config

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the units of measurement."""
        return self.hass.config.currency

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the sensor."""
        entity_registry = er.async_get(self.hass)
        if (registry_entry := entity_registry.async_get(self._config[self._adapter.stat_energy_key])):
            prefix = registry_entry.id
        else:
            prefix = self._config[self._adapter.stat_energy_key]
        return f'{prefix}_{self._adapter.source_type}_{self._adapter.entity_id_suffix}'

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the energy sensors."""
    sensor_manager = SensorManager(await async_get_manager(hass), async_add_entities)
    await sensor_manager.async_start()
