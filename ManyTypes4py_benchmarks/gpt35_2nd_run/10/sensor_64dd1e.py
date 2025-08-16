from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import logging
from typing import Any, cast, List, Dict, Union
from RFXtrx import ControlEvent, RFXtrxDevice, RFXtrxEvent, SensorEvent
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE, PERCENTAGE, SIGNAL_STRENGTH_DECIBELS_MILLIWATT, UV_INDEX, EntityCategory, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfPower, UnitOfPrecipitationDepth, UnitOfPressure, UnitOfSpeed, UnitOfTemperature, UnitOfVolumetricFlux
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from . import DeviceTuple, async_setup_platform_entry, get_rfx_object
from .const import ATTR_EVENT
from .entity import RfxtrxEntity
_LOGGER = logging.getLogger(__name__)

def _battery_convert(value: Union[int, None]) -> Union[int, None]:
    if value is None:
        return None
    return (value + 1) * 10

def _rssi_convert(value: Union[int, None]) -> Union[str, None]:
    if value is None:
        return None
    return f'{value * 8 - 120}'

@dataclass(frozen=True)
class RfxtrxSensorEntityDescription(SensorEntityDescription):
    convert: Callable[[Any], StateType] = lambda x: cast(StateType, x)

SENSOR_TYPES: List[RfxtrxSensorEntityDescription] = [
    RfxtrxSensorEntityDescription(key='Barometer', device_class=SensorDeviceClass.PRESSURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfPressure.HPA),
    RfxtrxSensorEntityDescription(key='Battery numeric', device_class=SensorDeviceClass.BATTERY, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=PERCENTAGE, convert=_battery_convert, entity_category=EntityCategory.DIAGNOSTIC),
    # Add other sensor descriptions here
]

SENSOR_TYPES_DICT: Dict[str, RfxtrxSensorEntityDescription] = {desc.key: desc for desc in SENSOR_TYPES}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    def _supported(event: RFXtrxEvent) -> bool:
        return isinstance(event, (ControlEvent, SensorEvent))

    def _constructor(event: RFXtrxEvent, auto: bool, device_id: DeviceTuple, entity_info: Any) -> List[RfxtrxSensor]:
        return [RfxtrxSensor(event.device, device_id, SENSOR_TYPES_DICT[data_type], event=event if auto else None) for data_type in set(event.values) & set(SENSOR_TYPES_DICT)]

    await async_setup_platform_entry(hass, config_entry, async_add_entities, _supported, _constructor)

class RfxtrxSensor(RfxtrxEntity, SensorEntity):
    _attr_force_update: bool = True

    def __init__(self, device: RFXtrxDevice, device_id: DeviceTuple, entity_description: RfxtrxSensorEntityDescription, event: Union[ControlEvent, SensorEvent, None] = None) -> None:
        super().__init__(device, device_id, event=event)
        self.entity_description = entity_description
        self._attr_unique_id = '_'.join((x for x in (*device_id, entity_description.key)))

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._event is None and (old_state := (await self.async_get_last_state())) is not None and (event := old_state.attributes.get(ATTR_EVENT)):
            self._apply_event(get_rfx_object(event))

    @property
    def native_value(self) -> StateType:
        if not self._event:
            return None
        value = self._event.values.get(self.entity_description.key)
        return self.entity_description.convert(value)

    @callback
    def _handle_event(self, event: RFXtrxEvent, device_id: DeviceTuple) -> None:
        if device_id != self._device_id:
            return
        if self.entity_description.key not in event.values:
            return
        _LOGGER.debug('Sensor update (Device ID: %s Class: %s Sub: %s)', event.device.id_string, event.device.__class__.__name__, event.device.subtype)
        self._apply_event(event)
        self.async_write_ha_state()
