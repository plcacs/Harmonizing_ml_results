"""Support for SleepIQ SleepNumber firmness number entities."""
from __future__ import annotations
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast
from asyncsleepiq import FootWarmingTemps, SleepIQActuator, SleepIQBed, SleepIQFootWarmer, SleepIQSleeper
from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import ACTUATOR, DOMAIN, ENTITY_TYPES, FIRMNESS, FOOT_WARMING_TIMER, ICON_OCCUPIED
from .coordinator import SleepIQData, SleepIQDataUpdateCoordinator
from .entity import SleepIQBedEntity, sleeper_for_side

@dataclass(frozen=True, kw_only=True)
class SleepIQNumberEntityDescription(NumberEntityDescription):
    """Class to describe a SleepIQ number entity."""

async def _async_set_firmness(sleeper, firmness):
    await sleeper.set_sleepnumber(firmness)

async def _async_set_actuator_position(actuator, position):
    await actuator.set_position(position)

def _get_actuator_name(bed, actuator):
    if actuator.side:
        return f'SleepNumber {bed.name} {actuator.side_full} {actuator.actuator_full} {ENTITY_TYPES[ACTUATOR]}'
    return f'SleepNumber {bed.name} {actuator.actuator_full} {ENTITY_TYPES[ACTUATOR]}'

def _get_actuator_unique_id(bed, actuator):
    if actuator.side:
        return f'{bed.id}_{actuator.side.value}_{actuator.actuator}'
    return f'{bed.id}_{actuator.actuator}'

def _get_sleeper_name(bed, sleeper):
    return f'SleepNumber {bed.name} {sleeper.name} {ENTITY_TYPES[FIRMNESS]}'

def _get_sleeper_unique_id(bed, sleeper):
    return f'{sleeper.sleeper_id}_{FIRMNESS}'

async def _async_set_foot_warmer_time(foot_warmer, time):
    temperature = FootWarmingTemps(foot_warmer.temperature)
    if temperature != FootWarmingTemps.OFF:
        await foot_warmer.turn_on(temperature, time)
    foot_warmer.timer = time

def _get_foot_warming_name(bed, foot_warmer):
    sleeper = sleeper_for_side(bed, foot_warmer.side)
    return f'SleepNumber {bed.name} {sleeper.name} {ENTITY_TYPES[FOOT_WARMING_TIMER]}'

def _get_foot_warming_unique_id(bed, foot_warmer):
    return f'{bed.id}_{foot_warmer.side.value}_{FOOT_WARMING_TIMER}'
NUMBER_DESCRIPTIONS = {FIRMNESS: SleepIQNumberEntityDescription(key=FIRMNESS, native_min_value=5, native_max_value=100, native_step=5, name=ENTITY_TYPES[FIRMNESS], icon=ICON_OCCUPIED, value_fn=lambda sleeper: cast(float, sleeper.sleep_number), set_value_fn=_async_set_firmness, get_name_fn=_get_sleeper_name, get_unique_id_fn=_get_sleeper_unique_id), ACTUATOR: SleepIQNumberEntityDescription(key=ACTUATOR, native_min_value=0, native_max_value=100, native_step=1, name=ENTITY_TYPES[ACTUATOR], icon=ICON_OCCUPIED, value_fn=lambda actuator: cast(float, actuator.position), set_value_fn=_async_set_actuator_position, get_name_fn=_get_actuator_name, get_unique_id_fn=_get_actuator_unique_id), FOOT_WARMING_TIMER: SleepIQNumberEntityDescription(key=FOOT_WARMING_TIMER, native_min_value=30, native_max_value=360, native_step=30, name=ENTITY_TYPES[FOOT_WARMING_TIMER], icon='mdi:timer', value_fn=lambda foot_warmer: foot_warmer.timer, set_value_fn=_async_set_foot_warmer_time, get_name_fn=_get_foot_warming_name, get_unique_id_fn=_get_foot_warming_unique_id)}

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up the SleepIQ bed sensors."""
    data = hass.data[DOMAIN][entry.entry_id]
    entities = []
    for bed in data.client.beds.values():
        entities.extend((SleepIQNumberEntity(data.data_coordinator, bed, sleeper, NUMBER_DESCRIPTIONS[FIRMNESS]) for sleeper in bed.sleepers))
        entities.extend((SleepIQNumberEntity(data.data_coordinator, bed, actuator, NUMBER_DESCRIPTIONS[ACTUATOR]) for actuator in bed.foundation.actuators))
        entities.extend((SleepIQNumberEntity(data.data_coordinator, bed, foot_warmer, NUMBER_DESCRIPTIONS[FOOT_WARMING_TIMER]) for foot_warmer in bed.foundation.foot_warmers))
    async_add_entities(entities)

class SleepIQNumberEntity(SleepIQBedEntity[SleepIQDataUpdateCoordinator], NumberEntity):
    """Representation of a SleepIQ number entity."""
    _attr_icon = 'mdi:bed'

    def __init__(self, coordinator, bed, device, description):
        """Initialize the number."""
        self.entity_description = description
        self.device = device
        self._attr_name = description.get_name_fn(bed, device)
        self._attr_unique_id = description.get_unique_id_fn(bed, device)
        if description.icon:
            self._attr_icon = description.icon
        super().__init__(coordinator, bed)

    @callback
    def _async_update_attrs(self):
        """Update number attributes."""
        self._attr_native_value = float(self.entity_description.value_fn(self.device))

    async def async_set_native_value(self, value):
        """Set the number value."""
        await self.entity_description.set_value_fn(self.device, int(value))
        self._attr_native_value = value
        self.async_write_ha_state()