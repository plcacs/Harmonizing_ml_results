from __future__ import annotations
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast, List, Union

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

async def _async_set_firmness(sleeper: SleepIQSleeper, firmness: int) -> None:
    await sleeper.set_sleepnumber(firmness)

async def _async_set_actuator_position(actuator: SleepIQActuator, position: int) -> None:
    await actuator.set_position(position)

def _get_actuator_name(bed: SleepIQBed, actuator: SleepIQActuator) -> str:
    ...

def _get_actuator_unique_id(bed: SleepIQBed, actuator: SleepIQActuator) -> str:
    ...

def _get_sleeper_name(bed: SleepIQBed, sleeper: SleepIQSleeper) -> str:
    ...

def _get_sleeper_unique_id(bed: SleepIQBed, sleeper: SleepIQSleeper) -> str:
    ...

async def _async_set_foot_warmer_time(foot_warmer: SleepIQFootWarmer, time: int) -> None:
    ...

def _get_foot_warming_name(bed: SleepIQBed, foot_warmer: SleepIQFootWarmer) -> str:
    ...

def _get_foot_warming_unique_id(bed: SleepIQBed, foot_warmer: SleepIQFootWarmer) -> str:
    ...

NUMBER_DESCRIPTIONS: dict[str, SleepIQNumberEntityDescription] = {
    FIRMNESS: SleepIQNumberEntityDescription(key=FIRMNESS, native_min_value=5, native_max_value=100, native_step=5, name=ENTITY_TYPES[FIRMNESS], icon=ICON_OCCUPIED, value_fn=lambda sleeper: cast(float, sleeper.sleep_number), set_value_fn=_async_set_firmness, get_name_fn=_get_sleeper_name, get_unique_id_fn=_get_sleeper_unique_id),
    ACTUATOR: SleepIQNumberEntityDescription(key=ACTUATOR, native_min_value=0, native_max_value=100, native_step=1, name=ENTITY_TYPES[ACTUATOR], icon=ICON_OCCUPIED, value_fn=lambda actuator: cast(float, actuator.position), set_value_fn=_async_set_actuator_position, get_name_fn=_get_actuator_name, get_unique_id_fn=_get_actuator_unique_id),
    FOOT_WARMING_TIMER: SleepIQNumberEntityDescription(key=FOOT_WARMING_TIMER, native_min_value=30, native_max_value=360, native_step=30, name=ENTITY_TYPES[FOOT_WARMING_TIMER], icon='mdi:timer', value_fn=lambda foot_warmer: foot_warmer.timer, set_value_fn=_async_set_foot_warmer_time, get_name_fn=_get_foot_warming_name, get_unique_id_fn=_get_foot_warming_unique_id)
}

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class SleepIQNumberEntity(SleepIQBedEntity[SleepIQDataUpdateCoordinator], NumberEntity):
    """Representation of a SleepIQ number entity."""
    _attr_icon: str = 'mdi:bed'

    def __init__(self, coordinator: SleepIQDataUpdateCoordinator, bed: SleepIQBed, device: Union[SleepIQSleeper, SleepIQActuator, SleepIQFootWarmer], description: SleepIQNumberEntityDescription) -> None:
        ...

    @callback
    def _async_update_attrs(self) -> None:
        ...

    async def async_set_native_value(self, value: float) -> None:
        ...
