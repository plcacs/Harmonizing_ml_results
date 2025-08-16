from __future__ import annotations
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast, List, Dict, Union

async def _async_set_firmness(sleeper: SleepIQSleeper, firmness: int) -> None:
def _get_actuator_name(bed: SleepIQBed, actuator: SleepIQActuator) -> str:
def _get_actuator_unique_id(bed: SleepIQBed, actuator: SleepIQActuator) -> str:
def _get_sleeper_name(bed: SleepIQBed, sleeper: SleepIQSleeper) -> str:
def _get_sleeper_unique_id(bed: SleepIQBed, sleeper: SleepIQSleeper) -> str:
async def _async_set_foot_warmer_time(foot_warmer: SleepIQFootWarmer, time: int) -> None:
def _get_foot_warming_name(bed: SleepIQBed, foot_warmer: SleepIQFootWarmer) -> str:
def _get_foot_warming_unique_id(bed: SleepIQBed, foot_warmer: SleepIQFootWarmer) -> str:
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
class SleepIQNumberEntity(SleepIQBedEntity[SleepIQDataUpdateCoordinator], NumberEntity):
