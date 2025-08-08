from __future__ import annotations
import asyncio
from typing import Any, Literal
from homeassistant import core as ha
from homeassistant.components import recorder
from homeassistant.components.recorder import Recorder, core, get_instance, migration, statistics
from homeassistant.components.recorder.db_schema import Events, EventTypes, RecorderRuns, States, StatesMeta
from homeassistant.components.recorder.tasks import RecorderTask, StatisticsTask
from homeassistant.const import UnitOfTemperature
from homeassistant.core import Event, HomeAssistant, State
from homeassistant.util import dt as dt_util

async def async_block_recorder(hass: HomeAssistant, seconds: int) -> None:
    ...

def get_start_time(start: datetime) -> datetime:
    ...

def do_adhoc_statistics(hass: HomeAssistant, **kwargs: Any) -> None:
    ...

def wait_recording_done(hass: HomeAssistant) -> None:
    ...

def trigger_db_commit(hass: HomeAssistant) -> None:
    ...

async def async_wait_recording_done(hass: HomeAssistant) -> None:
    ...

async def async_wait_purge_done(hass: HomeAssistant, max_number: int = None) -> None:
    ...

@ha.callback
def async_trigger_db_commit(hass: HomeAssistant) -> None:
    ...

async def async_recorder_block_till_done(hass: HomeAssistant) -> None:
    ...

def corrupt_db_file(test_db_file: str) -> None:
    ...

def create_engine_test(*args: Any, **kwargs: Any) -> Any:
    ...

def run_information_with_session(session: Session, point_in_time: datetime = None) -> RecorderRuns:
    ...

def statistics_during_period(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, statistic_ids: set = None, period: str = 'hour', units: Any = None, types: set = None) -> Any:
    ...

def assert_states_equal_without_context(state: State, other: State) -> None:
    ...

def assert_states_equal_without_context_and_last_changed(state: State, other: State) -> None:
    ...

def assert_multiple_states_equal_without_context_and_last_changed(states: Iterable[State], others: Iterable[State]) -> None:
    ...

def assert_multiple_states_equal_without_context(states: Iterable[State], others: Iterable[State]) -> None:
    ...

def assert_events_equal_without_context(event: Event, other: Event) -> None:
    ...

def assert_dict_of_states_equal_without_context(states: dict[str, State], others: dict[str, State]) -> None:
    ...

def assert_dict_of_states_equal_without_context_and_last_changed(states: dict[str, State], others: dict[str, State]) -> None:
    ...

async def async_record_states(hass: HomeAssistant) -> Any:
    ...

def record_states(hass: HomeAssistant) -> Any:
    ...

def convert_pending_states_to_meta(instance: Recorder, session: Session) -> None:
    ...

def convert_pending_events_to_event_types(instance: Recorder, session: Session) -> None:
    ...

def create_engine_test_for_schema_version_postfix(*args: Any, schema_version_postfix: Any, **kwargs: Any) -> Any:
    ...

def get_schema_module_path(schema_version_postfix: Any) -> str:
    ...

@contextmanager
def old_db_schema(schema_version_postfix: Any) -> None:
    ...

async def async_attach_db_engine(hass: HomeAssistant) -> None:
    ...
