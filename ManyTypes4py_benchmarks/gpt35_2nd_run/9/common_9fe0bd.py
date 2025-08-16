from __future__ import annotations
import asyncio
from typing import Any, Literal
from unittest.mock import MagicMock, patch

@dataclass
class BlockRecorderTask(RecorderTask):
    event: asyncio.Event
    seconds: float

@dataclass
class ForceReturnConnectionToPool(RecorderTask):
    pass

async def async_block_recorder(hass, seconds: float) -> None:
    pass

def get_start_time(start: datetime) -> datetime:
    pass

def do_adhoc_statistics(hass, **kwargs: Any) -> None:
    pass

def wait_recording_done(hass) -> None:
    pass

def trigger_db_commit(hass) -> None:
    pass

async def async_wait_recording_done(hass) -> None:
    pass

async def async_wait_purge_done(hass, max_number: int = None) -> None:
    pass

@ha.callback
def async_trigger_db_commit(hass) -> None:
    pass

async def async_recorder_block_till_done(hass) -> None:
    pass

def corrupt_db_file(test_db_file: str) -> None:
    pass

def create_engine_test(*args: Any, **kwargs: Any) -> Any:
    pass

def run_information_with_session(session: Session, point_in_time: datetime = None) -> RecorderRuns:
    pass

def statistics_during_period(hass, start_time: datetime, end_time: datetime = None, statistic_ids: set = None, period: str = 'hour', units: Any = None, types: set = None) -> Any:
    pass

def assert_states_equal_without_context(state: State, other: State) -> None:
    pass

def assert_states_equal_without_context_and_last_changed(state: State, other: State) -> None:
    pass

def assert_multiple_states_equal_without_context_and_last_changed(states: Iterable[State], others: Iterable[State]) -> None:
    pass

def assert_multiple_states_equal_without_context(states: Iterable[State], others: Iterable[State]) -> None:
    pass

def assert_events_equal_without_context(event: Event, other: Event) -> None:
    pass

def assert_dict_of_states_equal_without_context(states: dict[str, State], others: dict[str, State]) -> None:
    pass

def assert_dict_of_states_equal_without_context_and_last_changed(states: dict[str, State], others: dict[str, State]) -> None:
    pass

async def async_record_states(hass) -> Any:
    pass

def record_states(hass) -> Any:
    pass

def convert_pending_states_to_meta(instance, session) -> None:
    pass

def convert_pending_events_to_event_types(instance, session) -> None:
    pass

def create_engine_test_for_schema_version_postfix(*args: Any, schema_version_postfix: Any, **kwargs: Any) -> Any:
    pass

def get_schema_module_path(schema_version_postfix: Any) -> str:
    pass

@contextmanager
def old_db_schema(schema_version_postfix: Any) -> None:
    pass

async def async_attach_db_engine(hass) -> None:
    pass
