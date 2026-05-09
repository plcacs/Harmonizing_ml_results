"""Test data purging."""

from collections.abc import Generator
from datetime import datetime, timedelta
import json
import sqlite3
from unittest.mock import patch
from freezegun import freeze_time
import pytest
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm.session import Session
from voluptuous.error import MultipleInvalid
from homeassistant.components.recorder import DOMAIN as RECORDER_DOMAIN, Recorder
from homeassistant.components.recorder.const import SupportedDialect
from homeassistant.components.recorder.db_schema import Events, EventTypes, RecorderRuns, StateAttributes, States, StatesMeta, StatisticsRuns, StatisticsShortTerm
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.components.recorder.purge import purge_old_data
from homeassistant.components.recorder.queries import select_event_type_ids
from homeassistant.components.recorder.services import SERVICE_PURGE, SERVICE_PURGE_ENTITIES
from homeassistant.components.recorder.tasks import PurgeTask
from homeassistant.components.recorder.util import session_scope
from homeassistant.const import EVENT_STATE_CHANGED, EVENT_THEMES_UPDATED, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from .common import async_recorder_block_till_done, async_wait_purge_done, async_wait_recording_done, convert_pending_events_to_event_types, convert_pending_states_to_meta
from tests.typing import RecorderInstanceContextManager

TEST_EVENT_TYPES: tuple[str, ...] = ...

@pytest.fixture
async def mock_recorder_before_hass(async_test_recorder: Recorder) -> None:
    ...

@pytest.fixture(name='use_sqlite')
def mock_use_sqlite(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    ...

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
async def test_purge_method(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, use_sqlite: bool) -> None:
    ...

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
async def test_purge_edge_case(hass: HomeAssistant, recorder_mock: Recorder, use_sqlite: bool) -> None:
    ...

@pytest.mark.parametrize('recorder_config', [{'exclude': {'entities': ['sensor.excluded']}}])
async def test_purge_filtered_states(hass: HomeAssistant, recorder_mock: Recorder, use_sqlite: bool) -> None:
    ...

@pytest.mark.parametrize('recorder_config', [{'exclude': {'entities': ['sensor.excluded']}}])
async def test_purge_filtered_states_multiple_rounds(hass: HomeAssistant, recorder_mock: Recorder, caplog: pytest.LogCaptureFixture) -> None:
    ...

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
@pytest.mark.parametrize('recorder_config', [{'exclude': {'entities': ['sensor.excluded']}}])
async def test_purge_filtered_states_to_empty(hass: HomeAssistant, recorder_mock: Recorder, use_sqlite: bool) -> None:
    ...

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
@pytest.mark.parametrize('recorder_config', [{'exclude': {'entities': ['sensor.old_format']}}])
async def test_purge_without_state_attributes_filtered_states_to_empty(hass: HomeAssistant, recorder_mock: Recorder, use_sqlite: bool) -> None:
    ...

@pytest.mark.parametrize('recorder_config', [{'exclude': {'event_types': ['EVENT_PURGE']}}])
async def test_purge_filtered_events(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

@pytest.mark.parametrize('recorder_config', [{'exclude': {'event_types': ['excluded_event'], 'entities': ['sensor.excluded', 'sensor.old_format']}}])
async def test_purge_filtered_events_state_changed(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

async def test_purge_entities(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

@pytest.mark.timeout(30)
async def test_purge_many_old_events(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

async def test_purge_old_events_purges_the_event_type_ids(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

async def test_purge_old_states_purges_the_state_metadata_ids(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...

async def test_purge_entities_keep_days(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    ...