"""The tests for the logbook component."""
import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from http import HTTPStatus
from unittest.mock import Mock
from freezegun import freeze_time
import pytest
import voluptuous as vol
from homeassistant import core as ha
from homeassistant.components import logbook, recorder
from homeassistant.components.alexa.smart_home import EVENT_ALEXA_SMART_HOME
from homeassistant.components.automation import EVENT_AUTOMATION_TRIGGERED
from homeassistant.components.logbook.models import EventAsRow, LazyEventPartialState
from homeassistant.components.logbook.processor import EventProcessor
from homeassistant.components.logbook.queries.common import PSEUDO_EVENT_STATE_CHANGED
from homeassistant.components.recorder import Recorder
from homeassistant.components.script import EVENT_SCRIPT_STARTED
from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import (
    ATTR_DOMAIN,
    ATTR_ENTITY_ID,
    ATTR_FRIENDLY_NAME,
    ATTR_NAME,
    ATTR_SERVICE,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_DOMAINS,
    CONF_ENTITIES,
    CONF_EXCLUDE,
    CONF_INCLUDE,
    EVENT_CALL_SERVICE,
    EVENT_HOMEASSISTANT_START,
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_HOMEASSISTANT_STOP,
    EVENT_LOGBOOK_ENTRY,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entityfilter import CONF_ENTITY_GLOBS
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from .common import MockRow, mock_humanify
from tests.common import MockConfigEntry, async_capture_events, mock_platform
from tests.components.recorder.common import async_recorder_block_till_done, async_wait_recording_done
from tests.typing import ClientSessionGenerator, WebSocketGenerator

@pytest.fixture
async def hass_(recorder_mock: Mock, hass: HomeAssistant) -> HomeAssistant:
    ...

@pytest.fixture
async def set_utc(hass: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
async def test_service_call_create_logbook_entry(hass_: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_service_call_create_logbook_entry_invalid_entity_id(hass: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
async def test_service_call_create_log_book_entry_no_message(hass_: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
async def test_filter_sensor(hass_: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
async def test_home_assistant_start_stop_not_grouped(hass_: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
async def test_home_assistant_start(hass_: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
def test_process_custom_logbook_entries(hass_: HomeAssistant) -> None:
    ...

def assert_entry(
    entry: dict,
    when: datetime | None = None,
    name: str | None = None,
    message: str | None = None,
    domain: str | None = None,
    entity_id: str | None = None,
) -> None:
    ...

def create_state_changed_event(
    event_time_fired: datetime,
    entity_id: str,
    state: str,
    attributes: dict | None = None,
    last_changed: datetime | None = None,
    last_reported: datetime | None = None,
    last_updated: datetime | None = None,
) -> LazyEventPartialState:
    ...

def create_state_changed_event_from_old_new(
    entity_id: str,
    event_time_fired: datetime,
    old_state: dict,
    new_state: dict,
) -> LazyEventPartialState:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_start_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_end_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_logbook_view_period_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_describe_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_described_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_end_time_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_filter_with_automations(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_no_longer_in_state_machine(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_filter_continuous_sensor_values(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_new_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_removed_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_attribute_changes(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_context_id_automation_script_started_manually(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_parent_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_context_from_template(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_many_entities_multiple_calls(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_invalid_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_icon_and_state(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_fire_logbook_entries(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_no_globs(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_with_glob_filters(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_empty_config(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_context_filter(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_future_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_end_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_invalid_filters(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_device_ids(hass: HomeAssistant, hass_ws_client: WebSocketGenerator, device_registry: dr.DeviceRegistry) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_select_entities_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_context_state(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_empty_config(hass: HomeAssistant) -> None:
    ...

@pytest.mark.asyncio
@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_non_iterable_entity_filter(hass: HomeAssistant) -> None:
    ...