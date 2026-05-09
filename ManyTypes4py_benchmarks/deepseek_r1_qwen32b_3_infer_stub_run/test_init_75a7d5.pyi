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
async def hass_(recorder_mock, hass) -> HomeAssistant:
    """Set up things to be run when tests are started."""
    ...

@pytest.fixture
async def set_utc(hass: HomeAssistant) -> None:
    """Set timezone to UTC."""
    ...

async def test_service_call_create_logbook_entry(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry."""
    ...

async def test_service_call_create_logbook_entry_invalid_entity_id(hass: HomeAssistant) -> None:
    """Test if service call create log book entry with an invalid entity id."""
    ...

async def test_service_call_create_log_book_entry_no_message(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry without message."""
    ...

async def test_filter_sensor(hass_: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test numeric sensors are filtered."""
    ...

async def test_home_assistant_start_stop_not_grouped(hass_: HomeAssistant) -> None:
    """Test if HA start and stop events are no longer grouped."""
    ...

async def test_home_assistant_start(hass_: HomeAssistant) -> None:
    """Test if HA start is not filtered or converted into a restart."""
    ...

def test_process_custom_logbook_entries(hass_: HomeAssistant) -> None:
    """Test if custom log book entries get added as an entry."""
    ...

def assert_entry(entry: dict, when: datetime | None = None, name: str | None = None, message: str | None = None, domain: str | None = None, entity_id: str | None = None) -> None:
    """Assert an entry is what is expected."""
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
    """Create state changed event."""
    ...

def create_state_changed_event_from_old_new(
    entity_id: str,
    event_time_fired: datetime,
    old_state: dict,
    new_state: dict,
) -> LazyEventPartialState:
    """Create a state changed event from a old and new state."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_start_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with an invalid date time."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_end_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view."""
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_logbook_view_period_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with period and entity."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_describe_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test teaching logbook about a new event."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_described_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test exclusions of events that are described by another integration."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_end_time_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with end_time and entity."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_filter_with_automations(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_no_longer_in_state_machine(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with an entity that hass been removed from the state machine."""
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_filter_continuous_sensor_values(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test remove continuous sensor events from logbook."""
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_new_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are excluded on first update."""
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_removed_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are excluded on last update."""
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_attribute_changes(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events of attribute changes are filtered."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_context_id_automation_script_started_manually(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook populates context_ids for scripts and automations started manually."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_parent_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view links events via context parent_id."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_many_entities_multiple_calls(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with a many entities called multiple times."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_custom_log_entry_discoverable_via_(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if a custom log entry is later discoverable via ."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_multiple_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with a multiple entities."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_invalid_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with requesting an invalid entity."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_icon_and_state(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test to ensure state and custom icons are returned."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_fire_logbook_entries(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test many logbook entry calls."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if domain is excluded in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if domain or glob is excluded in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if entity is included in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if entity is excluded in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if domain is included in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if domain or glob is included in config."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_no_globs(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if include and exclude is configured."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_with_glob_filters(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test if events are filtered if include and exclude is configured."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_empty_config(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test we can handle an empty entity filter."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_context_filter(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test we can filter by context."""
    ...

async def _async_fetch_logbook(client: ClientSessionGenerator, params: dict | None = None) -> list[dict]:
    """Fetch logbook entries."""
    ...

def _assert_entry(entry: dict, when: datetime | None = None, name: str | None = None, message: str | None = None, domain: str | None = None, entity_id: str | None = None, state: str | None = None) -> None:
    """Assert an entry is what is expected."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test logbook get_events."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_future_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events with a future start time."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events bad start time."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_end_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events bad end time."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_invalid_filters(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events invalid filters."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_device_ids(hass: HomeAssistant, hass_ws_client: WebSocketGenerator, device_registry: dr.DeviceRegistry) -> None:
    """Test logbook get_events for device ids."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_select_entities_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_context_state(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test logbook get_events with a context state."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_empty_config(hass: HomeAssistant) -> None:
    """Test we handle a empty configuration."""
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_non_iterable_entity_filter(hass: HomeAssistant) -> None:
    """Test we handle a non-iterable entity filter."""
    ...