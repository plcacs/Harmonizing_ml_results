"""The tests for the logbook component."""

from datetime import datetime
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock
from uuid import UUID

import pytest
from freezegun import freeze_time
from homeassistant import core as ha
from homeassistant.components import logbook, recorder
from homeassistant.components.logbook.models import EventAsRow, LazyEventPartialState
from homeassistant.components.logbook.processor import EventProcessor
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers.device_registry import DeviceRegistry
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.setup import SetupError
from homeassistant.util import dt as dt_util
from pytest import fixture
from pytest.aio import ClientSessionGenerator, WebSocketGenerator

@pytest.fixture
async def hass_(recorder_mock: Mock, hass: HomeAssistant) -> HomeAssistant:
    ...

@pytest.fixture
async def set_utc(hass: HomeAssistant) -> None:
    ...

async def test_service_call_create_logbook_entry(hass_: HomeAssistant) -> None:
    ...

async def test_service_call_create_logbook_entry_invalid_entity_id(hass: HomeAssistant) -> None:
    ...

async def test_service_call_create_log_book_entry_no_message(hass_: HomeAssistant) -> None:
    ...

async def test_filter_sensor(hass_: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

async def test_home_assistant_start_stop_not_grouped(hass_: HomeAssistant) -> None:
    ...

async def test_home_assistant_start(hass_: HomeAssistant) -> None:
    ...

def test_process_custom_logbook_entries(hass_: HomeAssistant) -> None:
    ...

def assert_entry(entry: Dict, when: Optional[datetime] = None, name: Optional[str] = None, message: Optional[str] = None, domain: Optional[str] = None, entity_id: Optional[str] = None) -> None:
    ...

def create_state_changed_event(event_time_fired: datetime, entity_id: str, state: str, attributes: Optional[Dict] = None, last_changed: Optional[datetime] = None, last_reported: Optional[datetime] = None, last_updated: Optional[datetime] = None) -> LazyEventPartialState:
    ...

def create_state_changed_event_from_old_new(entity_id: str, event_time_fired: datetime, old_state: Dict, new_state: Dict) -> LazyEventPartialState:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_start_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_invalid_end_date_time(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_logbook_view_period_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_describe_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_described_event(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_view_end_time_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_filter_with_automations(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_no_longer_in_state_machine(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_filter_continuous_sensor_values(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_new_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_removed_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock', 'set_utc')
async def test_exclude_attribute_changes(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_context_id_automation_script_started_manually(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_entity_context_parent_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_context_from_template(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_many_entities_multiple_calls(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_custom_log_entry_discoverable_via_(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_multiple_entities(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_invalid_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_icon_and_state(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_fire_logbook_entries(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_exclude_events_entity(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_events_domain_glob(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_no_globs(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_include_exclude_events_with_glob_filters(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_empty_config(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_context_filter(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

async def _async_fetch_logbook(client: ClientSessionGenerator, params: Optional[Dict] = None) -> List[Dict]:
    ...

def _assert_entry(entry: Dict, when: Optional[datetime] = None, name: Optional[str] = None, message: Optional[str] = None, domain: Optional[str] = None, entity_id: Optional[str] = None, state: Optional[str] = None) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_future_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_bad_end_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_invalid_filters(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_device_ids(hass: HomeAssistant, hass_ws_client: WebSocketGenerator, device_registry: DeviceRegistry) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_select_entities_context_id(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_get_events_with_context_state(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_empty_config(hass: HomeAssistant) -> None:
    ...

@pytest.mark.usefixtures('recorder_mock')
async def test_logbook_with_non_iterable_entity_filter(hass: HomeAssistant) -> None:
    ...