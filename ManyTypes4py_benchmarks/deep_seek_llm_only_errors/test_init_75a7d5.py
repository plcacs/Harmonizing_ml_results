"""The tests for the logbook component."""
import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Any, Dict, List, Optional, cast
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
from homeassistant.core import Event, HomeAssistant, State
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entityfilter import CONF_ENTITY_GLOBS
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from .common import MockRow, mock_humanify
from tests.common import (
    MockConfigEntry,
    async_capture_events,
    mock_platform,
)
from tests.components.recorder.common import (
    async_recorder_block_till_done,
    async_wait_recording_done,
)
from tests.typing import ClientSessionGenerator, WebSocketGenerator

EMPTY_CONFIG: Dict[str, Any] = logbook.CONFIG_SCHEMA({logbook.DOMAIN: {}})


@pytest.fixture
async def hass_(recorder_mock: Recorder, hass: HomeAssistant) -> HomeAssistant:
    """Set up things to be run when tests are started."""
    assert await async_setup_component(hass, logbook.DOMAIN, EMPTY_CONFIG)
    return hass


@pytest.fixture
async def set_utc(hass: HomeAssistant) -> None:
    """Set timezone to UTC."""
    await hass.config.async_set_time_zone("UTC")


async def test_service_call_create_logbook_entry(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry."""
    calls = async_capture_events(hass_, logbook.EVENT_LOGBOOK_ENTRY)
    await hass_.services.async_call(
        logbook.DOMAIN,
        "log",
        {
            logbook.ATTR_NAME: "Alarm",
            logbook.ATTR_MESSAGE: "is triggered",
            logbook.ATTR_DOMAIN: "switch",
            logbook.ATTR_ENTITY_ID: "switch.test_switch",
        },
        True,
    )
    await hass_.services.async_call(
        logbook.DOMAIN,
        "log",
        {
            logbook.ATTR_NAME: "This entry",
            logbook.ATTR_MESSAGE: "has no domain or entity_id",
        },
        True,
    )
    await async_wait_recording_done(hass_)
    event_processor = EventProcessor(hass_, (EVENT_LOGBOOK_ENTRY,))
    events = list(
        event_processor.get_events(
            dt_util.utcnow() - timedelta(hours=1),
            dt_util.utcnow() + timedelta(hours=1),
        )
    )
    assert len(events) == 2
    assert len(calls) == 2
    first_call = calls[-2]
    assert first_call.data.get(logbook.ATTR_NAME) == "Alarm"
    assert first_call.data.get(logbook.ATTR_MESSAGE) == "is triggered"
    assert first_call.data.get(logbook.ATTR_DOMAIN) == "switch"
    assert first_call.data.get(logbook.ATTR_ENTITY_ID) == "switch.test_switch"
    last_call = calls[-1]
    assert last_call.data.get(logbook.ATTR_NAME) == "This entry"
    assert last_call.data.get(logbook.ATTR_MESSAGE) == "has no domain or entity_id"
    assert last_call.data.get(logbook.ATTR_DOMAIN) == "logbook"


@pytest.mark.usefixtures("recorder_mock")
async def test_service_call_create_logbook_entry_invalid_entity_id(hass: HomeAssistant) -> None:
    """Test if service call create log book entry with an invalid entity id."""
    await async_setup_component(hass, "logbook", {})
    await hass.async_block_till_done()
    hass.bus.async_fire(
        logbook.EVENT_LOGBOOK_ENTRY,
        {
            logbook.ATTR_NAME: "Alarm",
            logbook.ATTR_MESSAGE: "is triggered",
            logbook.ATTR_DOMAIN: "switch",
            logbook.ATTR_ENTITY_ID: 1234,
        },
    )
    await async_wait_recording_done(hass)
    event_processor = EventProcessor(hass, (EVENT_LOGBOOK_ENTRY,))
    events = list(
        event_processor.get_events(
            dt_util.utcnow() - timedelta(hours=1),
            dt_util.utcnow() + timedelta(hours=1),
        )
    )
    assert len(events) == 1
    assert events[0][logbook.ATTR_DOMAIN] == "switch"
    assert events[0][logbook.ATTR_NAME] == "Alarm"
    assert events[0][logbook.ATTR_ENTITY_ID] == 1234
    assert events[0][logbook.ATTR_MESSAGE] == "is triggered"


async def test_service_call_create_log_book_entry_no_message(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry without message."""
    calls = async_capture_events(hass_, logbook.EVENT_LOGBOOK_ENTRY)
    with pytest.raises(vol.Invalid):
        await hass_.services.async_call(logbook.DOMAIN, "log", {}, True)
    await hass_.async_block_till_done()
    assert len(calls) == 0


async def test_filter_sensor(hass_: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    """Test numeric sensors are filtered."""
    registry = er.async_get(hass_)
    entity_id1 = "sensor.bla"
    attributes_1 = None
    entity_id2 = "sensor.blu"
    attributes_2 = {ATTR_UNIT_OF_MEASUREMENT: "cats"}
    entity_id3 = registry.async_get_or_create(
        "sensor",
        "test",
        "unique_3",
        suggested_object_id="bli",
        capabilities={"state_class": SensorStateClass.MEASUREMENT},
    ).entity_id
    attributes_3 = None
    entity_id4 = registry.async_get_or_create(
        "sensor", "test", "unique_4", suggested_object_id="ble"
    ).entity_id
    attributes_4 = None
    hass_.states.async_set(entity_id1, None, attributes_1)
    hass_.states.async_set(entity_id1, 10, attributes_1)
    hass_.states.async_set(entity_id2, None, attributes_2)
    hass_.states.async_set(entity_id2, 10, attributes_2)
    hass_.states.async_set(entity_id3, None, attributes_3)
    hass_.states.async_set(entity_id3, 10, attributes_3)
    hass_.states.async_set(entity_id1, 20, attributes_1)
    hass_.states.async_set(entity_id2, 20, attributes_2)
    hass_.states.async_set(entity_id4, None, attributes_4)
    hass_.states.async_set(entity_id4, 10, attributes_4)
    await async_wait_recording_done(hass_)
    client = await hass_client()
    entries = await _async_fetch_logbook(client)
    assert len(entries) == 3
    _assert_entry(entries[0], name="bla", entity_id=entity_id1, state="10")
    _assert_entry(entries[1], name="bla", entity_id=entity_id1, state="20")
    _assert_entry(entries[2], name="ble", entity_id=entity_id4, state="10")


async def test_home_assistant_start_stop_not_grouped(hass_: HomeAssistant) -> None:
    """Test if HA start and stop events are no longer grouped."""
    await async_setup_component(hass_, "homeassistant", {})
    await hass_.async_block_till_done()
    entries = mock_humanify(
        hass_,
        (
            MockRow(EVENT_HOMEASSISTANT_STOP),
            MockRow(EVENT_HOMEASSISTANT_START),
        ),
    )
    assert len(entries) == 2
    assert_entry(
        entries[0], name="Home Assistant", message="stopped", domain=ha.DOMAIN
    )
    assert_entry(
        entries[1], name="Home Assistant", message="started", domain=ha.DOMAIN
    )


async def test_home_assistant_start(hass_: HomeAssistant) -> None:
    """Test if HA start is not filtered or converted into a restart."""
    await async_setup_component(hass_, "homeassistant", {})
    await hass_.async_block_till_done()
    entity_id = "switch.bla"
    pointA = dt_util.utcnow()
    entries = mock_humanify(
        hass_,
        (
            MockRow(EVENT_HOMEASSISTANT_START),
            create_state_changed_event(pointA, entity_id, 10).row,
        ),
    )
    assert len(entries) == 2
    assert_entry(
        entries[0], name="Home Assistant", message="started", domain=ha.DOMAIN
    )
    assert_entry(entries[1], pointA, "bla", entity_id=entity_id)


def test_process_custom_logbook_entries(hass_: HomeAssistant) -> None:
    """Test if custom log book entries get added as an entry."""
    name = "Nice name"
    message = "has a custom entry"
    entity_id = "sun.sun"
    entries = mock_humanify(
        hass_,
        (
            MockRow(
                logbook.EVENT_LOGBOOK_ENTRY,
                {
                    logbook.ATTR_NAME: name,
                    logbook.ATTR_MESSAGE: message,
                    logbook.ATTR_ENTITY_ID: entity_id,
                },
            ),
        ),
    )
    assert len(entries) == 1
    assert_entry(entries[0], name=name, message=message, entity_id=entity_id)


def assert_entry(
    entry: Dict[str, Any],
    when: Optional[datetime] = None,
    name: Optional[str] = None,
    message: Optional[str] = None,
    domain: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> None:
    """Assert an entry is what is expected."""
    return _assert_entry(entry, when, name, message, domain, entity_id)


def create_state_changed_event(
    event_time_fired: datetime,
    entity_id: str,
    state: Any,
    attributes: Optional[Dict[str, Any]] = None,
    last_changed: Optional[datetime] = None,
    last_reported: Optional[datetime] = None,
    last_updated: Optional[datetime] = None,
) -> LazyEventPartialState:
    """Create state changed event."""
    old_state = ha.State(
        entity_id,
        "old",
        attributes,
        last_changed=last_changed,
        last_reported=last_reported,
        last_updated=last_updated,
    ).as_dict()
    new_state = ha.State(
        entity_id,
        state,
        attributes,
        last_changed=last_changed,
        last_reported=last_reported,
        last_updated=last_updated,
    ).as_dict()
    return create_state_changed_event_from_old_new(entity_id, event_time_fired, old_state, new_state)


def create_state_changed_event_from_old_new(
    entity_id: str,
    event_time_fired: datetime,
    old_state: Dict[str, Any],
    new_state: Dict[str, Any],
) -> LazyEventPartialState:
    """Create a state changed event from a old and new state."""
    row = EventAsRow(
        row_id=1,
        event_type=PSEUDO_EVENT_STATE_CHANGED,
        event_data="{}",
        time_fired_ts=event_time_fired.timestamp(),
        context_id_bin=None,
        context_user_id_bin=None,
        context_parent_id_bin=None,
        state=new_state and new_state.get("state"),
        entity_id=entity_id,
        icon=None,
        context_only=False,
        data=None,
        context=None,
    )
    return LazyEventPartialState(row, {})


@pytest.mark.usefixtures("recorder_mock")
async def test_logbook_view(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Test the logbook view."""
    await async_setup_component(hass, "logbook", {})
    await async_recorder_block_till_done(hass)
    client = await hass_client()
    response = await client.get(f"/api/logbook/{dt_util.utcnow().isoformat()}")
    assert response.status == HTTPStatus.OK


@pytest.mark.usefixtures("recorder_mock")
async def test_logbook_view_invalid_start_date_time(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Test the logbook view with an invalid date time."""
    await async_setup_component(hass, "logbook", {})
    await async_recorder_block_till_done(hass)
    client = await hass_client()
    response = await client.get("/api/logbook/INVALID")
    assert response.status == HTTPStatus.BAD_REQUEST


@pytest.mark.usefixtures("recorder_mock")
async def test_logbook_view_invalid_end_date_time(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Test the logbook view."""
    await async_setup_component(hass, "logbook", {})
    await async_recorder_block_till_done(hass)
    client = await hass_client()
    response = await client.get(
        f"/api/logbook/{dt_util.utcnow().isoformat()}?end_time=INVALID"
    )
    assert response.status == HTTPStatus.BAD_REQUEST


@pytest.mark.usefixtures("recorder_mock", "set_utc")
async def test_logbook_view_period_entity(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Test the logbook view with period and entity."""
    await async_setup_component(hass, "logbook", {})
    await async_recorder_block_till_done(hass)
    entity_id_test = "switch.test"
    hass.states.async_set(entity_id_test, STATE_OFF)
    hass.states.async_set(entity_id_test, STATE_ON)
    entity_id_second = "switch.second"
    hass.states.async_set(entity_id_second, STATE_OFF)
    hass.states.async_set(entity_id_second, STATE_ON)
    await async_wait_recording_done(hass)
    client = await hass_client()
    start = dt_util.utcnow().date()
    start_date = datetime(start.year, start.month, start.day, tzinfo=dt_util.UTC)
    response = await client.get(f"/api/logbook/{start_date.isoformat()}")
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    assert len(response_json) == 2
    assert response_json[0]["entity_id"] == entity_id_test
    assert response_json[1]["entity_id"] == entity_id_second
    response = await client.get(f"/api/logbook/{start_date.isoformat()}?period=1")
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    assert len(response_json) == 2
    assert response_json[0]["entity_id"] == entity_id_test
    assert response_json[1]["entity_id"] == entity_id_second
    response = await client.get(
        f"/api/logbook/{start_date.isoformat()}?entity=switch.test"
    )
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    assert len(response_json) == 1
    assert response_json[0]["entity_id"] == entity_id_test
    response = await client.get(
        f"/api/logbook/{start_date.isoformat()}?period=3&entity=switch.test"
    )
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    assert len(response_json) == 1
    assert response_json[0]["entity_id"] == entity_id_test
    start = (dt_util.utcnow() + timedelta(days=1)).date()
    start_date = datetime(start.year, start.month, start.day, tzinfo=dt_util.UTC)
    response = await client.get(f"/api/logbook/{start_date.isoformat()}")
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
    assert len(response_json) == 0
    response = await client.get(
        f"/api/logbook/{start_date.isoformat()}?entity=switch.test"
    )
    assert response.status == HTTPStatus.OK
    response_json = await response.json()
