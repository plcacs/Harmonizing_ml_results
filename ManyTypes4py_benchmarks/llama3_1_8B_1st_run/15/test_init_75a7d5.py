import datetime
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

async def test_service_call_create_logbook_entry(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry."""
    # ... (rest of the code remains the same)

async def test_service_call_create_logbook_entry_invalid_entity_id(hass: HomeAssistant) -> None:
    """Test if service call create log book entry with an invalid entity id."""
    # ... (rest of the code remains the same)

async def test_service_call_create_log_book_entry_no_message(hass_: HomeAssistant) -> None:
    """Test if service call create log book entry without message."""
    # ... (rest of the code remains the same)

async def test_filter_sensor(hass_: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test numeric sensors are filtered."""
    # ... (rest of the code remains the same)

async def test_home_assistant_start_stop_not_grouped(hass_: HomeAssistant) -> None:
    """Test if HA start and stop events are no longer grouped."""
    # ... (rest of the code remains the same)

async def test_home_assistant_start(hass_: HomeAssistant) -> None:
    """Test if HA start is not filtered or converted into a restart."""
    # ... (rest of the code remains the same)

def test_process_custom_logbook_entries(hass_: HomeAssistant) -> None:
    """Test if custom log book entries get added as an entry."""
    # ... (rest of the code remains the same)

def assert_entry(entry: Dict[str, Any], when: Optional[datetime.datetime] = None, name: Optional[str] = None, message: Optional[str] = None, domain: Optional[str] = None, entity_id: Optional[str] = None) -> None:
    """Assert an entry is what is expected."""
    # ... (rest of the code remains the same)

async def test_logbook_view(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view."""
    # ... (rest of the code remains the same)

async def test_logbook_view_invalid_start_date_time(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with an invalid date time."""
    # ... (rest of the code remains the same)

async def test_logbook_view_invalid_end_date_time(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view."""
    # ... (rest of the code remains the same)

async def test_logbook_view_period_entity(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with period and entity."""
    # ... (rest of the code remains the same)

async def test_logbook_describe_event(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test teaching logbook about a new event."""
    # ... (rest of the code remains the same)

async def test_exclude_described_event(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test exclusions of events that are described by another integration."""
    # ... (rest of the code remains the same)

async def test_logbook_view_end_time_entity(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with end_time and entity."""
    # ... (rest of the code remains the same)

async def test_logbook_entity_filter_with_automations(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    # ... (rest of the code remains the same)

async def test_logbook_entity_no_longer_in_state_machine(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with an entity that hass been removed from the state machine."""
    # ... (rest of the code remains the same)

async def test_filter_continuous_sensor_values(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test remove continuous sensor events from logbook."""
    # ... (rest of the code remains the same)

async def test_exclude_new_entities(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are excluded on first update."""
    # ... (rest of the code remains the same)

async def test_exclude_removed_entities(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are excluded on last update."""
    # ... (rest of the code remains the same)

async def test_exclude_attribute_changes(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events of attribute changes are filtered."""
    # ... (rest of the code remains the same)

async def test_logbook_entity_context_id(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    # ... (rest of the code remains the same)

async def test_logbook_context_id_automation_script_started_manually(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook populates context_ids for scripts and automations started manually."""
    # ... (rest of the code remains the same)

async def test_logbook_entity_context_parent_id(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view links events via context parent_id."""
    # ... (rest of the code remains the same)

async def test_logbook_context_from_template(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    # ... (rest of the code remains the same)

async def test_logbook_(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with a single entity and ."""
    # ... (rest of the code remains the same)

async def test_logbook_many_entities_multiple_calls(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with a many entities called multiple times."""
    # ... (rest of the code remains the same)

async def test_custom_log_entry_discoverable_via_(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if a custom log entry is later discoverable via ."""
    # ... (rest of the code remains the same)

async def test_logbook_multiple_entities(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with a multiple entities."""
    # ... (rest of the code remains the same)

async def test_logbook_invalid_entity(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with requesting an invalid entity."""
    # ... (rest of the code remains the same)

async def test_icon_and_state(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test to ensure state and custom icons are returned."""
    # ... (rest of the code remains the same)

async def test_fire_logbook_entries(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test many logbook entry calls."""
    # ... (rest of the code remains the same)

async def test_exclude_events_domain(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if domain is excluded in config."""
    # ... (rest of the code remains the same)

async def test_exclude_events_domain_glob(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if domain or glob is excluded in config."""
    # ... (rest of the code remains the same)

async def test_include_events_entity(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if entity is included in config."""
    # ... (rest of the code remains the same)

async def test_exclude_events_entity(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if entity is excluded in config."""
    # ... (rest of the code remains the same)

async def test_include_events_domain(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if domain is included in config."""
    # ... (rest of the code remains the same)

async def test_include_events_domain_glob(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if domain or glob is included in config."""
    # ... (rest of the code remains the same)

async def test_include_exclude_events_no_globs(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if include and exclude is configured."""
    # ... (rest of the code remains the same)

async def test_include_exclude_events_with_glob_filters(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test if events are filtered if include and exclude is configured."""
    # ... (rest of the code remains the same)

async def test_empty_config(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test we can handle an empty entity filter."""
    # ... (rest of the code remains the same)

async def test_context_filter(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test we can filter by context."""
    # ... (rest of the code remains the same)

async def _async_fetch_logbook(client: WebSocketGenerator, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Fetch logbook."""
    # ... (rest of the code remains the same)

async def test_get_events(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test logbook get_events."""
    # ... (rest of the code remains the same)

async def test_get_events_future_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events with a future start time."""
    # ... (rest of the code remains the same)

async def test_get_events_bad_start_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events bad start time."""
    # ... (rest of the code remains the same)

async def test_get_events_bad_end_time(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events bad end time."""
    # ... (rest of the code remains the same)

async def test_get_events_invalid_filters(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test get_events invalid filters."""
    # ... (rest of the code remains the same)

async def test_get_events_with_device_ids(hass: HomeAssistant, hass_ws_client: WebSocketGenerator, device_registry: device_registry.DeviceRegistry) -> None:
    """Test logbook get_events for device ids."""
    # ... (rest of the code remains the same)

async def test_logbook_select_entities_context_id(hass: HomeAssistant, hass_client: WebSocketGenerator) -> None:
    """Test the logbook view with end_time and entity with automations and scripts."""
    # ... (rest of the code remains the same)

async def test_get_events_with_context_state(hass: HomeAssistant, hass_ws_client: WebSocketGenerator) -> None:
    """Test logbook get_events with a context state."""
    # ... (rest of the code remains the same)

async def test_logbook_with_empty_config(hass: HomeAssistant) -> None:
    """Test we handle a empty configuration."""
    # ... (rest of the code remains the same)

async def test_logbook_with_non_iterable_entity_filter(hass: HomeAssistant) -> None:
    """Test we handle a non-iterable entity filter."""
    # ... (rest of the code remains the same)
