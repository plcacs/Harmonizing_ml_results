"""Event parser and human readable log generator."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Iterable
from typing import Any, List, Set, Optional, Tuple, Sequence
from homeassistant.components.sensor import ATTR_STATE_CLASS
from homeassistant.const import (
    ATTR_DEVICE_ID,
    ATTR_DOMAIN,
    ATTR_ENTITY_ID,
    ATTR_UNIT_OF_MEASUREMENT,
    EVENT_LOGBOOK_ENTRY,
    EVENT_STATE_CHANGED,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    HomeAssistant,
    State,
    callback,
    is_callback,
    split_entity_id,
)
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util.event_type import EventType
from .const import ALWAYS_CONTINUOUS_DOMAINS, AUTOMATION_EVENTS, BUILT_IN_EVENTS, DOMAIN
from .models import LogbookConfig


def async_filter_entities(
    hass: HomeAssistant, entity_ids: Sequence[str]
) -> List[str]:
    """Filter out any entities that logbook will not produce results for."""
    ent_reg = er.async_get(hass)
    return [
        entity_id
        for entity_id in entity_ids
        if split_entity_id(entity_id)[0] not in ALWAYS_CONTINUOUS_DOMAINS
        and not is_sensor_continuous(hass, ent_reg, entity_id)
    ]


@callback
def _async_config_entries_for_ids(
    hass: HomeAssistant,
    entity_ids: Optional[Sequence[str]],
    device_ids: Optional[Sequence[str]],
) -> Set[str]:
    """Find the config entry ids for a set of entities or devices."""
    config_entry_ids: Set[str] = set()
    if entity_ids:
        ent_reg = er.async_get(hass)
        for entity_id in entity_ids:
            entry = ent_reg.async_get(entity_id)
            if entry and entry.config_entry_id:
                config_entry_ids.add(entry.config_entry_id)
    if device_ids:
        dev_reg = dr.async_get(hass)
        for device_id in device_ids:
            device = dev_reg.async_get(device_id)
            if device and device.config_entries:
                config_entry_ids |= device.config_entries
    return config_entry_ids


def async_determine_event_types(
    hass: HomeAssistant,
    entity_ids: Optional[Sequence[str]],
    device_ids: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Reduce the event types based on the entity ids and device ids."""
    logbook_config: LogbookConfig = hass.data[DOMAIN]
    external_events: Mapping[str, Tuple[str, Callable[..., Any]]] = logbook_config.external_events
    if not entity_ids and not device_ids:
        return (*BUILT_IN_EVENTS, *external_events)
    interested_domains: Set[str] = set()
    for entry_id in _async_config_entries_for_ids(hass, entity_ids, device_ids):
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry:
            interested_domains.add(entry.domain)
    interested_event_types: Set[str] = {
        external_event
        for external_event, domain_call in external_events.items()
        if domain_call[0] in interested_domains
    } | AUTOMATION_EVENTS
    if entity_ids:
        interested_event_types.add(EVENT_LOGBOOK_ENTRY)
    return tuple(interested_event_types)


@callback
def extract_attr(source: Mapping[str, Any], attr: str) -> List[str]:
    """Extract an attribute as a list or string."""
    value = source.get(attr)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return str(value).split(",")


@callback
def event_forwarder_filtered(
    target: Callable[[Event], None],
    entities_filter: Optional[Callable[[str], bool]],
    entity_ids: Optional[Sequence[str]],
    device_ids: Optional[Sequence[str]],
) -> Callable[[Event], None]:
    """Make a callable to filter events."""
    if not entities_filter and not entity_ids and not device_ids:
        return target
    if entities_filter:

        @callback
        def _forward_events_filtered_by_entities_filter(event: Event) -> None:
            assert entities_filter is not None
            event_data = event.data
            entity_ids_extracted = extract_attr(event_data, ATTR_ENTITY_ID)
            if entity_ids_extracted and not any(
                entities_filter(entity_id) for entity_id in entity_ids_extracted
            ):
                return
            domain = event_data.get(ATTR_DOMAIN)
            if domain and not entities_filter(f"{domain}._"):
                return
            target(event)

        return _forward_events_filtered_by_entities_filter

    entity_ids_set: Set[str] = set(entity_ids) if entity_ids else set()
    device_ids_set: Set[str] = set(device_ids) if device_ids else set()

    @callback
    def _forward_events_filtered_by_device_entity_ids(event: Event) -> None:
        event_data = event.data
        if (
            entity_ids_set.intersection(extract_attr(event_data, ATTR_ENTITY_ID))
            or device_ids_set.intersection(extract_attr(event_data, ATTR_DEVICE_ID))
        ):
            target(event)

    return _forward_events_filtered_by_device_entity_ids


@callback
def async_subscribe_events(
    hass: HomeAssistant,
    subscriptions: List[Callable[[], None]],
    target: CALLBACK_TYPE,
    event_types: Iterable[str],
    entities_filter: Optional[Callable[[str], bool]],
    entity_ids: Optional[Sequence[str]],
    device_ids: Optional[Sequence[str]],
) -> None:
    """Subscribe to events for the entities and devices or all.

    These are the events we need to listen for to do
    the live logbook stream.
    """
    assert is_callback(target), "target must be a callback"
    event_forwarder = event_forwarder_filtered(
        target, entities_filter, entity_ids, device_ids
    )
    for event_type in event_types:
        subscription = hass.bus.async_listen(event_type, event_forwarder)
        subscriptions.append(subscription)
    if device_ids and not entity_ids:
        return

    @callback
    def _forward_state_events_filtered(event: Event) -> None:
        old_state: Optional[State] = event.data.get("old_state")
        new_state: Optional[State] = event.data.get("new_state")
        if old_state is None or new_state is None:
            return
        if (
            _is_state_filtered(new_state, old_state)
            or (entities_filter and not entities_filter(new_state.entity_id))
        ):
            return
        target(event)

    if entity_ids:
        subscriptions.append(
            async_track_state_change_event(hass, entity_ids, _forward_state_events_filtered)
        )
        return
    subscriptions.append(hass.bus.async_listen(EVENT_STATE_CHANGED, _forward_state_events_filtered))


def is_sensor_continuous(
    hass: HomeAssistant, ent_reg: er.EntityRegistry, entity_id: str
) -> bool:
    """Determine if a sensor is continuous.

    Sensors with a unit_of_measurement or state_class are considered continuous.

    The unit_of_measurement check will already happen if this is
    called for historical data because the SQL query generated by _get_events
    will filter out any sensors with a unit_of_measurement.

    If the state still exists in the state machine, this function still
    checks for ATTR_UNIT_OF_MEASUREMENT since the live mode is not filtered
    by the SQL query.
    """
    state: Optional[State] = hass.states.get(entity_id)
    if state and (attributes := state.attributes):
        return (
            ATTR_UNIT_OF_MEASUREMENT in attributes
            or ATTR_STATE_CLASS in attributes
        )
    entry = ent_reg.async_get(entity_id)
    return bool(
        entry
        and entry.capabilities
        and entry.capabilities.get(ATTR_STATE_CLASS)
    )


def _is_state_filtered(new_state: State, old_state: State) -> bool:
    """Check if the logbook should filter a state.

    Used when we are in live mode to ensure
    we only get significant changes (state.last_changed != state.last_updated)
    """
    return bool(
        new_state.state == old_state.state
        or new_state.last_changed != new_state.last_updated
        or new_state.domain in ALWAYS_CONTINUOUS_DOMAINS
        or ATTR_UNIT_OF_MEASUREMENT in new_state.attributes
        or ATTR_STATE_CLASS in new_state.attributes
    )
