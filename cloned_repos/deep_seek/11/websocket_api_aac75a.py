"""Websocket API for the history integration."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta
import logging
from typing import Any, cast, Optional, Union, Dict, List, Tuple
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.recorder import get_instance, history
from homeassistant.components.websocket_api import ActiveConnection, messages
from homeassistant.const import COMPRESSED_STATE_ATTRIBUTES, COMPRESSED_STATE_LAST_CHANGED, COMPRESSED_STATE_LAST_UPDATED, COMPRESSED_STATE_STATE
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback, is_callback, valid_entity_id
from homeassistant.helpers.event import async_track_point_in_utc_time, async_track_state_change_event
from homeassistant.helpers.json import json_bytes
from homeassistant.util import dt as dt_util
from homeassistant.util.async_ import create_eager_task
from .const import EVENT_COALESCE_TIME, MAX_PENDING_HISTORY_STATES
from .helpers import entities_may_have_state_changes_after, has_states_before
_LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class HistoryLiveStream:
    """Track a history live stream."""
    subscriptions: List[CALLBACK_TYPE] = field(default_factory=list)
    stream_queue: asyncio.Queue[Event[EventStateChangedData]] = field(default_factory=lambda: asyncio.Queue(MAX_PENDING_HISTORY_STATES))
    end_time_unsub: Optional[CALLBACK_TYPE] = None
    task: Optional[asyncio.Task[None]] = None
    wait_sync_task: Optional[asyncio.Task[None]] = None

@callback
def async_setup(hass: HomeAssistant) -> None:
    """Set up the history websocket API."""
    websocket_api.async_register_command(hass, ws_get_history_during_period)
    websocket_api.async_register_command(hass, ws_stream)

def _ws_get_significant_states(hass: HomeAssistant, msg_id: int, start_time: dt, end_time: Optional[dt], entity_ids: List[str], include_start_time_state: bool, significant_changes_only: bool, minimal_response: bool, no_attributes: bool) -> bytes:
    """Fetch history significant_states and convert them to json in the executor."""
    return json_bytes(messages.result_message(msg_id, history.get_significant_states(hass, start_time, end_time, entity_ids, None, include_start_time_state, significant_changes_only, minimal_response, no_attributes, True)))

@websocket_api.websocket_command({vol.Required('type'): 'history/history_during_period', vol.Required('start_time'): str, vol.Optional('end_time'): str, vol.Required('entity_ids'): [str], vol.Optional('include_start_time_state', default=True): bool, vol.Optional('significant_changes_only', default=True): bool, vol.Optional('minimal_response', default=False): bool, vol.Optional('no_attributes', default=False): bool})
@websocket_api.async_response
async def ws_get_history_during_period(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle history during period websocket command."""
    start_time_str = msg['start_time']
    end_time_str = msg.get('end_time')
    if (start_time := dt_util.parse_datetime(start_time_str)):
        start_time = dt_util.as_utc(start_time)
    else:
        connection.send_error(msg['id'], 'invalid_start_time', 'Invalid start_time')
        return
    if end_time_str:
        if (end_time := dt_util.parse_datetime(end_time_str)):
            end_time = dt_util.as_utc(end_time)
        else:
            connection.send_error(msg['id'], 'invalid_end_time', 'Invalid end_time')
            return
    else:
        end_time = None
    if start_time > dt_util.utcnow():
        connection.send_result(msg['id'], {})
        return
    entity_ids = msg['entity_ids']
    for entity_id in entity_ids:
        if not hass.states.get(entity_id) and (not valid_entity_id(entity_id)):
            connection.send_error(msg['id'], 'invalid_entity_ids', 'Invalid entity_ids')
            return
    include_start_time_state = msg['include_start_time_state']
    no_attributes = msg['no_attributes']
    if end_time and (not has_states_before(hass, end_time)) or (not include_start_time_state and entity_ids and (not entities_may_have_state_changes_after(hass, entity_ids, start_time, no_attributes))):
        connection.send_result(msg['id'], {})
        return
    significant_changes_only = msg['significant_changes_only']
    minimal_response = msg['minimal_response']
    connection.send_message(await get_instance(hass).async_add_executor_job(_ws_get_significant_states, hass, msg['id'], start_time, end_time, entity_ids, include_start_time_state, significant_changes_only, minimal_response, no_attributes))

def _generate_stream_message(states: Dict[str, List[Dict[str, Any]]], start_day: dt, end_day: dt) -> Dict[str, Any]:
    """Generate a history stream message response."""
    return {'states': states, 'start_time': start_day.timestamp(), 'end_time': end_day.timestamp()}

@callback
def _async_send_empty_response(connection: ActiveConnection, msg_id: int, start_time: dt, end_time: dt) -> None:
    """Send an empty response when we know all results are filtered away."""
    connection.send_result(msg_id)
    stream_end_time = end_time or dt_util.utcnow()
    connection.send_message(_generate_websocket_response(msg_id, start_time, stream_end_time, {}))

def _generate_websocket_response(msg_id: int, start_time: dt, end_time: dt, states: Dict[str, List[Dict[str, Any]]]) -> bytes:
    """Generate a websocket response."""
    return json_bytes(messages.event_message(msg_id, _generate_stream_message(states, start_time, end_time)))

def _generate_historical_response(hass: HomeAssistant, msg_id: int, start_time: dt, end_time: dt, entity_ids: List[str], include_start_time_state: bool, significant_changes_only: bool, minimal_response: bool, no_attributes: bool, send_empty: bool) -> Tuple[float, Optional[dt], Optional[bytes]]:
    """Generate a historical response."""
    states = cast(dict[str, list[dict[str, Any]]], history.get_significant_states(hass, start_time, end_time, entity_ids, None, include_start_time_state, significant_changes_only, minimal_response, no_attributes, True))
    last_time_ts = 0.0
    for state_list in states.values():
        if state_list and (state_last_time := state_list[-1][COMPRESSED_STATE_LAST_UPDATED]) > last_time_ts:
            last_time_ts = cast(float, state_last_time)
    if last_time_ts == 0:
        if not send_empty:
            return (last_time_ts, None, None)
        last_time_dt = end_time
    else:
        last_time_dt = dt_util.utc_from_timestamp(last_time_ts)
    return (last_time_ts, last_time_dt, _generate_websocket_response(msg_id, start_time, last_time_dt, states))

async def _async_send_historical_states(hass: HomeAssistant, connection: ActiveConnection, msg_id: int, start_time: dt, end_time: dt, entity_ids: List[str], include_start_time_state: bool, significant_changes_only: bool, minimal_response: bool, no_attributes: bool, send_empty: bool) -> Optional[dt]:
    """Fetch history significant_states and send them to the client."""
    instance = get_instance(hass)
    last_time_ts, last_time_dt, payload = await instance.async_add_executor_job(_generate_historical_response, hass, msg_id, start_time, end_time, entity_ids, include_start_time_state, significant_changes_only, minimal_response, no_attributes, send_empty)
    if payload:
        connection.send_message(payload)
    return last_time_dt if last_time_ts != 0 else None

def _history_compressed_state(state: State, no_attributes: bool) -> Dict[str, Any]:
    """Convert a state to a compressed state."""
    comp_state = {COMPRESSED_STATE_STATE: state.state}
    if not no_attributes or state.domain in history.NEED_ATTRIBUTE_DOMAINS:
        comp_state[COMPRESSED_STATE_ATTRIBUTES] = state.attributes
    comp_state[COMPRESSED_STATE_LAST_UPDATED] = state.last_updated_timestamp
    if state.last_changed != state.last_updated:
        comp_state[COMPRESSED_STATE_LAST_CHANGED] = state.last_changed_timestamp
    return comp_state

def _events_to_compressed_states(events: List[Event[EventStateChangedData]], no_attributes: bool) -> Dict[str, List[Dict[str, Any]]]:
    """Convert events to a compressed states."""
    states_by_entity_ids: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        state = event.data['new_state']
        entity_id = state.entity_id
        states_by_entity_ids.setdefault(entity_id, []).append(_history_compressed_state(state, no_attributes))
    return states_by_entity_ids

async def _async_events_consumer(subscriptions_setup_complete_time: dt, connection: ActiveConnection, msg_id: int, stream_queue: asyncio.Queue[Event[EventStateChangedData]], no_attributes: bool) -> None:
    """Stream events from the queue."""
    subscriptions_setup_complete_timestamp = subscriptions_setup_complete_time.timestamp()
    while True:
        events = [await stream_queue.get()]
        if events[0].time_fired_timestamp <= subscriptions_setup_complete_timestamp:
            continue
        await asyncio.sleep(EVENT_COALESCE_TIME)
        while not stream_queue.empty():
            events.append(stream_queue.get_nowait())
        if (history_states := _events_to_compressed_states(events, no_attributes)):
            connection.send_message(json_bytes(messages.event_message(msg_id, {'states': history_states})))

@callback
def _async_subscribe_events(hass: HomeAssistant, subscriptions: List[CALLBACK_TYPE], target: Callable[[Event[EventStateChangedData]], None], entity_ids: List[str], significant_changes_only: bool, minimal_response: bool) -> None:
    """Subscribe to events for the entities and devices or all.

    These are the events we need to listen for to do
    the live history stream.
    """
    assert is_callback(target), 'target must be a callback'

    @callback
    def _forward_state_events_filtered(event: Event[EventStateChangedData]) -> None:
        """Filter state events and forward them."""
        if (new_state := event.data['new_state']) is None or (old_state := event.data['old_state']) is None:
            return
        if (significant_changes_only or minimal_response) and new_state.state == old_state.state and (new_state.domain not in history.SIGNIFICANT_DOMAINS):
            return
        target(event)
    subscriptions.append(async_track_state_change_event(hass, entity_ids, _forward_state_events_filtered))

@websocket_api.websocket_command({vol.Required('type'): 'history/stream', vol.Required('start_time'): str, vol.Optional('end_time'): str, vol.Required('entity_ids'): [str], vol.Optional('include_start_time_state', default=True): bool, vol.Optional('significant_changes_only', default=True): bool, vol.Optional('minimal_response', default=False): bool, vol.Optional('no_attributes', default=False): bool})
@websocket_api.async_response
async def ws_stream(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle history stream websocket command."""
    start_time_str = msg['start_time']
    msg_id = msg['id']
    utc_now = dt_util.utcnow()
    if (start_time := dt_util.parse_datetime(start_time_str)):
        start_time = dt_util.as_utc(start_time)
    if not start_time or start_time > utc_now:
        connection.send_error(msg_id, 'invalid_start_time', 'Invalid start_time')
        return
    end_time_str = msg.get('end_time')
    end_time = None
    if end_time_str:
        if not (end_time := dt_util.parse_datetime(end_time_str)):
            connection.send_error(msg_id, 'invalid_end_time', 'Invalid end_time')
            return
        end_time = dt_util.as_utc(end_time)
        if end_time < start_time:
            connection.send_error(msg_id, 'invalid_end_time', 'Invalid end_time')
            return
    entity_ids = msg['entity_ids']
    for entity_id in entity_ids:
        if not hass.states.get(entity_id) and (not valid_entity_id(entity_id)):
            connection.send_error(msg['id'], 'invalid_entity_ids', 'Invalid entity_ids')
            return
    include_start_time_state = msg['include_start_time_state']
    significant_changes_only = msg['significant_changes_only']
    no_attributes = msg['no_attributes']
    minimal_response = msg['minimal_response']
    if end_time and end_time <= utc_now:
        if not include_start_time_state and entity_ids and (not entities_may_have_state_changes_after(hass, entity_ids, start_time, no_attributes)):
            _async_send_empty_response(connection, msg_id, start_time, end_time)
            return
        connection.subscriptions[msg_id] = callback(lambda: None)
        connection.send_result(msg_id)
        await _async_send_historical_states(hass, connection, msg_id, start_time, end_time, entity_ids, include_start_time_state, significant_changes_only, minimal_response, no_attributes, True)
        return
    subscriptions: List[CALLBACK_TYPE] = []
    stream_queue = asyncio.Queue(MAX_PENDING_HISTORY_STATES)
    live_stream = HistoryLiveStream(subscriptions=subscriptions, stream_queue=stream_queue)

    @callback
    def _unsub(*_utc_time: Any) -> None:
        """Unsubscribe from all events."""
        for subscription in subscriptions:
            subscription()
        subscriptions.clear()
        if live_stream.task:
            live_stream.task.cancel()
        if live_stream.wait_sync_task:
            live_stream.wait_sync_task.cancel()
        if live_stream.end_time_unsub:
            live_stream.end_time_unsub()
            live_stream.end_time_unsub = None
    if end_time:
        live_stream.end_time_unsub = async_track_point_in_utc_time(hass, _unsub, end_time)

    @callback
    def _queue_or_cancel(event: Event[EventStateChangedData]) -> None:
        """Queue an event to be processed or cancel."""
        try:
            stream_queue.put_nowait(event)
        except asyncio.QueueFull:
            _LOGGER.debug('Client exceeded max pending messages of %s', MAX_PENDING_HISTORY_STATES)
            _unsub()
    _async_subscribe_events(hass, subscriptions, _queue_or_cancel, entity_ids, significant_changes_only=significant_changes_only, minimal_response=minimal_response)
    subscriptions_setup_complete_time = dt_util.utcnow()
    connection.subscriptions[msg_id] = _unsub
    connection.send_result(msg_id)
    last_event_time = await _async_send_historical_states(hass, connection, msg_id, start_time, subscriptions_setup_complete_time, entity_ids, include_start_time_state, significant_changes_only, minimal_response, no_attributes, True)
    if msg_id not in connection.subscriptions:
        return
    live_stream.task = create_eager_task(_async_events_consumer(subscriptions_setup_complete_time, connection, msg_id, stream_queue, no_attributes))
    live_stream.wait_sync_task = create_eager_task(get_instance(hass).async_block_till_done())
    await live_stream.wait_sync_task
    await _async_send_historical_states(hass, connection, msg_id, (last_event_time or start_time) + timedelta(microseconds=1), subscriptions_setup_complete_time, entity_ids, False, significant_changes_only, minimal_response, no_attributes, send_empty=not last_event_time)
