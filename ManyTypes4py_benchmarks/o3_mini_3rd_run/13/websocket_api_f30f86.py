#!/usr/bin/env python3
"""Event parser and human readable log generator."""
from __future__ import annotations
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta
import logging
from typing import Any, Optional, List, Tuple, Dict

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.components.recorder import get_instance
from homeassistant.components.websocket_api import ActiveConnection, messages
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.json import json_bytes
from homeassistant.util import dt as dt_util
from homeassistant.util.async_ import create_eager_task

from .const import DOMAIN
from .helpers import async_determine_event_types, async_filter_entities, async_subscribe_events
from .models import LogbookConfig, async_event_to_row
from .processor import EventProcessor

MAX_PENDING_LOGBOOK_EVENTS: int = 2048
EVENT_COALESCE_TIME: float = 0.35
BIG_QUERY_HOURS: int = 25
BIG_QUERY_RECENT_HOURS: int = 24

_LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class LogbookLiveStream:
    """Track a logbook live stream."""
    subscriptions: List[CALLBACK_TYPE] = field(default_factory=list)
    stream_queue: asyncio.Queue[Event] = field(default_factory=asyncio.Queue)
    end_time_unsub: Optional[Callable[[], None]] = None
    task: Optional[asyncio.Task] = None
    wait_sync_task: Optional[asyncio.Task] = None

@callback
def async_setup(hass: HomeAssistant) -> None:
    """Set up the logbook websocket API."""
    websocket_api.async_register_command(hass, ws_get_events)
    websocket_api.async_register_command(hass, ws_event_stream)

@callback
def _async_send_empty_response(
    connection: ActiveConnection,
    msg_id: int,
    start_time: dt,
    end_time: Optional[dt]
) -> None:
    """Send an empty response.

    The current case for this is when they ask for entity_ids
    that will all be filtered away because they have UOMs or
    state_class.
    """
    connection.send_result(msg_id)
    stream_end_time: dt = end_time or dt_util.utcnow()
    empty_stream_message: Dict[str, Any] = _generate_stream_message([], start_time, stream_end_time)
    empty_response: Dict[str, Any] = messages.event_message(msg_id, empty_stream_message)
    connection.send_message(json_bytes(empty_response))

async def _async_send_historical_events(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg_id: int,
    start_time: dt,
    end_time: dt,
    event_processor: EventProcessor,
    partial: bool,
    force_send: bool = False
) -> Optional[dt]:
    """Select historical data from the database and deliver it to the websocket.

    If the query is considered a big query we will split the request into
    two chunks so that they get the recent events first and the select
    that is expected to take a long time comes in after to ensure
    they are not stuck at a loading screen and can start looking at
    the data right away.

    This function returns the time of the most recent event we sent to the
    websocket.
    """
    is_big_query: bool = (
        not event_processor.entity_ids and
        not event_processor.device_ids and
        (end_time - start_time > timedelta(hours=BIG_QUERY_HOURS))
    )
    if not is_big_query:
        message, last_event_time = await _async_get_ws_stream_events(hass, msg_id, start_time, end_time, event_processor, partial)
        if last_event_time or not partial or force_send:
            connection.send_message(message)
        return last_event_time
    recent_query_start: dt = end_time - timedelta(hours=BIG_QUERY_RECENT_HOURS)
    recent_message, recent_query_last_event_time = await _async_get_ws_stream_events(hass, msg_id, recent_query_start, end_time, event_processor, partial=True)
    if recent_query_last_event_time:
        connection.send_message(recent_message)
    older_message, older_query_last_event_time = await _async_get_ws_stream_events(hass, msg_id, start_time, recent_query_start, event_processor, partial)
    if older_query_last_event_time or not partial or force_send:
        connection.send_message(older_message)
    return recent_query_last_event_time or older_query_last_event_time

async def _async_get_ws_stream_events(
    hass: HomeAssistant,
    msg_id: int,
    start_time: dt,
    end_time: dt,
    event_processor: EventProcessor,
    partial: bool
) -> Tuple[bytes, Optional[dt]]:
    """Async wrapper around _ws_formatted_get_events."""
    return await get_instance(hass).async_add_executor_job(
        _ws_stream_get_events, msg_id, start_time, end_time, event_processor, partial
    )

def _generate_stream_message(
    events: List[Any],
    start_day: dt,
    end_day: dt
) -> Dict[str, Any]:
    """Generate a logbook stream message response."""
    return {'events': events, 'start_time': start_day.timestamp(), 'end_time': end_day.timestamp()}

def _ws_stream_get_events(
    msg_id: int,
    start_day: dt,
    end_day: dt,
    event_processor: EventProcessor,
    partial: bool
) -> Tuple[bytes, Optional[dt]]:
    """Fetch events and convert them to json in the executor."""
    events: List[Dict[str, Any]] = event_processor.get_events(start_day, end_day)
    last_time: Optional[dt] = None
    if events:
        last_time = dt_util.utc_from_timestamp(events[-1]['when'])
    message: Dict[str, Any] = _generate_stream_message(events, start_day, end_day)
    if partial:
        message['partial'] = True
    return (json_bytes(messages.event_message(msg_id, message)), last_time)

async def _async_events_consumer(
    subscriptions_setup_complete_time: dt,
    connection: ActiveConnection,
    msg_id: int,
    stream_queue: asyncio.Queue[Event],
    event_processor: EventProcessor
) -> None:
    """Stream events from the queue."""
    subscriptions_setup_complete_timestamp: float = subscriptions_setup_complete_time.timestamp()
    while True:
        events: List[Event] = [await stream_queue.get()]
        if events[0].time_fired_timestamp <= subscriptions_setup_complete_timestamp:
            continue
        await asyncio.sleep(EVENT_COALESCE_TIME)
        while not stream_queue.empty():
            events.append(stream_queue.get_nowait())
        logbook_events: List[Any] = event_processor.humanify((async_event_to_row(e) for e in events))
        if logbook_events:
            connection.send_message(json_bytes(messages.event_message(msg_id, {'events': logbook_events})))

@websocket_api.websocket_command({
    vol.Required('type'): 'logbook/event_stream',
    vol.Required('start_time'): str,
    vol.Optional('end_time'): str,
    vol.Optional('entity_ids'): [str],
    vol.Optional('device_ids'): [str]
})
@websocket_api.async_response
async def ws_event_stream(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle logbook stream events websocket command."""
    start_time_str: str = msg['start_time']
    msg_id: int = msg['id']
    utc_now: dt = dt_util.utcnow()
    start_time: Optional[dt] = dt_util.parse_datetime(start_time_str)
    if start_time:
        start_time = dt_util.as_utc(start_time)
    if not start_time or start_time > utc_now:
        connection.send_error(msg_id, 'invalid_start_time', 'Invalid start_time')
        return

    end_time_str: Optional[str] = msg.get('end_time')
    end_time: Optional[dt] = None
    if end_time_str:
        end_time = dt_util.parse_datetime(end_time_str)
        if not end_time:
            connection.send_error(msg_id, 'invalid_end_time', 'Invalid end_time')
            return
        end_time = dt_util.as_utc(end_time)
        if end_time < start_time:
            connection.send_error(msg_id, 'invalid_end_time', 'Invalid end_time')
            return

    device_ids: Optional[List[str]] = msg.get('device_ids')
    entity_ids: Optional[List[str]] = msg.get('entity_ids')
    if entity_ids:
        entity_ids = async_filter_entities(hass, entity_ids)
        if not entity_ids and (not device_ids):
            _async_send_empty_response(connection, msg_id, start_time, end_time)
            return

    event_types: List[str] = async_determine_event_types(hass, entity_ids, device_ids)
    event_processor: EventProcessor = EventProcessor(
        hass,
        event_types,
        entity_ids,
        device_ids,
        None,
        timestamp=True,
        include_entity_name=False
    )
    if end_time and end_time <= utc_now:
        connection.subscriptions[msg_id] = lambda: None
        connection.send_result(msg_id)
        await _async_send_historical_events(hass, connection, msg_id, start_time, end_time, event_processor, partial=False)
        return

    subscriptions: List[CALLBACK_TYPE] = []
    stream_queue: asyncio.Queue[Event] = asyncio.Queue(MAX_PENDING_LOGBOOK_EVENTS)
    live_stream: LogbookLiveStream = LogbookLiveStream(subscriptions=subscriptions, stream_queue=stream_queue)

    @callback
    def _unsub(*args: Any, **kwargs: Any) -> None:
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
    def _queue_or_cancel(event: Event) -> None:
        """Queue an event to be processed or cancel."""
        try:
            stream_queue.put_nowait(event)
        except asyncio.QueueFull:
            _LOGGER.debug('Client exceeded max pending messages of %s', MAX_PENDING_LOGBOOK_EVENTS)
            _unsub()

    entities_filter: Optional[Callable[[str], bool]] = None
    if not event_processor.limited_select:
        logbook_config: LogbookConfig = hass.data[DOMAIN]
        entities_filter = logbook_config.entity_filter

    async_subscribe_events(hass, subscriptions, _queue_or_cancel, event_types, entities_filter, entity_ids, device_ids)
    subscriptions_setup_complete_time: dt = dt_util.utcnow()
    connection.subscriptions[msg_id] = _unsub
    connection.send_result(msg_id)

    last_event_time: Optional[dt] = await _async_send_historical_events(
        hass,
        connection,
        msg_id,
        start_time,
        subscriptions_setup_complete_time,
        event_processor,
        partial=True,
        force_send=True
    )
    if msg_id not in connection.subscriptions:
        return
    live_stream.task = create_eager_task(
        _async_events_consumer(subscriptions_setup_complete_time, connection, msg_id, stream_queue, event_processor)
    )
    live_stream.wait_sync_task = create_eager_task(get_instance(hass).async_block_till_done())
    await live_stream.wait_sync_task
    await _async_send_historical_events(
        hass,
        connection,
        msg_id,
        (last_event_time or start_time) + timedelta(microseconds=1),
        subscriptions_setup_complete_time,
        event_processor,
        partial=False
    )
    event_processor.switch_to_live()

def _ws_formatted_get_events(
    msg_id: int,
    start_time: dt,
    end_time: dt,
    event_processor: EventProcessor
) -> bytes:
    """Fetch events and convert them to json in the executor."""
    return json_bytes(messages.result_message(msg_id, event_processor.get_events(start_time, end_time)))

@websocket_api.websocket_command({
    vol.Required('type'): 'logbook/get_events',
    vol.Required('start_time'): str,
    vol.Optional('end_time'): str,
    vol.Optional('entity_ids'): [str],
    vol.Optional('device_ids'): [str],
    vol.Optional('context_id'): str
})
@websocket_api.async_response
async def ws_get_events(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle logbook get events websocket command."""
    start_time_str: str = msg['start_time']
    end_time_str: Optional[str] = msg.get('end_time')
    utc_now: dt = dt_util.utcnow()
    start_time: Optional[dt] = dt_util.parse_datetime(start_time_str)
    if start_time:
        start_time = dt_util.as_utc(start_time)
    else:
        connection.send_error(msg['id'], 'invalid_start_time', 'Invalid start_time')
        return

    if not end_time_str:
        end_time: dt = utc_now
    elif (parsed_end_time := dt_util.parse_datetime(end_time_str)):
        end_time = dt_util.as_utc(parsed_end_time)
    else:
        connection.send_error(msg['id'], 'invalid_end_time', 'Invalid end_time')
        return

    if start_time > utc_now:
        connection.send_result(msg['id'], [])
        return

    device_ids: Optional[List[str]] = msg.get('device_ids')
    entity_ids: Optional[List[str]] = msg.get('entity_ids')
    context_id: Optional[str] = msg.get('context_id')
    if entity_ids:
        entity_ids = async_filter_entities(hass, entity_ids)
        if not entity_ids and (not device_ids):
            connection.send_result(msg['id'], [])
            return

    event_types: List[str] = async_determine_event_types(hass, entity_ids, device_ids)
    event_processor: EventProcessor = EventProcessor(
        hass,
        event_types,
        entity_ids,
        device_ids,
        context_id,
        timestamp=True,
        include_entity_name=False
    )
    result: bytes = await get_instance(hass).async_add_executor_job(
        _ws_formatted_get_events, msg['id'], start_time, end_time, event_processor
    )
    connection.send_message(result)