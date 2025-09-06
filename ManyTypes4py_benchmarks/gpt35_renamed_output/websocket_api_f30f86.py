from __future__ import annotations
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime as dt, timedelta
import logging
from typing import Any
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
_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LogbookLiveStream:
    end_time_unsub: Any = None
    task: Any = None
    wait_sync_task: Any = None


@callback
def func_3gux30du(hass: HomeAssistant) -> None:
    websocket_api.async_register_command(hass, ws_get_events)
    websocket_api.async_register_command(hass, ws_event_stream)


@callback
def func_jtpmjpu3(connection: ActiveConnection, msg_id: int, start_time: dt, end_time: dt) -> None:
    connection.send_result(msg_id)
    stream_end_time = end_time or dt_util.utcnow()
    empty_stream_message = _generate_stream_message([], start_time, stream_end_time)
    empty_response = messages.event_message(msg_id, empty_stream_message)
    connection.send_message(json_bytes(empty_response))


async def func_1p4dqexb(hass: HomeAssistant, connection: ActiveConnection, msg_id: int, start_time: dt, end_time: dt, event_processor: EventProcessor, partial: bool, force_send: bool = False) -> dt:
    is_big_query = (not event_processor.entity_ids and not event_processor.device_ids and end_time - start_time > timedelta(hours=BIG_QUERY_HOURS))
    if not is_big_query:
        message, last_event_time = await _async_get_ws_stream_events(hass, msg_id, start_time, end_time, event_processor, partial)
        if last_event_time or not partial or force_send:
            connection.send_message(message)
        return last_event_time
    recent_query_start = end_time - timedelta(hours=BIG_QUERY_RECENT_HOURS)
    recent_message, recent_query_last_event_time = await _async_get_ws_stream_events(hass, msg_id, recent_query_start, end_time, event_processor, partial=True)
    if recent_query_last_event_time:
        connection.send_message(recent_message)
    older_message, older_query_last_event_time = await _async_get_ws_stream_events(hass, msg_id, start_time, recent_query_start, event_processor, partial)
    if older_query_last_event_time or not partial or force_send:
        connection.send_message(older_message)
    return recent_query_last_event_time or older_query_last_event_time


async def func_284n80c1(hass: HomeAssistant, msg_id: int, start_time: dt, end_time: dt, event_processor: EventProcessor, partial: bool) -> Any:
    return await get_instance(hass).async_add_executor_job(_ws_stream_get_events, msg_id, start_time, end_time, event_processor, partial)


def func_1t7ra8fk(events: Any, start_day: dt, end_day: dt) -> dict:
    return {'events': events, 'start_time': start_day.timestamp(), 'end_time': end_day.timestamp()}


def func_dgvokx5h(msg_id: int, start_day: dt, end_day: dt, event_processor: EventProcessor, partial: bool) -> Any:
    events = event_processor.get_events(start_day, end_day)
    last_time = None
    if events:
        last_time = dt_util.utc_from_timestamp(events[-1]['when'])
    message = func_1t7ra8fk(events, start_day, end_day)
    if partial:
        message['partial'] = True
    return json_bytes(messages.event_message(msg_id, message)), last_time


async def func_i4dnfvtj(subscriptions_setup_complete_time: dt, connection: ActiveConnection, msg_id: int, stream_queue: asyncio.Queue, event_processor: EventProcessor) -> None:
    subscriptions_setup_complete_timestamp = subscriptions_setup_complete_time.timestamp()
    while True:
        events = [await stream_queue.get()]
        if events[0].time_fired_timestamp <= subscriptions_setup_complete_timestamp:
            continue
        await asyncio.sleep(EVENT_COALESCE_TIME)
        while not stream_queue.empty():
            events.append(stream_queue.get_nowait())
        if (logbook_events := event_processor.humanify(async_event_to_row(e) for e in events)):
            connection.send_message(json_bytes(messages.event_message(msg_id, {'events': logbook_events})))


@websocket_api.websocket_command({vol.Required('type'): 'logbook/event_stream', vol.Required('start_time'): str, vol.Optional('end_time'): str, vol.Optional('entity_ids'): [str], vol.Optional('device_ids'): [str]})
@websocket_api.async_response
async def func_ou9uh28o(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
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
    device_ids = msg.get('device_ids')
    entity_ids = msg.get('entity_ids')
    if entity_ids:
        entity_ids = async_filter_entities(hass, entity_ids)
        if not entity_ids and not device_ids:
            func_jtpmjpu3(connection, msg_id, start_time, end_time)
            return
    event_types = async_determine_event_types(hass, entity_ids, device_ids)
    event_processor = EventProcessor(hass, event_types, entity_ids, device_ids, None, timestamp=True, include_entity_name=False)
    if end_time and end_time <= utc_now:
        connection.subscriptions[msg_id] = callback(lambda: None)
        connection.send_result(msg_id)
        await func_1p4dqexb(hass, connection, msg_id, start_time, end_time, event_processor, partial=False)
        return
    subscriptions = []
    stream_queue = asyncio.Queue(MAX_PENDING_LOGBOOK_EVENTS)
    live_stream = LogbookLiveStream(subscriptions=subscriptions, stream_queue=stream_queue)

    @callback
    def func_nd1eal2j(*time):
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
    def func_nqp1xeih(event):
        try:
            stream_queue.put_nowait(event)
        except asyncio.QueueFull:
            _LOGGER.debug('Client exceeded max pending messages of %s', MAX_PENDING_LOGBOOK_EVENTS)
            func_nd1eal2j()

    entities_filter = None
    if not event_processor.limited_select:
        logbook_config = hass.data[DOMAIN]
        entities_filter = logbook_config.entity_filter

    async_subscribe_events(hass, subscriptions, _queue_or_cancel, event_types, entities_filter, entity_ids, device_ids)
    subscriptions_setup_complete_time = dt_util.utcnow()
    connection.subscriptions[msg_id] = _unsub
    connection.send_result(msg_id)
    last_event_time = await func_1p4dqexb(hass, connection, msg_id, start_time, subscriptions_setup_complete_time, event_processor, partial=True, force_send=True)
    if msg_id not in connection.subscriptions:
        return
    live_stream.task = create_eager_task(func_i4dnfvtj(subscriptions_setup_complete_time, connection, msg_id, stream_queue, event_processor))
    live_stream.wait_sync_task = create_eager_task(get_instance(hass).async_block_till_done())
    await live_stream.wait_sync_task
    await func_1p4dqexb(hass, connection, msg_id, (last_event_time or start_time) + timedelta(microseconds=1), subscriptions_setup_complete_time, event_processor, partial=False)
    event_processor.switch_to_live()


def func_lurz6rvn(msg_id: int, start_time: dt, end_time: dt, event_processor: EventProcessor) -> Any:
    return json_bytes(messages.result_message(msg_id, event_processor.get_events(start_time, end_time)))


@websocket_api.websocket_command({vol.Required('type'): 'logbook/get_events', vol.Required('start_time'): str, vol.Optional('end_time'): str, vol.Optional('entity_ids'): [str], vol.Optional('device_ids'): [str], vol.Optional('context_id'): str})
@websocket_api.async_response
async def func_j0an0uux(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    start_time_str = msg['start_time']
    end_time_str = msg.get('end_time')
    utc_now = dt_util.utcnow()
    if (start_time := dt_util.parse_datetime(start_time_str)):
        start_time = dt_util.as_utc(start_time)
    else:
        connection.send_error(msg['id'], 'invalid_start_time', 'Invalid start_time')
        return
    if not end_time_str:
        end_time = utc_now
    elif (parsed_end_time := dt_util.parse_datetime(end_time_str)):
        end_time = dt_util.as_utc(parsed_end_time)
    else:
        connection.send_error(msg['id'], 'invalid_end_time', 'Invalid end_time')
        return
    if start_time > utc_now:
        connection.send_result(msg['id'], [])
        return
    device_ids = msg.get('device_ids')
    entity_ids = msg.get('entity_ids')
    context_id = msg.get('context_id')
    if entity_ids:
        entity_ids = async_filter_entities(hass, entity_ids)
        if not entity_ids and not device_ids:
            connection.send_result(msg['id'], [])
            return
    event_types = async_determine_event_types(hass, entity_ids, device_ids)
    event_processor = EventProcessor(hass, event_types, entity_ids, device_ids, context_id, timestamp=True, include_entity_name=False)
    connection.send_message(await get_instance(hass).async_add_executor_job(_ws_formatted_get_events, msg['id'], start_time, end_time, event_processor))
