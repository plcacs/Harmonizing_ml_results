"""Message templates for websocket commands."""
from __future__ import annotations
from functools import lru_cache
import logging
from typing import Any, Final
import voluptuous as vol
from homeassistant.const import COMPRESSED_STATE_ATTRIBUTES, COMPRESSED_STATE_CONTEXT, COMPRESSED_STATE_LAST_CHANGED, COMPRESSED_STATE_LAST_UPDATED, COMPRESSED_STATE_STATE
from homeassistant.core import CompressedState, Event, EventStateChangedData
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.json import JSON_DUMP, find_paths_unserializable_data, json_bytes
from homeassistant.util.json import format_unserializable_data
from . import const
_LOGGER = logging.getLogger(__name__)
MINIMAL_MESSAGE_SCHEMA = vol.Schema({vol.Required('id'): cv.positive_int, vol.Required('type'): cv.string}, extra=vol.ALLOW_EXTRA)
BASE_COMMAND_MESSAGE_SCHEMA = vol.Schema({vol.Required('id'): cv.positive_int})
STATE_DIFF_ADDITIONS = '+'
STATE_DIFF_REMOVALS = '-'
ENTITY_EVENT_ADD = 'a'
ENTITY_EVENT_REMOVE = 'r'
ENTITY_EVENT_CHANGE = 'c'
BASE_ERROR_MESSAGE = {'type': const.TYPE_RESULT, 'success': False}
INVALID_JSON_PARTIAL_MESSAGE = json_bytes({**BASE_ERROR_MESSAGE, 'error': {'code': const.ERR_UNKNOWN_ERROR, 'message': 'Invalid JSON in response'}})

def result_message(iden, result=None):
    """Return a success result message."""
    return {'id': iden, 'type': const.TYPE_RESULT, 'success': True, 'result': result}

def construct_result_message(iden, payload):
    """Construct a success result message JSON."""
    return b''.join((b'{"id":', str(iden).encode(), b',"type":"result","success":true,"result":', payload, b'}'))

def error_message(iden, code, message, translation_key=None, translation_domain=None, translation_placeholders=None):
    """Return an error result message."""
    error_payload = {'code': code, 'message': message}
    if translation_key is not None:
        error_payload['translation_key'] = translation_key
        error_payload['translation_placeholders'] = translation_placeholders
        error_payload['translation_domain'] = translation_domain
    return {'id': iden, **BASE_ERROR_MESSAGE, 'error': error_payload}

def event_message(iden, event):
    """Return an event message."""
    return {'id': iden, 'type': 'event', 'event': event}

def cached_event_message(message_id_as_bytes, event):
    """Return an event message.

    Serialize to json once per message.

    Since we can have many clients connected that are
    all getting many of the same events (mostly state changed)
    we can avoid serializing the same data for each connection.
    """
    return b''.join((_partial_cached_event_message(event)[:-1], b',"id":', message_id_as_bytes, b'}'))

@lru_cache(maxsize=128)
def _partial_cached_event_message(event):
    """Cache and serialize the event to json.

    The message is constructed without the id which appended
    in cached_event_message.
    """
    return _message_to_json_bytes_or_none({'type': 'event', 'event': event.json_fragment}) or INVALID_JSON_PARTIAL_MESSAGE

def cached_state_diff_message(message_id_as_bytes, event):
    """Return an event message.

    Serialize to json once per message.

    Since we can have many clients connected that are
    all getting many of the same events (mostly state changed)
    we can avoid serializing the same data for each connection.
    """
    return b''.join((_partial_cached_state_diff_message(event)[:-1], b',"id":', message_id_as_bytes, b'}'))

@lru_cache(maxsize=128)
def _partial_cached_state_diff_message(event):
    """Cache and serialize the event to json.

    The message is constructed without the id which
    will be appended in cached_state_diff_message
    """
    return _message_to_json_bytes_or_none({'type': 'event', 'event': _state_diff_event(event)}) or INVALID_JSON_PARTIAL_MESSAGE

def _state_diff_event(event):
    """Convert a state_changed event to the minimal version.

    State update example

    {
        "a": {entity_id: compressed_state,…}
        "c": {entity_id: diff,…}
        "r": [entity_id,…]
    }
    """
    if (new_state := event.data['new_state']) is None:
        return {ENTITY_EVENT_REMOVE: [event.data['entity_id']]}
    if (old_state := event.data['old_state']) is None:
        return {ENTITY_EVENT_ADD: {new_state.entity_id: new_state.as_compressed_state}}
    additions = {}
    diff = {STATE_DIFF_ADDITIONS: additions}
    new_state_context = new_state.context
    old_state_context = old_state.context
    if old_state.state != new_state.state:
        additions[COMPRESSED_STATE_STATE] = new_state.state
    if old_state.last_changed != new_state.last_changed:
        additions[COMPRESSED_STATE_LAST_CHANGED] = new_state.last_changed_timestamp
    elif old_state.last_updated != new_state.last_updated:
        additions[COMPRESSED_STATE_LAST_UPDATED] = new_state.last_updated_timestamp
    if old_state_context.parent_id != new_state_context.parent_id:
        additions[COMPRESSED_STATE_CONTEXT] = {'parent_id': new_state_context.parent_id}
    if old_state_context.user_id != new_state_context.user_id:
        if COMPRESSED_STATE_CONTEXT in additions:
            additions[COMPRESSED_STATE_CONTEXT]['user_id'] = new_state_context.user_id
        else:
            additions[COMPRESSED_STATE_CONTEXT] = {'user_id': new_state_context.user_id}
    if old_state_context.id != new_state_context.id:
        if COMPRESSED_STATE_CONTEXT in additions:
            additions[COMPRESSED_STATE_CONTEXT]['id'] = new_state_context.id
        else:
            additions[COMPRESSED_STATE_CONTEXT] = new_state_context.id
    if (old_attributes := old_state.attributes) != (new_attributes := new_state.attributes):
        if (added := {key: value for key, value in new_attributes.items() if key not in old_attributes or old_attributes[key] != value}):
            additions[COMPRESSED_STATE_ATTRIBUTES] = added
        if (removed := (old_attributes.keys() - new_attributes)):
            diff[STATE_DIFF_REMOVALS] = {COMPRESSED_STATE_ATTRIBUTES: list(removed)}
    return {ENTITY_EVENT_CHANGE: {new_state.entity_id: diff}}

def _message_to_json_bytes_or_none(message):
    """Serialize a websocket message to json or return None."""
    try:
        return json_bytes(message)
    except (ValueError, TypeError):
        _LOGGER.error('Unable to serialize to JSON. Bad data found at %s', format_unserializable_data(find_paths_unserializable_data(message, dump=JSON_DUMP)))
    return None

def message_to_json_bytes(message):
    """Serialize a websocket message to json or return an error."""
    return _message_to_json_bytes_or_none(message) or json_bytes(error_message(message['id'], const.ERR_UNKNOWN_ERROR, 'Invalid JSON in response'))