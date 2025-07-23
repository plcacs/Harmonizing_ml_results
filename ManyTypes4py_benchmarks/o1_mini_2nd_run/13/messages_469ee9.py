"""Message templates for websocket commands."""
from __future__ import annotations
from functools import lru_cache
import logging
from typing import Any, Final, Optional, Dict, List, Union
import voluptuous as vol
from homeassistant.const import (
    COMPRESSED_STATE_ATTRIBUTES,
    COMPRESSED_STATE_CONTEXT,
    COMPRESSED_STATE_LAST_CHANGED,
    COMPRESSED_STATE_LAST_UPDATED,
    COMPRESSED_STATE_STATE,
)
from homeassistant.core import CompressedState, Event, EventStateChangedData
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.json import (
    JSON_DUMP,
    find_paths_unserializable_data,
    json_bytes,
)
from homeassistant.util.json import format_unserializable_data
from . import const

_LOGGER = logging.getLogger(__name__)

MINIMAL_MESSAGE_SCHEMA: Final = vol.Schema(
    {vol.Required("id"): cv.positive_int, vol.Required("type"): cv.string},
    extra=vol.ALLOW_EXTRA,
)
BASE_COMMAND_MESSAGE_SCHEMA: Final = vol.Schema({vol.Required("id"): cv.positive_int})

STATE_DIFF_ADDITIONS: Final = "+"
STATE_DIFF_REMOVALS: Final = "-"
ENTITY_EVENT_ADD: Final = "a"
ENTITY_EVENT_REMOVE: Final = "r"
ENTITY_EVENT_CHANGE: Final = "c"

BASE_ERROR_MESSAGE: Final[Dict[str, Any]] = {"type": const.TYPE_RESULT, "success": False}
INVALID_JSON_PARTIAL_MESSAGE: Final = json_bytes(
    {
        **BASE_ERROR_MESSAGE,
        "error": {
            "code": const.ERR_UNKNOWN_ERROR,
            "message": "Invalid JSON in response",
        },
    }
)


def result_message(iden: int, result: Optional[Any] = None) -> Dict[str, Any]:
    """Return a success result message."""
    return {"id": iden, "type": const.TYPE_RESULT, "success": True, "result": result}


def construct_result_message(iden: int, payload: bytes) -> bytes:
    """Construct a success result message JSON."""
    return b"".join(
        (
            b'{"id":',
            str(iden).encode(),
            b',"type":"result","success":true,"result":',
            payload,
            b"}",
        )
    )


def error_message(
    iden: int,
    code: Union[str, int],
    message: str,
    translation_key: Optional[str] = None,
    translation_domain: Optional[str] = None,
    translation_placeholders: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an error result message."""
    error_payload: Dict[str, Any] = {"code": code, "message": message}
    if translation_key is not None:
        error_payload["translation_key"] = translation_key
        error_payload["translation_placeholders"] = translation_placeholders
        error_payload["translation_domain"] = translation_domain
    return {"id": iden, **BASE_ERROR_MESSAGE, "error": error_payload}


def event_message(iden: int, event: Event) -> Dict[str, Any]:
    """Return an event message."""
    return {"id": iden, "type": "event", "event": event}


def cached_event_message(message_id_as_bytes: bytes, event: Event) -> bytes:
    """Return an event message.

    Serialize to json once per message.

    Since we can have many clients connected that are
    all getting many of the same events (mostly state changed)
    we can avoid serializing the same data for each connection.
    """
    partial_message: bytes = _partial_cached_event_message(event)
    return b"".join((partial_message[:-1], b',"id":', message_id_as_bytes, b"}"))


@lru_cache(maxsize=128)
def _partial_cached_event_message(event: Event) -> bytes:
    """Cache and serialize the event to json.

    The message is constructed without the id which appended
    in cached_event_message.
    """
    return (
        _message_to_json_bytes_or_none({"type": "event", "event": event.json_fragment})
        or INVALID_JSON_PARTIAL_MESSAGE
    )


def cached_state_diff_message(message_id_as_bytes: bytes, event: Event) -> bytes:
    """Return an event message.

    Serialize to json once per message.

    Since we can have many clients connected that are
    all getting many of the same events (mostly state changed)
    we can avoid serializing the same data for each connection.
    """
    partial_message: bytes = _partial_cached_state_diff_message(event)
    return b"".join((partial_message[:-1], b',"id":', message_id_as_bytes, b"}"))


@lru_cache(maxsize=128)
def _partial_cached_state_diff_message(event: Event) -> bytes:
    """Cache and serialize the event to json.

    The message is constructed without the id which
    will be appended in cached_state_diff_message
    """
    return (
        _message_to_json_bytes_or_none({"type": "event", "event": _state_diff_event(event)})
        or INVALID_JSON_PARTIAL_MESSAGE
    )


def _state_diff_event(event: Event) -> Dict[str, Any]:
    """Convert a state_changed event to the minimal version.

    State update example

    {
        "a": {entity_id: compressed_state,…}
        "c": {entity_id: diff,…}
        "r": [entity_id,…]
    }
    """
    new_state: Optional[CompressedState] = event.data.get("new_state")
    if new_state is None:
        return {ENTITY_EVENT_REMOVE: [event.data["entity_id"]]}
    old_state: Optional[CompressedState] = event.data.get("old_state")
    if old_state is None:
        return {ENTITY_EVENT_ADD: {new_state.entity_id: new_state.as_compressed_state}}

    additions: Dict[str, Any] = {}
    diff: Dict[str, Any] = {STATE_DIFF_ADDITIONS: additions}

    new_state_context = new_state.context
    old_state_context = old_state.context

    if old_state.state != new_state.state:
        additions[COMPRESSED_STATE_STATE] = new_state.state
    if old_state.last_changed != new_state.last_changed:
        additions[COMPRESSED_STATE_LAST_CHANGED] = new_state.last_changed_timestamp
    elif old_state.last_updated != new_state.last_updated:
        additions[COMPRESSED_STATE_LAST_UPDATED] = new_state.last_updated_timestamp
    if old_state_context.parent_id != new_state_context.parent_id:
        additions[COMPRESSED_STATE_CONTEXT] = {"parent_id": new_state_context.parent_id}
    if old_state_context.user_id != new_state_context.user_id:
        if COMPRESSED_STATE_CONTEXT in additions:
            additions[COMPRESSED_STATE_CONTEXT]["user_id"] = new_state_context.user_id
        else:
            additions[COMPRESSED_STATE_CONTEXT] = {"user_id": new_state_context.user_id}
    if old_state_context.id != new_state_context.id:
        if COMPRESSED_STATE_CONTEXT in additions:
            additions[COMPRESSED_STATE_CONTEXT]["id"] = new_state_context.id
        else:
            additions[COMPRESSED_STATE_CONTEXT] = new_state_context.id
    if (old_attributes := old_state.attributes) != (new_attributes := new_state.attributes):
        if added := {
            key: value
            for key, value in new_attributes.items()
            if key not in old_attributes or old_attributes[key] != value
        }:
            additions[COMPRESSED_STATE_ATTRIBUTES] = added
        removed_keys: List[str] = list(old_attributes.keys() - new_attributes)
        if removed_keys:
            diff[STATE_DIFF_REMOVALS] = {COMPRESSED_STATE_ATTRIBUTES: removed_keys}
    return {ENTITY_EVENT_CHANGE: {new_state.entity_id: diff}}


def _message_to_json_bytes_or_none(message: Dict[str, Any]) -> Optional[bytes]:
    """Serialize a websocket message to json or return None."""
    try:
        return json_bytes(message)
    except (ValueError, TypeError):
        _LOGGER.error(
            "Unable to serialize to JSON. Bad data found at %s",
            format_unserializable_data(
                find_paths_unserializable_data(message, dump=JSON_DUMP)
            ),
        )
    return None


def message_to_json_bytes(message: Dict[str, Any]) -> bytes:
    """Serialize a websocket message to json or return an error."""
    serialized: Optional[bytes] = _message_to_json_bytes_or_none(message)
    if serialized is not None:
        return serialized
    return json_bytes(
        error_message(
            cast(int, message.get("id", -1)),
            const.ERR_UNKNOWN_ERROR,
            "Invalid JSON in response",
        )
    )
