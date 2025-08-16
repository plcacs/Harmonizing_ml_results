from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache, partial
import json
import logging
from typing import Any, cast
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ
from homeassistant.auth.permissions.events import SUBSCRIBE_ALLOWLIST
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL, SIGNAL_BOOTSTRAP_INTEGRATIONS
from homeassistant.core import Context, Event, EventStateChangedData, HomeAssistant, ServiceResponse, State, callback
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, ServiceValidationError, TemplateError, Unauthorized
from homeassistant.helpers import config_validation as cv, entity, template
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA, convert_include_exclude_filter
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, async_track_template_result
from homeassistant.helpers.json import JSON_DUMP, ExtendedJSONEncoder, find_paths_unserializable_data, json_bytes, json_fragment
from homeassistant.helpers.service import async_get_all_descriptions
from homeassistant.loader import IntegrationNotFound, async_get_integration, async_get_integration_descriptions, async_get_integrations
from homeassistant.setup import async_get_loaded_integrations, async_get_setup_timings
from homeassistant.util.json import format_unserializable_data
from . import const, decorators, messages
from .connection import ActiveConnection
from .messages import construct_result_message

@callback
def async_register_commands(hass: HomeAssistant, async_reg: Callable[[HomeAssistant, Callable], None]) -> None:
    ...

def pong_message(iden: Any) -> dict:
    ...

@callback
def _forward_events_check_permissions(send_message: Callable, user: User, message_id_as_bytes: bytes, event: Event) -> None:
    ...

@callback
def _forward_events_unconditional(send_message: Callable, message_id_as_bytes: bytes, event: Event) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_events', vol.Optional('event_type', default=MATCH_ALL): str})
def handle_subscribe_events(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_bootstrap_integrations'})
def handle_subscribe_bootstrap_integrations(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'unsubscribe_events', vol.Required('subscription'): cv.positive_int})
def handle_unsubscribe_events(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'call_service', vol.Required('domain'): str, vol.Required('service'): str, vol.Optional('target'): cv.ENTITY_SERVICE_FIELDS, vol.Optional('service_data'): dict, vol.Optional('return_response', default=False): bool})
@decorators.async_response
async def handle_call_service(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
def _async_get_allowed_states(hass: HomeAssistant, connection: ActiveConnection) -> list[State]:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'get_states'})
def handle_get_states(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

def _send_handle_get_states_response(connection: ActiveConnection, msg_id: Any, serialized_states: list[dict]) -> None:
    ...

@callback
def _forward_entity_changes(send_message: Callable, entity_ids: set[str], entity_filter: Callable, user: User, message_id_as_bytes: bytes, event: Event) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_entities', vol.Optional('entity_ids'): cv.entity_ids, **INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA.schema})
def handle_subscribe_entities(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

def _send_handle_entities_init_response(connection: ActiveConnection, message_id_as_bytes: bytes, serialized_states: list[dict]) -> None:
    ...

async def _async_get_all_descriptions_json(hass: HomeAssistant) -> bytes:
    ...

@decorators.websocket_command({vol.Required('type'): 'get_services'})
@decorators.async_response
async def handle_get_services(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'get_config'})
def handle_get_config(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'manifest/list', vol.Optional('integrations'): list[str]})
@decorators.async_response
async def handle_manifest_list(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'manifest/get', vol.Required('integration'): str})
@decorators.async_response
async def handle_manifest_get(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'integration/setup_info'})
def handle_integration_setup_info(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'ping'})
def handle_ping(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@lru_cache
def _cached_template(template_str: str, hass: HomeAssistant) -> template.Template:
    ...

@decorators.websocket_command({vol.Required('type'): 'render_template', vol.Required('template'): str, vol.Optional('entity_ids'): cv.entity_ids, vol.Optional('variables'): dict, vol.Optional('timeout'): vol.Coerce(float), vol.Optional('strict', default=False): bool, vol.Optional('report_errors', default=False): bool})
@decorators.async_response
async def handle_render_template(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
def _serialize_entity_sources(entity_infos: dict[str, dict]) -> dict:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'entity/source'})
def handle_entity_source(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'subscribe_trigger', vol.Required('trigger'): cv.TRIGGER_SCHEMA, vol.Optional('variables'): dict})
@decorators.require_admin
@decorators.async_response
async def handle_subscribe_trigger(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'test_condition', vol.Required('condition'): cv.CONDITION_SCHEMA, vol.Optional('variables'): dict})
@decorators.require_admin
@decorators.async_response
async def handle_test_condition(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'execute_script', vol.Required('sequence'): cv.SCRIPT_SCHEMA, vol.Optional('variables'): dict})
@decorators.require_admin
@decorators.async_response
async def handle_execute_script(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'fire_event', vol.Required('event_type'): str, vol.Optional('event_data'): dict})
@decorators.require_admin
def handle_fire_event(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.websocket_command({vol.Required('type'): 'validate_config', vol.Optional('triggers'): cv.match_all, vol.Optional('conditions'): cv.match_all, vol.Optional('actions'): cv.match_all})
@decorators.async_response
async def handle_validate_config(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@callback
@decorators.websocket_command({vol.Required('type'): 'supported_features', vol.Required('features'): dict[str, int]})
def handle_supported_features(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...

@decorators.require_admin
@decorators.websocket_command({'type': 'integration/descriptions'})
@decorators.async_response
async def handle_integration_descriptions(hass: HomeAssistant, connection: ActiveConnection, msg: dict) -> None:
    ...
