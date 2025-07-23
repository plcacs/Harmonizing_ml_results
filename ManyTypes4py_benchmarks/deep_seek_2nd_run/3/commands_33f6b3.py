"""Commands part of Websocket API."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache, partial
import json
import logging
from typing import Any, cast, Optional, Union, Dict, List, Set, Tuple, TypedDict

import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ
from homeassistant.auth.permissions.events import SUBSCRIBE_ALLOWLIST
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL, SIGNAL_BOOTSTRAP_INTEGRATIONS
from homeassistant.core import Context, Event, EventStateChangedData, HomeAssistant, ServiceResponse, State, callback
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, ServiceValidationError, TemplateError, Unauthorized
from homeassistant.helpers import config_validation as cv, entity, template
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA, EntityFilter, convert_include_exclude_filter
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, async_track_template_result, TemplateListenerInfo
from homeassistant.helpers.json import JSON_DUMP, ExtendedJSONEncoder, find_paths_unserializable_data, json_bytes, json_fragment
from homeassistant.helpers.service import async_get_all_descriptions
from homeassistant.loader import IntegrationNotFound, async_get_integration, async_get_integration_descriptions, async_get_integrations
from homeassistant.setup import async_get_loaded_integrations, async_get_setup_timings
from homeassistant.util.json import format_unserializable_data
from . import const, decorators, messages
from .connection import ActiveConnection
from .messages import construct_result_message

ALL_SERVICE_DESCRIPTIONS_JSON_CACHE = 'websocket_api_all_service_descriptions_json'
_LOGGER = logging.getLogger(__name__)

class PongMessage(TypedDict):
    id: int
    type: str

@callback
def async_register_commands(hass: HomeAssistant, async_reg: Callable[[HomeAssistant, Callable], None]) -> None:
    """Register commands."""
    async_reg(hass, handle_call_service)
    async_reg(hass, handle_entity_source)
    async_reg(hass, handle_execute_script)
    async_reg(hass, handle_fire_event)
    async_reg(hass, handle_get_config)
    async_reg(hass, handle_get_services)
    async_reg(hass, handle_get_states)
    async_reg(hass, handle_manifest_get)
    async_reg(hass, handle_integration_setup_info)
    async_reg(hass, handle_manifest_list)
    async_reg(hass, handle_ping)
    async_reg(hass, handle_render_template)
    async_reg(hass, handle_subscribe_bootstrap_integrations)
    async_reg(hass, handle_subscribe_events)
    async_reg(hass, handle_subscribe_trigger)
    async_reg(hass, handle_test_condition)
    async_reg(hass, handle_unsubscribe_events)
    async_reg(hass, handle_validate_config)
    async_reg(hass, handle_subscribe_entities)
    async_reg(hass, handle_supported_features)
    async_reg(hass, handle_integration_descriptions)

def pong_message(iden: int) -> PongMessage:
    """Return a pong message."""
    return {'id': iden, 'type': 'pong'}

@callback
def _forward_events_check_permissions(
    send_message: Callable[[bytes], None],
    user: User,
    message_id_as_bytes: bytes,
    event: Event[EventStateChangedData]
) -> None:
    """Forward state changed events to websocket."""
    permissions = user.permissions
    if not user.is_admin and (not permissions.access_all_entities(POLICY_READ)) and (not permissions.check_entity(event.data['entity_id'], POLICY_READ)):
        return
    send_message(messages.cached_event_message(message_id_as_bytes, event))

@callback
def _forward_events_unconditional(
    send_message: Callable[[bytes], None],
    message_id_as_bytes: bytes,
    event: Event[Any]
) -> None:
    """Forward events to websocket."""
    send_message(messages.cached_event_message(message_id_as_bytes, event))

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_events', vol.Optional('event_type', default=MATCH_ALL): str})
def handle_subscribe_events(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle subscribe events command."""
    event_type = msg['event_type']
    if event_type not in SUBSCRIBE_ALLOWLIST and (not connection.user.is_admin):
        _LOGGER.error('Refusing to allow %s to subscribe to event %s', connection.user.name, event_type)
        raise Unauthorized(user_id=connection.user.id)
    message_id_as_bytes = str(msg['id']).encode()
    if event_type == EVENT_STATE_CHANGED:
        forward_events = partial(_forward_events_check_permissions, connection.send_message, connection.user, message_id_as_bytes)
    else:
        forward_events = partial(_forward_events_unconditional, connection.send_message, message_id_as_bytes)
    connection.subscriptions[msg['id']] = hass.bus.async_listen(event_type, forward_events)
    connection.send_result(msg['id'])

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_bootstrap_integrations'})
def handle_subscribe_bootstrap_integrations(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle subscribe bootstrap integrations command."""

    @callback
    def forward_bootstrap_integrations(message: Any) -> None:
        """Forward bootstrap integrations to websocket."""
        connection.send_message(messages.event_message(msg['id'], message))
    connection.subscriptions[msg['id']] = async_dispatcher_connect(hass, SIGNAL_BOOTSTRAP_INTEGRATIONS, forward_bootstrap_integrations)
    connection.send_result(msg['id'])

@callback
@decorators.websocket_command({vol.Required('type'): 'unsubscribe_events', vol.Required('subscription'): cv.positive_int})
def handle_unsubscribe_events(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle unsubscribe events command."""
    subscription = msg['subscription']
    if subscription in connection.subscriptions:
        connection.subscriptions.pop(subscription)()
        connection.send_result(msg['id'])
    else:
        connection.send_error(msg['id'], const.ERR_NOT_FOUND, 'Subscription not found.')

@decorators.websocket_command({vol.Required('type'): 'call_service', vol.Required('domain'): str, vol.Required('service'): str, vol.Optional('target'): cv.ENTITY_SERVICE_FIELDS, vol.Optional('service_data'): dict, vol.Optional('return_response', default=False): bool})
@decorators.async_response
async def handle_call_service(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle call service command."""
    target = msg.get('target')
    if template.is_complex(target):
        raise vol.Invalid('Templates are not supported here')
    try:
        context = connection.context(msg)
        response = await hass.services.async_call(domain=msg['domain'], service=msg['service'], service_data=msg.get('service_data'), blocking=True, context=context, target=target, return_response=msg['return_response'])
        result = {'context': context}
        if msg['return_response']:
            result['response'] = response
        connection.send_result(msg['id'], result)
    except ServiceNotFound as err:
        if err.domain == msg['domain'] and err.service == msg['service']:
            connection.send_error(msg['id'], const.ERR_NOT_FOUND, f'Service {err.domain}.{err.service} not found.', translation_domain=err.translation_domain, translation_key=err.translation_key, translation_placeholders=err.translation_placeholders)
        else:
            connection.send_error(msg['id'], const.ERR_HOME_ASSISTANT_ERROR, f'Service {err.domain}.{err.service} called service {msg['domain']}.{msg['service']} which was not found.', translation_domain=const.DOMAIN, translation_key='child_service_not_found', translation_placeholders={'domain': msg['domain'], 'service': msg['service'], 'child_domain': err.domain, 'child_service': err.service})
    except vol.Invalid as err:
        connection.send_error(msg['id'], const.ERR_INVALID_FORMAT, str(err))
    except ServiceValidationError as err:
        connection.logger.error(err)
        connection.logger.debug('', exc_info=err)
        connection.send_error(msg['id'], const.ERR_SERVICE_VALIDATION_ERROR, f'Validation error: {err}', translation_domain=err.translation_domain, translation_key=err.translation_key, translation_placeholders=err.translation_placeholders)
    except HomeAssistantError as err:
        connection.logger.exception('Unexpected exception')
        connection.send_error(msg['id'], const.ERR_HOME_ASSISTANT_ERROR, str(err), translation_domain=err.translation_domain, translation_key=err.translation_key, translation_placeholders=err.translation_placeholders)
    except Exception as err:
        connection.logger.exception('Unexpected exception')
        connection.send_error(msg['id'], const.ERR_UNKNOWN_ERROR, str(err))

@callback
def _async_get_allowed_states(hass: HomeAssistant, connection: ActiveConnection) -> List[State]:
    user = connection.user
    if user.is_admin or user.permissions.access_all_entities(POLICY_READ):
        return hass.states.async_all()
    entity_perm = connection.user.permissions.check_entity
    return [state for state in hass.states.async_all() if entity_perm(state.entity_id, POLICY_READ)]

@callback
@decorators.websocket_command({vol.Required('type'): 'get_states'})
def handle_get_states(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle get states command."""
    states = _async_get_allowed_states(hass, connection)
    try:
        serialized_states = [state.as_dict_json for state in states]
    except (ValueError, TypeError):
        pass
    else:
        _send_handle_get_states_response(connection, msg['id'], serialized_states)
        return
    serialized_states = []
    for state in states:
        try:
            serialized_states.append(state.as_dict_json)
        except (ValueError, TypeError):
            connection.logger.error('Unable to serialize to JSON. Bad data found at %s', format_unserializable_data(find_paths_unserializable_data(state, dump=JSON_DUMP)))
    _send_handle_get_states_response(connection, msg['id'], serialized_states)

def _send_handle_get_states_response(connection: ActiveConnection, msg_id: int, serialized_states: List[bytes]) -> None:
    """Send handle get states response."""
    connection.send_message(construct_result_message(msg_id, b''.join((b'[', b','.join(serialized_states), b']'))))

@callback
def _forward_entity_changes(
    send_message: Callable[[bytes], None],
    entity_ids: Optional[Set[str]],
    entity_filter: Optional[EntityFilter],
    user: User,
    message_id_as_bytes: bytes,
    event: Event[EventStateChangedData]
) -> None:
    """Forward entity state changed events to websocket."""
    entity_id = event.data['entity_id']
    if entity_ids and entity_id not in entity_ids or (entity_filter and (not entity_filter(entity_id))):
        return
    permissions = user.permissions
    if not user.is_admin and (not permissions.access_all_entities(POLICY_READ)) and (not permissions.check_entity(entity_id, POLICY_READ)):
        return
    send_message(messages.cached_state_diff_message(message_id_as_bytes, event))

@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_entities', vol.Optional('entity_ids'): cv.entity_ids, **INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA.schema})
def handle_subscribe_entities(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle subscribe entities command."""
    entity_ids = set(msg.get('entity_ids', [])) or None
    _filter = convert_include_exclude_filter(msg)
    entity_filter = None if _filter.empty_filter else _filter.get_filter()
    states = _async_get_allowed_states(hass, connection)
    msg_id = msg['id']
    message_id_as_bytes = str(msg_id).encode()
    connection.subscriptions[msg_id] = hass.bus.async_listen(EVENT_STATE_CHANGED, partial(_forward_entity_changes, connection.send_message, entity_ids, entity_filter, connection.user, message_id_as_bytes))
    connection.send_result(msg_id)
    try:
        if entity_ids or entity_filter:
            serialized_states = [state.as_compressed_state_json for state in states if (not entity_ids or state.entity_id in entity_ids) and (not entity_filter or entity_filter(state.entity_id))]
        else:
            serialized_states = [state.as_compressed_state_json for state in states]
    except (ValueError, TypeError):
        pass
    else:
        _send_handle_entities_init_response(connection, message_id_as_bytes, serialized_states)
        return
    serialized_states = []
    for state in states:
        try:
            serialized_states.append(state.as_compressed_state_json)
        except (ValueError, TypeError):
            connection.logger.error('Unable to serialize to JSON. Bad data found at %s', format_unserializable_data(find_paths_unserializable_data(state, dump=JSON_DUMP)))
    _send_handle_entities_init_response(connection, message_id_as_bytes, serialized_states)

def _send_handle_entities_init_response(connection: ActiveConnection, message_id_as_bytes: bytes, serialized_states: List[bytes]) -> None:
    """Send handle entities init response."""
    connection.send_message(b''.join((b'{"id":', message_id_as_bytes, b',"type":"event","event":{"a":{', b','.join(serialized_states), b'}}}')))

async def _async_get_all_descriptions_json(hass: HomeAssistant) -> bytes:
    """Return JSON of descriptions (i.e. user documentation) for all service calls."""
    descriptions = await async_get_all_descriptions(hass)
    if ALL_SERVICE_DESCRIPTIONS_JSON_CACHE in hass.data:
        cached_descriptions, cached_json_payload = hass.data[ALL_SERVICE_DESCRIPTIONS_JSON_CACHE]
        if cached_descriptions is descriptions:
            return cast(bytes, cached_json_payload)
    json_payload = json_bytes(descriptions)
    hass.data[ALL_SERVICE_DESCRIPTIONS_JSON_CACHE] = (descriptions, json_payload)
    return json_payload

@decorators.websocket_command({vol.Required('type'): 'get_services'})
@decorators.async_response
async def handle_get_services(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle get services command."""
    payload = await _async_get_all_descriptions_json(hass)
    connection.send_message(construct_result_message(msg['id'], payload))

@callback
@decorators.websocket_command({vol.Required('type'): 'get_config'})
def handle_get_config(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle get config command."""
    connection.send_result(msg['id'], hass.config.as_dict())

@decorators.websocket_command({vol.Required('type'): 'manifest/list', vol.Optional('integrations'): [str]})
@decorators.async_response
async def handle_manifest_list(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle integrations command."""
    ints_or_excs = await async_get_integrations(hass, msg.get('integrations') or async_get_loaded_integrations(hass))
    manifest_json_fragments = []
    for int_or_exc in ints_or_excs.values():
        if isinstance(int_or_exc, Exception):
            raise int_or_exc
        manifest_json_fragments.append(int_or_exc.manifest_json_fragment)
    connection.send_result(msg['id'], manifest_json_fragments)

@decorators.websocket_command({vol.Required('type'): 'manifest/get', vol.Required('integration'): str})
@decorators.async_response
async def handle_manifest_get(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle integrations command."""
    try:
        integration = await async_get_integration(hass, msg['integration'])
    except IntegrationNotFound:
        connection.send_error(msg['id'], const.ERR_NOT_FOUND, 'Integration not found')
    else:
        connection.send_result(msg['id'], integration.manifest_json_fragment)

@callback
@decorators.websocket_command({vol.Required('type'): 'integration/setup_info'})
def handle_integration_setup_info(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle integrations command."""
    connection.send_result(msg['id'], [{'domain': integration, 'seconds': seconds} for integration, seconds in async_get_setup_timings(hass).items()])

@callback
@decorators.websocket_command({vol.Required('type'): 'ping'})
def handle_ping(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """