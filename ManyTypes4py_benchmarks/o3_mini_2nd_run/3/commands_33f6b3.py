#!/usr/bin/env python3
"""Commands part of Websocket API."""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
import json
import logging
from typing import Any, cast, Dict, List, Optional

import voluptuous as vol

from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ
from homeassistant.auth.permissions.events import SUBSCRIBE_ALLOWLIST
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL, SIGNAL_BOOTSTRAP_INTEGRATIONS
from homeassistant.core import (
    Context,
    Event,
    HomeAssistant,
    ServiceResponse,
    State,
    callback,
)
from homeassistant.exceptions import (
    HomeAssistantError,
    ServiceNotFound,
    ServiceValidationError,
    TemplateError,
    Unauthorized,
)
from homeassistant.helpers import config_validation as cv, entity, template
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import (
    INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA,
    convert_include_exclude_filter,
)
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, async_track_template_result
from homeassistant.helpers.json import JSON_DUMP, ExtendedJSONEncoder, find_paths_unserializable_data, json_bytes, json_fragment
from homeassistant.helpers.service import async_get_all_descriptions
from homeassistant.loader import IntegrationNotFound, async_get_integration, async_get_integration_descriptions, async_get_integrations
from homeassistant.setup import async_get_loaded_integrations, async_get_setup_timings
from homeassistant.util.json import format_unserializable_data

from . import const, decorators, messages
from .connection import ActiveConnection
from .messages import construct_result_message

ALL_SERVICE_DESCRIPTIONS_JSON_CACHE: str = 'websocket_api_all_service_descriptions_json'
_LOGGER: logging.Logger = logging.getLogger(__name__)


@callback
def async_register_commands(
    hass: HomeAssistant,
    async_reg: Callable[[HomeAssistant, Callable[[HomeAssistant, ActiveConnection, Dict[str, Any]], Any]], None]
) -> None:
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


def pong_message(iden: Any) -> Dict[str, Any]:
    """Return a pong message."""
    return {'id': iden, 'type': 'pong'}


@callback
def _forward_events_check_permissions(
    send_message: Callable[[Any], None],
    user: User,
    message_id_as_bytes: bytes,
    event: Event,
) -> None:
    """Forward state changed events to websocket."""
    permissions = user.permissions
    if (
        not user.is_admin
        and (not permissions.access_all_entities(POLICY_READ))
        and (not permissions.check_entity(event.data['entity_id'], POLICY_READ))
    ):
        return
    send_message(messages.cached_event_message(message_id_as_bytes, event))


@callback
def _forward_events_unconditional(
    send_message: Callable[[Any], None],
    message_id_as_bytes: bytes,
    event: Event,
) -> None:
    """Forward events to websocket."""
    send_message(messages.cached_event_message(message_id_as_bytes, event))


@callback
@decorators.websocket_command({
    vol.Required('type'): 'subscribe_events',
    vol.Optional('event_type', default=MATCH_ALL): str,
})
def handle_subscribe_events(
    hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]
) -> None:
    """Handle subscribe events command."""
    event_type: str = msg['event_type']
    if event_type not in SUBSCRIBE_ALLOWLIST and (not connection.user.is_admin):
        _LOGGER.error('Refusing to allow %s to subscribe to event %s', connection.user.name, event_type)
        raise Unauthorized(user_id=connection.user.id)
    message_id_as_bytes: bytes = str(msg['id']).encode()
    if event_type == EVENT_STATE_CHANGED:
        forward_events = partial(_forward_events_check_permissions, connection.send_message, connection.user, message_id_as_bytes)
    else:
        forward_events = partial(_forward_events_unconditional, connection.send_message, message_id_as_bytes)
    connection.subscriptions[msg['id']] = hass.bus.async_listen(event_type, forward_events)
    connection.send_result(msg['id'])


@callback
@decorators.websocket_command({vol.Required('type'): 'subscribe_bootstrap_integrations'})
def handle_subscribe_bootstrap_integrations(
    hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]
) -> None:
    """Handle subscribe bootstrap integrations command."""

    @callback
    def forward_bootstrap_integrations(message: Any) -> None:
        """Forward bootstrap integrations to websocket."""
        connection.send_message(messages.event_message(msg['id'], message))

    connection.subscriptions[msg['id']] = async_dispatcher_connect(hass, SIGNAL_BOOTSTRAP_INTEGRATIONS, forward_bootstrap_integrations)
    connection.send_result(msg['id'])


@callback
@decorators.websocket_command({
    vol.Required('type'): 'unsubscribe_events',
    vol.Required('subscription'): cv.positive_int,
})
def handle_unsubscribe_events(
    hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]
) -> None:
    """Handle unsubscribe events command."""
    subscription = msg['subscription']
    if subscription in connection.subscriptions:
        connection.subscriptions.pop(subscription)()
        connection.send_result(msg['id'])
    else:
        connection.send_error(msg['id'], const.ERR_NOT_FOUND, 'Subscription not found.')


@decorators.websocket_command({
    vol.Required('type'): 'call_service',
    vol.Required('domain'): str,
    vol.Required('service'): str,
    vol.Optional('target'): cv.ENTITY_SERVICE_FIELDS,
    vol.Optional('service_data'): dict,
    vol.Optional('return_response', default=False): bool,
})
@decorators.async_response
async def handle_call_service(
    hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]
) -> None:
    """Handle call service command."""
    target: Optional[Dict[str, Any]] = msg.get('target')
    if template.is_complex(target):
        raise vol.Invalid('Templates are not supported here')
    try:
        context: Context = connection.context(msg)
        response: Optional[ServiceResponse] = await hass.services.async_call(
            domain=msg['domain'],
            service=msg['service'],
            service_data=msg.get('service_data'),
            blocking=True,
            context=context,
            target=target,
            return_response=msg['return_response'],
        )
        result: Dict[str, Any] = {'context': context}
        if msg['return_response']:
            result['response'] = response
        connection.send_result(msg['id'], result)
    except ServiceNotFound as err:
        if err.domain == msg['domain'] and err.service == msg['service']:
            connection.send_error(
                msg['id'],
                const.ERR_NOT_FOUND,
                f"Service {err.domain}.{err.service} not found.",
                translation_domain=err.translation_domain,
                translation_key=err.translation_key,
                translation_placeholders=err.translation_placeholders,
            )
        else:
            connection.send_error(
                msg['id'],
                const.ERR_HOME_ASSISTANT_ERROR,
                f"Service {err.domain}.{err.service} called service {msg['domain']}.{msg['service']} which was not found.",
                translation_domain=const.DOMAIN,
                translation_key='child_service_not_found',
                translation_placeholders={
                    'domain': msg['domain'],
                    'service': msg['service'],
                    'child_domain': err.domain,
                    'child_service': err.service,
                },
            )
    except vol.Invalid as err:
        connection.send_error(msg['id'], const.ERR_INVALID_FORMAT, str(err))
    except ServiceValidationError as err:
        connection.logger.error(err)
        connection.logger.debug('', exc_info=err)
        connection.send_error(
            msg['id'],
            const.ERR_SERVICE_VALIDATION_ERROR,
            f"Validation error: {err}",
            translation_domain=err.translation_domain,
            translation_key=err.translation_key,
            translation_placeholders=err.translation_placeholders,
        )
    except HomeAssistantError as err:
        connection.logger.exception('Unexpected exception')
        connection.send_error(
            msg['id'],
            const.ERR_HOME_ASSISTANT_ERROR,
            str(err),
            translation_domain=err.translation_domain,
            translation_key=err.translation_key,
            translation_placeholders=err.translation_placeholders,
        )
    except Exception as err:
        connection.logger.exception('Unexpected exception')
        connection.send_error(msg['id'], const.ERR_UNKNOWN_ERROR, str(err))


@callback
def _async_get_allowed_states(hass: HomeAssistant, connection: ActiveConnection) -> List[State]:
    user: User = connection.user
    if user.is_admin or user.permissions.access_all_entities(POLICY_READ):
        return hass.states.async_all()
    entity_perm = connection.user.permissions.check_entity
    return [state for state in hass.states.async_all() if entity_perm(state.entity_id, POLICY_READ)]


@callback
@decorators.websocket_command({vol.Required('type'): 'get_states'})
def handle_get_states(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get states command."""
    states: List[State] = _async_get_allowed_states(hass, connection)
    try:
        serialized_states: List[bytes] = [state.as_dict_json for state in states]
    except (ValueError, TypeError):
        pass
    else:
        _send_handle_get_states_response(connection, msg['id'], serialized_states)
        return
    serialized_states = []  # type: List[bytes]
    for state in states:
        try:
            serialized_states.append(state.as_dict_json)
        except (ValueError, TypeError):
            connection.logger.error(
                'Unable to serialize to JSON. Bad data found at %s',
                format_unserializable_data(find_paths_unserializable_data(state, dump=JSON_DUMP)),
            )
    _send_handle_get_states_response(connection, msg['id'], serialized_states)


def _send_handle_get_states_response(connection: ActiveConnection, msg_id: Any, serialized_states: List[bytes]) -> None:
    """Send handle get states response."""
    connection.send_message(construct_result_message(msg_id, b''.join((b'[', b','.join(serialized_states), b']'))))


@callback
def _forward_entity_changes(
    send_message: Callable[[Any], None],
    entity_ids: Optional[set[str]],
    entity_filter: Optional[Callable[[str], bool]],
    user: User,
    message_id_as_bytes: bytes,
    event: Event,
) -> None:
    """Forward entity state changed events to websocket."""
    entity_id: str = event.data['entity_id']
    if entity_ids and entity_id not in entity_ids or (entity_filter and (not entity_filter(entity_id))):
        return
    permissions = user.permissions
    if (
        not user.is_admin
        and (not permissions.access_all_entities(POLICY_READ))
        and (not permissions.check_entity(entity_id, POLICY_READ))
    ):
        return
    send_message(messages.cached_state_diff_message(message_id_as_bytes, event))


@callback
@decorators.websocket_command({
    vol.Required('type'): 'subscribe_entities',
    vol.Optional('entity_ids'): cv.entity_ids,
    **INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA.schema
})
def handle_subscribe_entities(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle subscribe entities command."""
    entity_ids: Optional[set[str]] = set(msg.get('entity_ids', [])) or None
    _filter = convert_include_exclude_filter(msg)
    entity_filter: Optional[Callable[[str], bool]] = None if _filter.empty_filter else _filter.get_filter()
    states: List[State] = _async_get_allowed_states(hass, connection)
    msg_id = msg['id']
    message_id_as_bytes: bytes = str(msg_id).encode()
    connection.subscriptions[msg_id] = hass.bus.async_listen(
        EVENT_STATE_CHANGED,
        partial(_forward_entity_changes, connection.send_message, entity_ids, entity_filter, connection.user, message_id_as_bytes),
    )
    connection.send_result(msg_id)
    try:
        if entity_ids or entity_filter:
            serialized_states: List[bytes] = [
                state.as_compressed_state_json
                for state in states
                if (not entity_ids or state.entity_id in entity_ids)
                and (not entity_filter or entity_filter(state.entity_id))
            ]
        else:
            serialized_states = [state.as_compressed_state_json for state in states]
    except (ValueError, TypeError):
        pass
    else:
        _send_handle_entities_init_response(connection, message_id_as_bytes, serialized_states)
        return
    serialized_states = []  # type: List[bytes]
    for state in states:
        try:
            serialized_states.append(state.as_compressed_state_json)
        except (ValueError, TypeError):
            connection.logger.error(
                'Unable to serialize to JSON. Bad data found at %s',
                format_unserializable_data(find_paths_unserializable_data(state, dump=JSON_DUMP)),
            )
    _send_handle_entities_init_response(connection, message_id_as_bytes, serialized_states)


def _send_handle_entities_init_response(connection: ActiveConnection, message_id_as_bytes: bytes, serialized_states: List[bytes]) -> None:
    """Send handle entities init response."""
    connection.send_message(b''.join((b'{"id":', message_id_as_bytes, b',"type":"event","event":{"a":{', b','.join(serialized_states), b'}}}')))


async def _async_get_all_descriptions_json(hass: HomeAssistant) -> bytes:
    """Return JSON of descriptions (i.e. user documentation) for all service calls."""
    descriptions: Any = await async_get_all_descriptions(hass)
    if ALL_SERVICE_DESCRIPTIONS_JSON_CACHE in hass.data:
        cached_descriptions, cached_json_payload = hass.data[ALL_SERVICE_DESCRIPTIONS_JSON_CACHE]
        if cached_descriptions is descriptions:
            return cast(bytes, cached_json_payload)
    json_payload: bytes = json_bytes(descriptions)
    hass.data[ALL_SERVICE_DESCRIPTIONS_JSON_CACHE] = (descriptions, json_payload)
    return json_payload


@decorators.websocket_command({vol.Required('type'): 'get_services'})
@decorators.async_response
async def handle_get_services(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get services command."""
    payload: bytes = await _async_get_all_descriptions_json(hass)
    connection.send_message(construct_result_message(msg['id'], payload))


@callback
@decorators.websocket_command({vol.Required('type'): 'get_config'})
def handle_get_config(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get config command."""
    connection.send_result(msg['id'], hass.config.as_dict())


@decorators.websocket_command({
    vol.Required('type'): 'manifest/list',
    vol.Optional('integrations'): [str],
})
@decorators.async_response
async def handle_manifest_list(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle integrations command."""
    ints_or_excs: Dict[str, Any] = await async_get_integrations(
        hass, msg.get('integrations') or async_get_loaded_integrations(hass)
    )
    manifest_json_fragments: List[Any] = []
    for int_or_exc in ints_or_excs.values():
        if isinstance(int_or_exc, Exception):
            raise int_or_exc
        manifest_json_fragments.append(int_or_exc.manifest_json_fragment)
    connection.send_result(msg['id'], manifest_json_fragments)


@decorators.websocket_command({
    vol.Required('type'): 'manifest/get',
    vol.Required('integration'): str,
})
@decorators.async_response
async def handle_manifest_get(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle integrations command."""
    try:
        integration = await async_get_integration(hass, msg['integration'])
    except IntegrationNotFound:
        connection.send_error(msg['id'], const.ERR_NOT_FOUND, 'Integration not found')
    else:
        connection.send_result(msg['id'], integration.manifest_json_fragment)


@callback
@decorators.websocket_command({vol.Required('type'): 'integration/setup_info'})
def handle_integration_setup_info(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle integrations command."""
    setup_timings = async_get_setup_timings(hass)
    result: List[Dict[str, Any]] = [{'domain': integration, 'seconds': seconds} for integration, seconds in setup_timings.items()]
    connection.send_result(msg['id'], result)


@callback
@decorators.websocket_command({vol.Required('type'): 'ping'})
def handle_ping(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle ping command."""
    connection.send_message(pong_message(msg['id']))


@lru_cache
def _cached_template(template_str: str, hass: HomeAssistant) -> template.Template:
    """Return a cached template."""
    return template.Template(template_str, hass)


@decorators.websocket_command({
    vol.Required('type'): 'render_template',
    vol.Required('template'): str,
    vol.Optional('entity_ids'): cv.entity_ids,
    vol.Optional('variables'): dict,
    vol.Optional('timeout'): vol.Coerce(float),
    vol.Optional('strict', default=False): bool,
    vol.Optional('report_errors', default=False): bool,
})
@decorators.async_response
async def handle_render_template(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle render_template command."""
    template_str: str = msg['template']
    report_errors: bool = msg['report_errors']
    if report_errors:
        template_obj: template.Template = template.Template(template_str, hass)
    else:
        template_obj = _cached_template(template_str, hass)
    variables: Optional[Dict[str, Any]] = msg.get('variables')
    timeout: Optional[float] = msg.get('timeout')

    @callback
    def _error_listener(level: int, template_error: TemplateError) -> None:
        connection.send_message(messages.event_message(msg['id'], {'error': template_error, 'level': logging.getLevelName(level)}))

    @callback
    def _thread_safe_error_listener(level: int, template_error: TemplateError) -> None:
        hass.loop.call_soon_threadsafe(_error_listener, level, template_error)

    if timeout:
        try:
            log_fn: Optional[Callable[[int, TemplateError], None]] = _thread_safe_error_listener if report_errors else None
            timed_out: bool = await template_obj.async_render_will_timeout(timeout, variables, strict=msg['strict'], log_fn=log_fn)
        except TemplateError:
            timed_out = False
        if timed_out:
            connection.send_error(msg['id'], const.ERR_TEMPLATE_ERROR, f'Exceeded maximum execution time of {timeout}s')
            return

    @callback
    def _template_listener(event: Event, updates: List[TrackTemplateResult]) -> None:
        track_template_result: TrackTemplateResult = updates.pop()
        result = track_template_result.result
        if isinstance(result, TemplateError):
            if not report_errors:
                return
            connection.send_message(messages.event_message(msg['id'], {'error': str(result), 'level': 'ERROR'}))
            return
        connection.send_message(messages.event_message(msg['id'], {'result': result, 'listeners': info.listeners}))

    try:
        log_fn = _error_listener if report_errors else None
        info = async_track_template_result(hass, [TrackTemplate(template_obj, variables)], _template_listener, strict=msg['strict'], log_fn=log_fn)
    except TemplateError as ex:
        connection.send_error(msg['id'], const.ERR_TEMPLATE_ERROR, str(ex))
        return
    connection.subscriptions[msg['id']] = info.async_remove
    connection.send_result(msg['id'])
    hass.loop.call_soon_threadsafe(info.async_refresh)


def _serialize_entity_sources(entity_infos: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a websocket response from a dict of entity sources."""
    return {entity_id: {'domain': entity_info['domain']} for entity_id, entity_info in entity_infos.items()}


@callback
@decorators.websocket_command({vol.Required('type'): 'entity/source'})
def handle_entity_source(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle entity source command."""
    all_entity_sources: Dict[str, Any] = entity.entity_sources(hass)
    entity_perm = connection.user.permissions.check_entity
    if connection.user.permissions.access_all_entities(POLICY_READ):
        entity_sources = all_entity_sources
    else:
        entity_sources = {entity_id: source for entity_id, source in all_entity_sources.items() if entity_perm(entity_id, POLICY_READ)}
    connection.send_result(msg['id'], _serialize_entity_sources(entity_sources))


@decorators.websocket_command({
    vol.Required('type'): 'subscribe_trigger',
    vol.Required('trigger'): cv.TRIGGER_SCHEMA,
    vol.Optional('variables'): dict,
})
@decorators.require_admin
@decorators.async_response
async def handle_subscribe_trigger(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle subscribe trigger command."""
    from homeassistant.helpers import trigger
    trigger_config: Dict[str, Any] = await trigger.async_validate_trigger_config(hass, msg['trigger'])

    @callback
    def forward_triggers(variables: Dict[str, Any], context: Optional[Any] = None) -> None:
        """Forward events to websocket."""
        message: Dict[str, Any] = messages.event_message(msg['id'], {'variables': variables, 'context': context})
        connection.send_message(json.dumps(message, cls=ExtendedJSONEncoder, allow_nan=False, separators=(',', ':')))

    subscription = await trigger.async_initialize_triggers(
        hass,
        trigger_config,
        forward_triggers,
        const.DOMAIN,
        const.DOMAIN,
        connection.logger.log,
        variables=msg.get('variables'),
    )
    connection.subscriptions[msg['id']] = subscription or (lambda: None)
    connection.send_result(msg['id'])


@decorators.websocket_command({
    vol.Required('type'): 'test_condition',
    vol.Required('condition'): cv.CONDITION_SCHEMA,
    vol.Optional('variables'): dict,
})
@decorators.require_admin
@decorators.async_response
async def handle_test_condition(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle test condition command."""
    from homeassistant.helpers import condition
    config: Dict[str, Any] = await condition.async_validate_condition_config(hass, msg['condition'])
    check_condition = await condition.async_from_config(hass, config)
    connection.send_result(msg['id'], {'result': check_condition(hass, msg.get('variables'))})


@decorators.websocket_command({
    vol.Required('type'): 'execute_script',
    vol.Required('sequence'): cv.SCRIPT_SCHEMA,
    vol.Optional('variables'): dict,
})
@decorators.require_admin
@decorators.async_response
async def handle_execute_script(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle execute script command."""
    from homeassistant.helpers.script import Script, async_validate_actions_config
    script_config: Dict[str, Any] = await async_validate_actions_config(hass, msg['sequence'])
    context: Context = connection.context(msg)
    script_obj: Script = Script(hass, script_config, f'{const.DOMAIN} script', const.DOMAIN)
    try:
        script_result = await script_obj.async_run(msg.get('variables'), context=context)
    except ServiceValidationError as err:
        connection.logger.error(err)
        connection.logger.debug('', exc_info=err)
        connection.send_error(
            msg['id'],
            const.ERR_SERVICE_VALIDATION_ERROR,
            str(err),
            translation_domain=err.translation_domain,
            translation_key=err.translation_key,
            translation_placeholders=err.translation_placeholders,
        )
        return
    connection.send_result(msg['id'], {'context': context, 'response': script_result.service_response if script_result else None})


@callback
@decorators.websocket_command({
    vol.Required('type'): 'fire_event',
    vol.Required('event_type'): str,
    vol.Optional('event_data'): dict,
})
@decorators.require_admin
def handle_fire_event(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle fire event command."""
    context: Context = connection.context(msg)
    hass.bus.async_fire(msg['event_type'], msg.get('event_data'), context=context)
    connection.send_result(msg['id'], {'context': context})


@decorators.websocket_command({
    vol.Required('type'): 'validate_config',
    vol.Optional('triggers'): cv.match_all,
    vol.Optional('conditions'): cv.match_all,
    vol.Optional('actions'): cv.match_all,
})
@decorators.async_response
async def handle_validate_config(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle validate config command."""
    from homeassistant.helpers import condition, script, trigger
    result: Dict[str, Any] = {}
    for key, schema, validator in (
        ('triggers', cv.TRIGGER_SCHEMA, trigger.async_validate_trigger_config),
        ('conditions', cv.CONDITIONS_SCHEMA, condition.async_validate_conditions_config),
        ('actions', cv.SCRIPT_SCHEMA, script.async_validate_actions_config),
    ):
        if key not in msg:
            continue
        try:
            await validator(hass, schema(msg[key]))
        except (vol.Invalid, HomeAssistantError) as err:
            result[key] = {'valid': False, 'error': str(err)}
        else:
            result[key] = {'valid': True, 'error': None}
    connection.send_result(msg['id'], result)


@callback
@decorators.websocket_command({
    vol.Required('type'): 'supported_features',
    vol.Required('features'): {str: int},
})
def handle_supported_features(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle setting supported features."""
    connection.set_supported_features(msg['features'])
    connection.send_result(msg['id'])


@decorators.require_admin
@decorators.websocket_command({'type': 'integration/descriptions'})
@decorators.async_response
async def handle_integration_descriptions(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Get metadata for all brands and integrations."""
    integration_descriptions = await async_get_integration_descriptions(hass)
    connection.send_result(msg['id'], integration_descriptions)