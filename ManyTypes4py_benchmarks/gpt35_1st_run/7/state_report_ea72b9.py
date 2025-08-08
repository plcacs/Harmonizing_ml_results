from __future__ import annotations
from asyncio import timeout
from http import HTTPStatus
import json
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, Dict, List, Optional
from uuid import uuid4
import aiohttp
from homeassistant.components import event
from homeassistant.const import EVENT_STATE_CHANGED, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.significant_change import create_checker
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType, json_loads_object
from .const import API_CHANGE, API_CONTEXT, API_DIRECTIVE, API_ENDPOINT, API_EVENT, API_HEADER, API_PAYLOAD, API_SCOPE, DATE_FORMAT, DOMAIN, Cause
from .diagnostics import async_redact_auth_data
from .entities import ENTITY_ADAPTERS, AlexaEntity
from .errors import AlexaInvalidEndpointError, NoTokenAvailable, RequireRelink

if TYPE_CHECKING:
    from .config import AbstractConfig

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT: int = 10
TO_REDACT: set[str] = {'correlationToken', 'token'}

class AlexaDirective:
    def __init__(self, request: dict[str, Any]) -> None:
        self._directive: dict[str, Any] = request[API_DIRECTIVE]
        self.namespace: str = self._directive[API_HEADER]['namespace']
        self.name: str = self._directive[API_HEADER]['name']
        self.payload: dict[str, Any] = self._directive[API_PAYLOAD]
        self.has_endpoint: bool = API_ENDPOINT in self._directive
        self.instance: Optional[str] = None
        self.entity_id: Optional[str] = None

    def load_entity(self, hass: HomeAssistant, config: AbstractConfig) -> None:
        _endpoint_id: str = self._directive[API_ENDPOINT]['endpointId']
        self.entity_id = _endpoint_id.replace('#', '.')
        entity: State = hass.states.get(self.entity_id)
        if not entity or not config.should_expose(self.entity_id):
            raise AlexaInvalidEndpointError(_endpoint_id)
        self.entity: State = entity
        self.endpoint: AlexaEntity = ENTITY_ADAPTERS[self.entity.domain](hass, config, self.entity)
        if 'instance' in self._directive[API_HEADER]:
            self.instance = self._directive[API_HEADER]['instance']

    def response(self, name: str = 'Response', namespace: str = 'Alexa', payload: Optional[dict[str, Any]] = None) -> AlexaResponse:
        response: AlexaResponse = AlexaResponse(name, namespace, payload)
        token: Optional[str] = self._directive[API_HEADER].get('correlationToken')
        if token:
            response.set_correlation_token(token)
        if self.has_endpoint:
            response.set_endpoint(self._directive[API_ENDPOINT].copy())
        return response

    def error(self, namespace: str = 'Alexa', error_type: str = 'INTERNAL_ERROR', error_message: str = '', payload: Optional[dict[str, Any]] = None) -> AlexaResponse:
        payload = payload or {}
        payload['type'] = error_type
        payload['message'] = error_message
        _LOGGER.info('Request %s/%s error %s: %s', self._directive[API_HEADER]['namespace'], self._directive[API_HEADER]['name'], error_type, error_message)
        return self.response(name='ErrorResponse', namespace=namespace, payload=payload)

class AlexaResponse:
    def __init__(self, name: str, namespace: str, payload: Optional[dict[str, Any]] = None) -> None:
        payload = payload or {}
        self._response: dict[str, Any] = {API_EVENT: {API_HEADER: {'namespace': namespace, 'name': name, 'messageId': str(uuid4()), 'payloadVersion': '3'}, API_PAYLOAD: payload}}

    @property
    def name(self) -> str:
        return self._response[API_EVENT][API_HEADER]['name']

    @property
    def namespace(self) -> str:
        return self._response[API_EVENT][API_HEADER]['namespace']

    def set_correlation_token(self, token: str) -> None:
        self._response[API_EVENT][API_HEADER]['correlationToken'] = token

    def set_endpoint_full(self, bearer_token: str, endpoint_id: str) -> None:
        self._response[API_EVENT][API_ENDPOINT] = {API_SCOPE: {'type': 'BearerToken', 'token': bearer_token}}
        if endpoint_id is not None:
            self._response[API_EVENT][API_ENDPOINT]['endpointId'] = endpoint_id

    def set_endpoint(self, endpoint: dict[str, Any]) -> None:
        self._response[API_EVENT][API_ENDPOINT] = endpoint

    def _properties(self) -> List[dict[str, Any]]:
        context: dict[str, Any] = self._response.setdefault(API_CONTEXT, {})
        properties: List[dict[str, Any]] = context.setdefault('properties', [])
        return properties

    def add_context_property(self, prop: dict[str, Any]) -> None:
        self._properties().append(prop)

    def merge_context_properties(self, endpoint: AlexaEntity) -> None:
        properties: List[dict[str, Any]] = self._properties()
        already_set: set[tuple[str, str]] = {(p['namespace'], p['name']) for p in properties}
        for prop in endpoint.serialize_properties():
            if (prop['namespace'], prop['name']) not in already_set:
                self.add_context_property(prop)

    def serialize(self) -> dict[str, Any]:
        return self._response

async def async_enable_proactive_mode(hass: HomeAssistant, smart_home_config: AbstractConfig) -> CALLBACK_TYPE:
    await smart_home_config.async_get_access_token()

    @callback
    def extra_significant_check(hass: HomeAssistant, old_state: State, old_attrs: dict[str, Any], old_extra_arg: Any, new_state: State, new_attrs: dict[str, Any], new_extra_arg: Any) -> bool:
        return old_extra_arg is not None and old_extra_arg != new_extra_arg

    checker = await create_checker(hass, DOMAIN, extra_significant_check)

    @callback
    def _async_entity_state_filter(data: dict[str, Any]) -> bool:
        if not hass.is_running:
            return False
        if not (new_state := data['new_state']):
            return False
        if new_state.domain not in ENTITY_ADAPTERS:
            return False
        changed_entity: str = data['entity_id']
        if not smart_home_config.should_expose(changed_entity):
            _LOGGER.debug('Not exposing %s because filtered by config', changed_entity)
            return False
        return True

    async def _async_entity_state_listener(event_: Event) -> None:
        data: dict[str, Any] = event_.data
        new_state: State = data['new_state']
        if TYPE_CHECKING:
            assert new_state is not None
        alexa_changed_entity: AlexaEntity = ENTITY_ADAPTERS[new_state.domain](hass, smart_home_config, new_state)
        should_report: bool = False
        should_doorbell: bool = False
        for interface in alexa_changed_entity.interfaces():
            if not should_report and interface.properties_proactively_reported():
                should_report = True
            if interface.name() == 'Alexa.DoorbellEventSource':
                should_doorbell = True
                break
        if not should_report and (not should_doorbell):
            return
        if should_doorbell:
            old_state: State = data['old_state']
            if new_state.domain == event.DOMAIN or (new_state.state == STATE_ON and (old_state is None or old_state.state != STATE_ON)):
                await async_send_doorbell_event_message(hass, smart_home_config, alexa_changed_entity)
            return
        alexa_properties: List[dict[str, Any]] = list(alexa_changed_entity.serialize_properties())
        if not checker.async_is_significant_change(new_state, extra_arg=alexa_properties):
            return
        await async_send_changereport_message(hass, smart_home_config, alexa_changed_entity, alexa_properties)

    return hass.bus.async_listen(EVENT_STATE_CHANGED, _async_entity_state_listener, event_filter=_async_entity_state_filter)

async def async_send_changereport_message(hass: HomeAssistant, config: AbstractConfig, alexa_entity: AlexaEntity, alexa_properties: List[dict[str, Any]], invalidate_access_token: bool = True) -> None:
    try:
        token: str = await config.async_get_access_token()
    except (RequireRelink, NoTokenAvailable):
        await config.set_authorized(False)
        _LOGGER.error('Error when sending ChangeReport to Alexa, could not get access token')
        return
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'}
    endpoint: str = alexa_entity.alexa_id()
    payload: dict[str, Any] = {API_CHANGE: {'cause': {'type': Cause.APP_INTERACTION}, 'properties': alexa_properties}}
    message: AlexaResponse = AlexaResponse(name='ChangeReport', namespace='Alexa', payload=payload)
    message.set_endpoint_full(token, endpoint)
    message_serialized: dict[str, Any] = message.serialize()
    session = async_get_clientsession(hass)
    assert config.endpoint is not None
    try:
        async with timeout(DEFAULT_TIMEOUT):
            response = await session.post(config.endpoint, headers=headers, json=message_serialized, allow_redirects=True)
    except (TimeoutError, aiohttp.ClientError):
        _LOGGER.error('Timeout sending report to Alexa for %s', alexa_entity.entity_id)
        return
    response_text: str = await response.text()
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug('Sent: %s', json.dumps(async_redact_auth_data(message_serialized)))
        _LOGGER.debug('Received (%s): %s', response.status, response_text)
    if response.status == HTTPStatus.ACCEPTED:
        return
    response_json: dict[str, Any] = json_loads_object(response_text)
    response_payload: JsonObjectType = cast(JsonObjectType, response_json['payload'])
    if response_payload['code'] == 'INVALID_ACCESS_TOKEN_EXCEPTION':
        if invalidate_access_token:
            config.async_invalidate_access_token()
            await async_send_changereport_message(hass, config, alexa_entity, alexa_properties, invalidate_access_token=False)
            return
        await config.set_authorized(False)
    _LOGGER.error('Error when sending ChangeReport for %s to Alexa: %s: %s', alexa_entity.entity_id, response_payload['code'], response_payload['description'])

async def async_send_add_or_update_message(hass: HomeAssistant, config: AbstractConfig, entity_ids: List[str]) -> None:
    token: str = await config.async_get_access_token()
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'}
    endpoints: List[dict[str, Any]] = []
    for entity_id in entity_ids:
        if (domain := entity_id.split('.', 1)[0]) not in ENTITY_ADAPTERS:
            continue
        if (state := hass.states.get(entity_id)) is None:
            continue
        alexa_entity: AlexaEntity = ENTITY_ADAPTERS[domain](hass, config, state)
        endpoints.append(alexa_entity.serialize_discovery())
    payload: dict[str, Any] = {'endpoints': endpoints, 'scope': {'type': 'BearerToken', 'token': token}}
    message: AlexaResponse = AlexaResponse(name='AddOrUpdateReport', namespace='Alexa.Discovery', payload=payload)
    message_serialized: dict[str, Any] = message.serialize()
    session = async_get_clientsession(hass)
    assert config.endpoint is not None
    return await session.post(config.endpoint, headers=headers, json=message_serialized, allow_redirects=True)

async def async_send_delete_message(hass: HomeAssistant, config: AbstractConfig, entity_ids: List[str]) -> None:
    token: str = await config.async_get_access_token()
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'}
    endpoints: List[dict[str, Any]] = []
    for entity_id in entity_ids:
        domain: str = entity_id.split('.', 1)[0]
        if domain not in ENTITY_ADAPTERS:
            continue
        endpoints.append({'endpointId': config.generate_alexa_id(entity_id)})
    payload: dict[str, Any] = {'endpoints': endpoints, 'scope': {'type': 'BearerToken', 'token': token}}
    message: AlexaResponse = AlexaResponse(name='DeleteReport', namespace='Alexa.Discovery', payload=payload)
    message_serialized: dict[str, Any] = message.serialize()
    session = async_get_clientsession(hass)
    assert config.endpoint is not None
    return await session.post(config.endpoint, headers=headers, json=message_serialized, allow_redirects=True)

async def async_send_doorbell_event_message(hass: HomeAssistant, config: AbstractConfig, alexa_entity: AlexaEntity) -> None:
    token: str = await config.async_get_access_token()
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'}
    endpoint: str = alexa_entity.alexa_id()
    message: AlexaResponse = AlexaResponse(name='DoorbellPress', namespace='Alexa.DoorbellEventSource', payload={'cause': {'type': Cause.PHYSICAL_INTERACTION}, 'timestamp': dt_util.utcnow().strftime(DATE_FORMAT)})
    message.set_endpoint_full(token, endpoint)
    message_serialized: dict[str, Any] = message.serialize()
    session = async_get_clientsession(hass)
    assert config.endpoint is not None
    try:
        async with timeout(DEFAULT_TIMEOUT):
            response = await session.post(config.endpoint, headers=headers, json=message_serialized, allow_redirects=True)
    except (TimeoutError, aiohttp.ClientError):
        _LOGGER.error('Timeout sending report to Alexa for %s', alexa_entity.entity_id)
        return
    response_text: str = await response.text()
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug('Sent: %s', json.dumps(async_redact_auth_data(message_serialized)))
        _LOGGER.debug('Received (%s): %s', response.status, response_text)
    if response.status == HTTPStatus.ACCEPTED:
        return
    response_json: dict[str, Any] = json_loads_object(response_text)
    response_payload: JsonObjectType = cast(JsonObjectType, response_json['payload'])
    _LOGGER.error('Error when sending DoorbellPress event for %s to Alexa: %s: %s', alexa_entity.entity_id, response_payload['code'], response_payload['description'])
