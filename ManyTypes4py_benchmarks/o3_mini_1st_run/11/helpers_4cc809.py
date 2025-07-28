from __future__ import annotations
from abc import ABC, abstractmethod
from asyncio import gather
from collections.abc import Callable, Collection, Mapping
from datetime import datetime, timedelta
from functools import lru_cache
from http import HTTPStatus
import logging
import pprint
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union
from aiohttp.web import Request, Response, json_response
from awesomeversion import AwesomeVersion
from yarl import URL

from homeassistant.components import webhook
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_SUPPORTED_FEATURES,
    CLOUD_NEVER_EXPOSED_ENTITIES,
    CONF_NAME,
    STATE_UNAVAILABLE,
)
from homeassistant.core import CALLBACK_TYPE, Context, HomeAssistant, State, callback
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er, start
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import get_url
from homeassistant.helpers.redact import partial_redact
from homeassistant.util.dt import utcnow

from . import trait
from .const import (
    CONF_ALIASES,
    CONF_ROOM_HINT,
    DEVICE_CLASS_TO_GOOGLE_TYPES,
    DOMAIN,
    DOMAIN_TO_GOOGLE_TYPES,
    ERR_FUNCTION_NOT_SUPPORTED,
    NOT_EXPOSE_LOCAL,
    SOURCE_LOCAL,
)
from .data_redaction import async_redact_msg
from .error import SmartHomeError

SYNC_DELAY: int = 15
_LOGGER = logging.getLogger(__name__)
LOCAL_SDK_VERSION_HEADER: str = 'HA-Cloud-Version'
LOCAL_SDK_MIN_VERSION: AwesomeVersion = AwesomeVersion('2.1.5')


@callback
def _get_registry_entries(hass: HomeAssistant, entity_id: str) -> Tuple[Optional[er.RegistryEntry], Optional[dr.DeviceEntry], Optional[ar.AreaEntry]]:
    """Get registry entries."""
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)
    if (entity_entry := ent_reg.async_get(entity_id)) and entity_entry.device_id:
        device_entry = dev_reg.devices.get(entity_entry.device_id)
    else:
        device_entry = None
    if entity_entry and entity_entry.area_id:
        area_id = entity_entry.area_id
    elif device_entry and device_entry.area_id:
        area_id = device_entry.area_id
    else:
        area_id = None
    if area_id is not None:
        area_entry = area_reg.async_get_area(area_id)
    else:
        area_entry = None
    return (entity_entry, device_entry, area_entry)


class AbstractConfig(ABC):
    """Hold the configuration for Google Assistant."""
    _unsub_report_state: Optional[CALLBACK_TYPE] = None

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize abstract config."""
        self.hass: HomeAssistant = hass
        self._google_sync_unsub: Dict[str, CALLBACK_TYPE] = {}
        self._local_sdk_active: bool = False
        self._local_last_active: Optional[datetime] = None
        self._local_sdk_version_warn: bool = False
        self.is_supported_cache: Dict[str, Tuple[Any, bool]] = {}
        self._on_deinitialize: List[CALLBACK_TYPE] = []

    async def async_initialize(self) -> None:
        """Perform async initialization of config."""
        if not self.enabled:
            return

        async def sync_google(_: Any) -> None:
            """Sync entities to Google."""
            await self.async_sync_entities_all()

        self._on_deinitialize.append(start.async_at_start(self.hass, sync_google))

    @callback
    def async_deinitialize(self) -> None:
        """Remove listeners."""
        _LOGGER.debug('async_deinitialize')
        while self._on_deinitialize:
            self._on_deinitialize.pop()()

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Return if Google is enabled."""

    @property
    @abstractmethod
    def entity_config(self) -> Mapping[str, Any]:
        """Return entity config."""

    @property
    @abstractmethod
    def secure_devices_pin(self) -> str:
        """Return entity config."""

    @property
    def is_reporting_state(self) -> bool:
        """Return if we're actively reporting states."""
        return self._unsub_report_state is not None

    @property
    def is_local_sdk_active(self) -> bool:
        """Return if we're actively accepting local messages."""
        return self._local_sdk_active

    @property
    @abstractmethod
    def should_report_state(self) -> bool:
        """Return if states should be proactively reported."""

    @abstractmethod
    def get_local_user_id(self, webhook_id: str) -> Optional[str]:
        """Map webhook ID to a Home Assistant user ID.

        Any action initiated by Google Assistant via the local SDK will be attributed
        to the returned user ID.

        Return None if no user id is found for the webhook_id.
        """

    @abstractmethod
    def get_local_webhook_id(self, agent_user_id: str) -> Optional[str]:
        """Return the webhook ID to be used for actions for a given agent user id via the local SDK."""

    @abstractmethod
    def get_agent_user_id_from_context(self, context: Context) -> str:
        """Get agent user ID from context."""

    @abstractmethod
    def get_agent_user_id_from_webhook(self, webhook_id: str) -> Optional[str]:
        """Map webhook ID to a Google agent user ID.

        Return None if no agent user id is found for the webhook_id.
        """

    @abstractmethod
    def should_expose(self, state: State) -> bool:
        """Return if entity should be exposed."""

    @abstractmethod
    def should_2fa(self, state: State) -> bool:
        """If an entity should have 2FA checked."""

    @abstractmethod
    async def async_report_state(self, message: Any, agent_user_id: str, event_id: Optional[str] = None) -> HTTPStatus:
        """Send a state report to Google."""

    async def async_report_state_all(self, message: Any) -> None:
        """Send a state report to Google for all previously synced users."""
        jobs = [self.async_report_state(message, agent_user_id) for agent_user_id in self.async_get_agent_users()]
        await gather(*jobs)

    @callback
    def async_enable_report_state(self) -> None:
        """Enable proactive mode."""
        from .report_state import async_enable_report_state
        if self._unsub_report_state is None:
            self._unsub_report_state = async_enable_report_state(self.hass, self)

    @callback
    def async_disable_report_state(self) -> None:
        """Disable report state."""
        if self._unsub_report_state is not None:
            self._unsub_report_state()
            self._unsub_report_state = None

    async def async_sync_entities(self, agent_user_id: str) -> HTTPStatus:
        """Sync all entities to Google."""
        self._google_sync_unsub.pop(agent_user_id, lambda: None)()
        status: HTTPStatus = await self._async_request_sync_devices(agent_user_id)
        if status == HTTPStatus.NOT_FOUND:
            await self.async_disconnect_agent_user(agent_user_id)
        return status

    async def async_sync_entities_all(self) -> int:
        """Sync all entities to Google for all registered agents."""
        if not self.async_get_agent_users():
            return 204
        res: List[HTTPStatus] = await gather(*(self.async_sync_entities(agent_user_id) for agent_user_id in self.async_get_agent_users()))
        return max(res, default=HTTPStatus.NO_CONTENT).value

    async def async_sync_notification(self, agent_user_id: str, event_id: str, payload: Any) -> HTTPStatus:
        """Sync notifications to Google."""
        self._google_sync_unsub.pop(agent_user_id, lambda: None)()
        status: HTTPStatus = await self.async_report_state(payload, agent_user_id, event_id)
        assert status is not None
        if status == HTTPStatus.NOT_FOUND:
            await self.async_disconnect_agent_user(agent_user_id)
        return status

    async def async_sync_notification_all(self, event_id: str, payload: Any) -> HTTPStatus:
        """Sync notification to Google for all registered agents."""
        if not self.async_get_agent_users():
            return HTTPStatus.NO_CONTENT
        res: List[HTTPStatus] = await gather(*(self.async_sync_notification(agent_user_id, event_id, payload) for agent_user_id in self.async_get_agent_users()))
        return max(res, default=HTTPStatus.NO_CONTENT)

    @callback
    def async_schedule_google_sync(self, agent_user_id: str) -> None:
        """Schedule a sync."""

        async def _schedule_callback(_now: Any) -> None:
            """Handle a scheduled sync callback."""
            self._google_sync_unsub.pop(agent_user_id, None)
            await self.async_sync_entities(agent_user_id)

        self._google_sync_unsub.pop(agent_user_id, lambda: None)()
        self._google_sync_unsub[agent_user_id] = async_call_later(self.hass, SYNC_DELAY, _schedule_callback)

    @callback
    def async_schedule_google_sync_all(self) -> None:
        """Schedule a sync for all registered agents."""
        for agent_user_id in self.async_get_agent_users():
            self.async_schedule_google_sync(agent_user_id)

    async def _async_request_sync_devices(self, agent_user_id: str) -> HTTPStatus:
        """Trigger a sync with Google.

        Return value is the HTTP status code of the sync request.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_connect_agent_user(self, agent_user_id: str) -> None:
        """Add a synced and known agent_user_id.

        Called before sending a sync response to Google.
        """

    @abstractmethod
    async def async_disconnect_agent_user(self, agent_user_id: str) -> None:
        """Turn off report state and disable further state reporting.

        Called when:
         - The user disconnects their account from Google.
         - When the cloud configuration is initialized
         - When sync entities fails with 404
        """

    @callback
    @abstractmethod
    def async_get_agent_users(self) -> Collection[str]:
        """Return known agent users."""

    @callback
    def async_enable_local_sdk(self) -> None:
        """Enable the local SDK."""
        _LOGGER.debug('async_enable_local_sdk')
        setup_successful: bool = True
        setup_webhook_ids: List[str] = []
        if self.hass.config.api and self.hass.config.api.use_ssl:
            self._local_sdk_active = False
            return
        for user_agent_id in self.async_get_agent_users():
            webhook_id = self.get_local_webhook_id(user_agent_id)
            if webhook_id is None:
                setup_successful = False
                break
            _LOGGER.debug(
                'Register webhook handler %s for agent user id %s',
                partial_redact(webhook_id),
                partial_redact(user_agent_id),
            )
            try:
                webhook.async_register(
                    self.hass,
                    DOMAIN,
                    'Local Support for ' + user_agent_id,
                    webhook_id,
                    self._handle_local_webhook,
                    local_only=True,
                )
                setup_webhook_ids.append(webhook_id)
            except ValueError:
                _LOGGER.warning(
                    'Webhook handler %s for agent user id %s is already defined!',
                    partial_redact(webhook_id),
                    partial_redact(user_agent_id),
                )
                setup_successful = False
                break
        if not setup_successful:
            _LOGGER.warning('Local fulfillment failed to setup, falling back to cloud fulfillment')
            for setup_webhook_id in setup_webhook_ids:
                webhook.async_unregister(self.hass, setup_webhook_id)
        self._local_sdk_active = setup_successful

    @callback
    def async_disable_local_sdk(self) -> None:
        """Disable the local SDK."""
        _LOGGER.debug('async_disable_local_sdk')
        if not self._local_sdk_active:
            return
        for agent_user_id in self.async_get_agent_users():
            webhook_id = self.get_local_webhook_id(agent_user_id)
            _LOGGER.debug(
                'Unregister webhook handler %s for agent user id %s',
                partial_redact(webhook_id),
                partial_redact(agent_user_id),
            )
            webhook.async_unregister(self.hass, webhook_id)
        self._local_sdk_active = False

    async def _handle_local_webhook(self, hass: HomeAssistant, webhook_id: str, request: Request) -> Response:
        """Handle an incoming local SDK message."""
        from . import smart_home
        self._local_last_active = utcnow()
        version: Optional[str] = request.headers.get('HA-Cloud-Version')
        if not self._local_sdk_version_warn and (not version or AwesomeVersion(version) < LOCAL_SDK_MIN_VERSION):
            _LOGGER.warning('Local SDK version is too old (%s), check documentation on how to update to the latest version', version)
            self._local_sdk_version_warn = True
        payload: Any = await request.json()
        if _LOGGER.isEnabledFor(logging.DEBUG):
            msgid: str = '<UNKNOWN>'
            if isinstance(payload, dict):
                msgid = payload.get('requestId', '<UNKNOWN>')
            _LOGGER.debug('Received local message %s from %s (JS %s)', msgid, request.remote, request.headers.get('HA-Cloud-Version', 'unknown'))
        agent_user_id: Optional[str] = self.get_agent_user_id_from_webhook(webhook_id)
        if agent_user_id is None:
            _LOGGER.error(
                'Cannot process request for webhook %s as no linked agent user is found:\n%s\n',
                partial_redact(webhook_id),
                pprint.pformat(async_redact_msg(payload, agent_user_id)),
            )
            webhook.async_unregister(self.hass, webhook_id)
            return json_response({})
        if not self.enabled:
            return json_response(smart_home.api_disabled_response(payload, agent_user_id))
        result: Any = await smart_home.async_handle_message(
            self.hass,
            self,
            agent_user_id,
            self.get_local_user_id(webhook_id),
            payload,
            SOURCE_LOCAL,
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            if isinstance(payload, dict):
                _LOGGER.debug('Responding to local message %s', msgid)
            else:
                _LOGGER.debug('Empty response to local message %s', msgid)
        return json_response(result)


class RequestData:
    """Hold data associated with a particular request."""

    def __init__(self, config: AbstractConfig, user_id: str, source: str, request_id: str, devices: Any) -> None:
        """Initialize the request data."""
        self.config: AbstractConfig = config
        self.source: str = source
        self.request_id: str = request_id
        self.context: Context = Context(user_id=user_id)
        self.devices: Any = devices

    @property
    def is_local_request(self) -> bool:
        """Return if this is a local request."""
        return self.source == SOURCE_LOCAL


def get_google_type(domain: str, device_class: Optional[Any]) -> Any:
    """Google type based on domain and device class."""
    typ: Any = DEVICE_CLASS_TO_GOOGLE_TYPES.get((domain, device_class))
    return typ if typ is not None else DOMAIN_TO_GOOGLE_TYPES[domain]


@lru_cache(maxsize=4096)
def supported_traits_for_state(state: State) -> List[Any]:
    """Return all supported traits for state."""
    domain: str = state.domain
    attributes: Dict[str, Any] = state.attributes
    features: Any = attributes.get(ATTR_SUPPORTED_FEATURES, 0)
    if not isinstance(features, int):
        _LOGGER.warning('Entity %s contains invalid supported_features value %s', state.entity_id, features)
        return []
    device_class = state.attributes.get(ATTR_DEVICE_CLASS)
    return [Trait for Trait in trait.TRAITS if Trait.supported(domain, features, device_class, attributes)]


class GoogleEntity:
    """Adaptation of Entity expressed in Google's terms."""
    __slots__ = ('_traits', 'config', 'entity_id', 'hass', 'state')

    def __init__(self, hass: HomeAssistant, config: AbstractConfig, state: State) -> None:
        """Initialize a Google entity."""
        self.hass: HomeAssistant = hass
        self.config: AbstractConfig = config
        self.state: State = state
        self.entity_id: str = state.entity_id
        self._traits: Optional[List[Any]] = None

    def __repr__(self) -> str:
        """Return the representation."""
        return f'<GoogleEntity {self.state.entity_id}: {self.state.name}>'

    @callback
    def traits(self) -> List[Any]:
        """Return traits for entity."""
        if self._traits is not None:
            return self._traits
        state: State = self.state
        self._traits = [Trait(self.hass, state, self.config) for Trait in supported_traits_for_state(state)]
        return self._traits

    @callback
    def should_expose(self) -> bool:
        """If entity should be exposed."""
        return self.config.should_expose(self.state)

    @callback
    def should_expose_local(self) -> bool:
        """Return if the entity should be exposed locally."""
        return self.should_expose() and get_google_type(self.state.domain, self.state.attributes.get(ATTR_DEVICE_CLASS)) not in NOT_EXPOSE_LOCAL and (not self.might_2fa())

    @callback
    def is_supported(self) -> bool:
        """Return if entity is supported."""
        return bool(self.traits())

    @callback
    def might_2fa(self) -> bool:
        """Return if the entity might encounter 2FA."""
        if not self.config.should_2fa(self.state):
            return False
        return self.might_2fa_traits()

    @callback
    def might_2fa_traits(self) -> bool:
        """Return if the entity might encounter 2FA based on just traits."""
        state: State = self.state
        domain: str = state.domain
        features: Any = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)
        device_class = state.attributes.get(ATTR_DEVICE_CLASS)
        return any((trait.might_2fa(domain, features, device_class) for trait in self.traits()))

    def sync_serialize(self, agent_user_id: str, instance_uuid: str) -> Dict[str, Any]:
        """Serialize entity for a SYNC response.

        https://developers.google.com/actions/smarthome/create-app#actiondevicessync
        """
        state: State = self.state
        traits: List[Any] = self.traits()
        entity_config: Dict[str, Any] = self.config.entity_config.get(state.entity_id, {})
        name: str = (entity_config.get(CONF_NAME) or state.name).strip()
        entity_entry, device_entry, area_entry = _get_registry_entries(self.hass, self.entity_id)
        device: Dict[str, Any] = {
            'id': state.entity_id,
            'name': {'name': name},
            'attributes': {},
            'traits': [trait.name for trait in traits],
            'willReportState': self.config.should_report_state,
            'type': get_google_type(state.domain, state.attributes.get(ATTR_DEVICE_CLASS)),
        }
        config_aliases: List[str] = entity_config.get(CONF_ALIASES, [])
        if config_aliases or (entity_entry and entity_entry.aliases):
            device['name']['nicknames'] = [name, *config_aliases]
            if entity_entry:
                device['name']['nicknames'].extend(entity_entry.aliases)
        if self.config.is_local_sdk_active and self.should_expose_local():
            device['otherDeviceIds'] = [{'deviceId': self.entity_id}]
            device['customData'] = {
                'webhookId': self.config.get_local_webhook_id(agent_user_id),
                'httpPort': URL(get_url(self.hass, allow_external=False)).port,
                'uuid': instance_uuid,
            }
        for trt in traits:
            device['attributes'].update(trt.sync_attributes())
        for trt in traits:
            device.update(trt.sync_options())
        if (room := entity_config.get(CONF_ROOM_HINT)):
            device['roomHint'] = room
        elif area_entry and area_entry.name:
            device['roomHint'] = area_entry.name
        if not device_entry:
            return device
        if 'matter' in self.hass.config.components and any((x for x in device_entry.identifiers if x[0] == 'matter')):
            from homeassistant.components.matter import get_matter_device_info
            if (matter_info := get_matter_device_info(self.hass, device_entry.id)):
                device['matterUniqueId'] = matter_info['unique_id']
                device['matterOriginalVendorId'] = matter_info['vendor_id']
                device['matterOriginalProductId'] = matter_info['product_id']
        device_info: Dict[str, Any] = {}
        if device_entry.manufacturer:
            device_info['manufacturer'] = device_entry.manufacturer
        if device_entry.model:
            device_info['model'] = device_entry.model
        if device_entry.sw_version:
            device_info['swVersion'] = device_entry.sw_version
        if device_info:
            device['deviceInfo'] = device_info
        return device

    @callback
    def query_serialize(self) -> Dict[str, Any]:
        """Serialize entity for a QUERY response.

        https://developers.google.com/actions/smarthome/create-app#actiondevicesquery
        """
        state: State = self.state
        if state.state == STATE_UNAVAILABLE:
            return {'online': False}
        attrs: Dict[str, Any] = {'online': True}
        for trt in self.traits():
            deep_update(attrs, trt.query_attributes())
        return attrs

    @callback
    def notifications_serialize(self) -> Optional[Dict[str, Any]]:
        """Serialize the payload for notifications to be sent."""
        notifications: Dict[str, Any] = {}
        for trt in self.traits():
            deep_update(notifications, trt.query_notifications() or {})
        return notifications or None

    @callback
    def reachable_device_serialize(self) -> Dict[str, Any]:
        """Serialize entity for a REACHABLE_DEVICE response."""
        return {'verificationId': self.entity_id}

    async def execute(self, data: RequestData, command_payload: Dict[str, Any]) -> None:
        """Execute a command.

        https://developers.google.com/actions/smarthome/create-app#actiondevicesexecute
        """
        command: str = command_payload['command']
        params: Dict[str, Any] = command_payload.get('params', {})
        challenge: Dict[str, Any] = command_payload.get('challenge', {})
        executed: bool = False
        for trt in self.traits():
            if trt.can_execute(command, params):
                await trt.execute(command, data, params, challenge)
                executed = True
                break
        if not executed:
            raise SmartHomeError(ERR_FUNCTION_NOT_SUPPORTED, f'Unable to execute {command} for {self.state.entity_id}')

    @callback
    def async_update(self) -> None:
        """Update the entity with latest info from Home Assistant."""
        self.state = self.hass.states.get(self.entity_id)
        if self._traits is None:
            return
        for trt in self._traits:
            trt.state = self.state


def deep_update(target: Dict[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
    """Update a nested dictionary with another nested dictionary."""
    for key, value in source.items():
        if isinstance(value, Mapping):
            target[key] = deep_update(target.get(key, {}), value)
        else:
            target[key] = value
    return target


@callback
def async_get_google_entity_if_supported_cached(
    hass: HomeAssistant, config: AbstractConfig, state: State
) -> Optional[GoogleEntity]:
    """Return a GoogleEntity if entity is supported checking the cache first.

    This function will check the cache, and call async_get_google_entity_if_supported
    if the entity is not in the cache, which will update the cache.
    """
    entity_id: str = state.entity_id
    is_supported_cache: Dict[str, Tuple[Any, bool]] = config.is_supported_cache
    features: Any = state.attributes.get(ATTR_SUPPORTED_FEATURES)
    if (result := is_supported_cache.get(entity_id)):
        cached_features, supported = result
        if cached_features == features:
            return GoogleEntity(hass, config, state) if supported else None
    return async_get_google_entity_if_supported(hass, config, state)


@callback
def async_get_google_entity_if_supported(
    hass: HomeAssistant, config: AbstractConfig, state: State
) -> Optional[GoogleEntity]:
    """Return a GoogleEntity if entity is supported.

    This function will update the cache, but it does not check the cache first.
    """
    features: Any = state.attributes.get(ATTR_SUPPORTED_FEATURES)
    entity: GoogleEntity = GoogleEntity(hass, config, state)
    is_supported: bool = bool(entity.traits())
    config.is_supported_cache[state.entity_id] = (features, is_supported)
    return entity if is_supported else None


@callback
def async_get_entities(hass: HomeAssistant, config: AbstractConfig) -> List[GoogleEntity]:
    """Return all entities that are supported by Google."""
    entities: List[GoogleEntity] = []
    is_supported_cache: Dict[str, Tuple[Any, bool]] = config.is_supported_cache
    for state in hass.states.async_all():
        entity_id: str = state.entity_id
        if entity_id in CLOUD_NEVER_EXPOSED_ENTITIES:
            continue
        features: Any = state.attributes.get(ATTR_SUPPORTED_FEATURES)
        if (result := is_supported_cache.get(entity_id)):
            cached_features, supported = result
            if cached_features == features:
                if supported:
                    entities.append(GoogleEntity(hass, config, state))
                continue
        if (entity := async_get_google_entity_if_supported(hass, config, state)):
            entities.append(entity)
    return entities