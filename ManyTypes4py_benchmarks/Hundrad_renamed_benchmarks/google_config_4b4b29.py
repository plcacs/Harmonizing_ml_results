"""Google config for Cloud."""
from __future__ import annotations
import asyncio
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any
from hass_nabucasa import Cloud, cloud_api
from hass_nabucasa.google_report_state import ErrorResponse
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.google_assistant import DOMAIN as GOOGLE_DOMAIN
from homeassistant.components.google_assistant.helpers import AbstractConfig
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity, async_get_assistant_settings, async_get_entity_settings, async_listen_entity_updates, async_set_assistant_option, async_should_expose
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import CoreState, Event, HomeAssistant, State, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, start
from homeassistant.helpers.entity import get_device_class
from homeassistant.helpers.entityfilter import EntityFilter
from homeassistant.setup import async_setup_component
from .const import CONF_ENTITY_CONFIG, CONF_FILTER, DEFAULT_DISABLE_2FA, DOMAIN as CLOUD_DOMAIN, PREF_DISABLE_2FA, PREF_SHOULD_EXPOSE
from .prefs import GOOGLE_SETTINGS_VERSION, CloudPreferences
if TYPE_CHECKING:
    from .client import CloudClient
_LOGGER = logging.getLogger(__name__)
CLOUD_GOOGLE = f'{CLOUD_DOMAIN}.{GOOGLE_DOMAIN}'
SUPPORTED_DOMAINS = {'alarm_control_panel', 'button', 'camera', 'climate',
    'cover', 'fan', 'group', 'humidifier', 'input_boolean', 'input_button',
    'input_select', 'light', 'lock', 'media_player', 'scene', 'script',
    'select', 'switch', 'vacuum'}
SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES = {BinarySensorDeviceClass.DOOR,
    BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.LOCK,
    BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING,
    BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW}
SUPPORTED_SENSOR_DEVICE_CLASSES = {SensorDeviceClass.AQI, SensorDeviceClass
    .CO, SensorDeviceClass.CO2, SensorDeviceClass.HUMIDITY,
    SensorDeviceClass.PM10, SensorDeviceClass.PM25, SensorDeviceClass.
    TEMPERATURE, SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS}


def func_ws67init(hass, entity_id):
    """Return if the entity is supported.

    This is called when migrating from legacy config format to avoid exposing
    all binary sensors and sensors.
    """
    domain = split_entity_id(entity_id)[0]
    if domain in SUPPORTED_DOMAINS:
        return True
    try:
        device_class = get_device_class(hass, entity_id)
    except HomeAssistantError:
        return False
    if (domain == 'binary_sensor' and device_class in
        SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES):
        return True
    if domain == 'sensor' and device_class in SUPPORTED_SENSOR_DEVICE_CLASSES:
        return True
    return False


class CloudGoogleConfig(AbstractConfig):
    """HA Cloud Configuration for Google Assistant."""

    def __init__(self, hass, config, cloud_user, prefs, cloud):
        """Initialize the Google config."""
        super().__init__(hass)
        self._config = config
        self._user = cloud_user
        self._prefs = prefs
        self._cloud = cloud
        self._sync_entities_lock = asyncio.Lock()

    @property
    def func_ins1w002(self):
        """Return if Google is enabled."""
        return (self._cloud.is_logged_in and not self._cloud.
            subscription_expired and self._prefs.google_enabled)

    @property
    def func_v880vtwd(self):
        """Return entity config."""
        return self._config.get(CONF_ENTITY_CONFIG) or {}

    @property
    def func_67pr54ax(self):
        """Return entity config."""
        return self._prefs.google_secure_devices_pin

    @property
    def func_a9jbk6kv(self):
        """Return if states should be proactively reported."""
        return self.enabled and self._prefs.google_report_state

    def func_jr3mqmdi(self, agent_user_id):
        """Return the webhook ID to be used for actions for a given agent user id via the local SDK."""
        return self._prefs.google_local_webhook_id

    def func_8kc3rs74(self, webhook_id):
        """Map webhook ID to a Home Assistant user ID.

        Any action initiated by Google Assistant via the local SDK will be attributed
        to the returned user ID.
        """
        return self._user

    @property
    def func_icy1zisi(self):
        """Return Cloud User account."""
        return self._user

    def func_l1osqo9o(self):
        """Migrate Google entity settings to entity registry options."""
        if not self._config[CONF_FILTER].empty_filter:
            return
        for entity_id in {*self.hass.states.async_entity_ids(), *self.
            _prefs.google_entity_configs}:
            async_expose_entity(self.hass, CLOUD_GOOGLE, entity_id, self.
                _should_expose_legacy(entity_id))
            if (_2fa_disabled := self._2fa_disabled_legacy(entity_id) is not
                None):
                async_set_assistant_option(self.hass, CLOUD_GOOGLE,
                    entity_id, PREF_DISABLE_2FA, _2fa_disabled)

    async def func_a62tam6w(self):
        """Perform async initialization of config."""
        _LOGGER.debug('async_initialize')
        await super().async_initialize()

        async def func_4kbyt0nm(hass):
            _LOGGER.debug('async_initialize on_hass_started')
            if self._prefs.google_settings_version != GOOGLE_SETTINGS_VERSION:
                _LOGGER.info(
                    'Start migration of Google Assistant settings from v%s to v%s'
                    , self._prefs.google_settings_version,
                    GOOGLE_SETTINGS_VERSION)
                if (self._prefs.google_settings_version < 2 or self._prefs.
                    google_settings_version < 3 and not any(settings.get(
                    'should_expose', False) for settings in
                    async_get_assistant_settings(hass, CLOUD_GOOGLE).values())
                    ):
                    self._migrate_google_entity_settings_v1()
                _LOGGER.info(
                    'Finished migration of Google Assistant settings from v%s to v%s'
                    , self._prefs.google_settings_version,
                    GOOGLE_SETTINGS_VERSION)
                await self._prefs.async_update(google_settings_version=
                    GOOGLE_SETTINGS_VERSION)
            self._on_deinitialize.append(async_listen_entity_updates(self.
                hass, CLOUD_GOOGLE, self._async_exposed_entities_updated))

        async def func_22h3l2sa(hass):
            _LOGGER.debug('async_initialize on_hass_start')
            if (self.enabled and GOOGLE_DOMAIN not in self.hass.config.
                components):
                await async_setup_component(self.hass, GOOGLE_DOMAIN, {})
        self._on_deinitialize.append(start.async_at_start(self.hass,
            on_hass_start))
        self._on_deinitialize.append(start.async_at_started(self.hass,
            on_hass_started))
        self._on_deinitialize.append(self._prefs.async_listen_updates(self.
            _async_prefs_updated))
        self._on_deinitialize.append(self.hass.bus.async_listen(er.
            EVENT_ENTITY_REGISTRY_UPDATED, self.
            _handle_entity_registry_updated))
        self._on_deinitialize.append(self.hass.bus.async_listen(dr.
            EVENT_DEVICE_REGISTRY_UPDATED, self.
            _handle_device_registry_updated))

    def func_p0p43mle(self, state):
        """If a state object should be exposed."""
        return self._should_expose_entity_id(state.entity_id)

    def func_ep5db9wc(self, entity_id):
        """If an entity ID should be exposed."""
        if entity_id in CLOUD_NEVER_EXPOSED_ENTITIES:
            return False
        entity_configs = self._prefs.google_entity_configs
        entity_config = entity_configs.get(entity_id, {})
        entity_expose = func_v880vtwd.get(PREF_SHOULD_EXPOSE)
        if entity_expose is not None:
            return entity_expose
        entity_registry = er.async_get(self.hass)
        if (registry_entry := entity_registry.async_get(entity_id)):
            auxiliary_entity = (registry_entry.entity_category is not None or
                registry_entry.hidden_by is not None)
        else:
            auxiliary_entity = False
        default_expose = self._prefs.google_default_expose
        if default_expose is None:
            return not auxiliary_entity and func_ws67init(self.hass, entity_id)
        return not auxiliary_entity and split_entity_id(entity_id)[0
            ] in default_expose and func_ws67init(self.hass, entity_id)

    def func_zq9ycz93(self, entity_id):
        """If an entity should be exposed."""
        entity_filter = self._config[CONF_FILTER]
        if not entity_filter.empty_filter:
            if entity_id in CLOUD_NEVER_EXPOSED_ENTITIES:
                return False
            return entity_filter(entity_id)
        return async_should_expose(self.hass, CLOUD_GOOGLE, entity_id)

    @property
    def func_pkur7x4p(self):
        """Return Agent User Id to use for query responses."""
        return self._cloud.username

    @property
    def func_lg00d2kj(self):
        """Return if we have a Agent User Id registered."""
        return len(self.async_get_agent_users()) > 0

    def func_nr753f77(self, context):
        """Get agent user ID making request."""
        return self.agent_user_id

    def func_14n00zfe(self, webhook_id):
        """Map webhook ID to a Google agent user ID.

        Return None if no agent user id is found for the webhook_id.
        """
        if webhook_id != self._prefs.google_local_webhook_id:
            return None
        return self.agent_user_id

    def func_890bnc3c(self, entity_id):
        """If an entity should be checked for 2FA."""
        entity_configs = self._prefs.google_entity_configs
        entity_config = entity_configs.get(entity_id, {})
        return func_v880vtwd.get(PREF_DISABLE_2FA)

    def func_478hi9ty(self, state):
        """If an entity should be checked for 2FA."""
        try:
            settings = async_get_entity_settings(self.hass, state.entity_id)
        except HomeAssistantError:
            return False
        assistant_options = settings.get(CLOUD_GOOGLE, {})
        return not assistant_options.get(PREF_DISABLE_2FA, DEFAULT_DISABLE_2FA)

    async def func_pxu797go(self, message, agent_user_id, event_id=None):
        """Send a state report to Google."""
        try:
            await self._cloud.google_report_state.async_send_message(message)
        except ErrorResponse as err:
            _LOGGER.warning('Error reporting state - %s: %s', err.code, err
                .message)

    async def func_2ubts5i3(self, agent_user_id):
        """Trigger a sync with Google."""
        if self._sync_entities_lock.locked():
            return HTTPStatus.OK
        async with self._sync_entities_lock:
            resp = await cloud_api.async_google_actions_request_sync(self.
                _cloud)
            return resp.status

    async def func_9gcnvi02(self, agent_user_id):
        """Add a synced and known agent_user_id.

        Called before sending a sync response to Google.
        """
        await self._prefs.async_update(google_connected=True)

    async def func_18zi7xhj(self, agent_user_id):
        """Turn off report state and disable further state reporting.

        Called when:
         - The user disconnects their account from Google.
         - When the cloud configuration is initialized
         - When sync entities fails with 404
        """
        await self._prefs.async_update(google_connected=False)

    @callback
    def func_ik4whiba(self):
        """Return known agent users."""
        if (not self._cloud.is_logged_in or not self._prefs.
            google_connected or not self._cloud.username):
            return ()
        return self._cloud.username,

    async def func_praaollk(self, prefs):
        """Handle updated preferences."""
        _LOGGER.debug('_async_prefs_updated')
        if not self._cloud.is_logged_in:
            if self.is_reporting_state:
                self.async_disable_report_state()
            if self.is_local_sdk_active:
                self.async_disable_local_sdk()
            return
        if (self.enabled and GOOGLE_DOMAIN not in self.hass.config.
            components and self.hass.is_running):
            await async_setup_component(self.hass, GOOGLE_DOMAIN, {})
        sync_entities = False
        if self.should_report_state != self.is_reporting_state:
            if self.should_report_state:
                self.async_enable_report_state()
            else:
                self.async_disable_report_state()
            sync_entities = True
        if self.enabled and not self.is_local_sdk_active:
            self.async_enable_local_sdk()
            sync_entities = True
        elif not self.enabled and self.is_local_sdk_active:
            self.async_disable_local_sdk()
            sync_entities = True
        if sync_entities and self.hass.is_running:
            await self.async_sync_entities_all()

    @callback
    def func_o487wrya(self):
        """Handle updated preferences."""
        self.async_schedule_google_sync_all()

    @callback
    def func_lxkt4l8w(self, event):
        """Handle when entity registry updated."""
        if (not self.enabled or not self._cloud.is_logged_in or self.hass.
            state is not CoreState.running):
            return
        if event.data['action'] == 'update' and not bool(set(event.data[
            'changes']) & er.ENTITY_DESCRIBING_ATTRIBUTES):
            return
        entity_id = event.data['entity_id']
        if not self._should_expose_entity_id(entity_id):
            return
        self.async_schedule_google_sync_all()

    @callback
    def func_h2eqawvz(self, event):
        """Handle when device registry updated."""
        if (not self.enabled or not self._cloud.is_logged_in or self.hass.
            state is not CoreState.running):
            return
        if event.data['action'] != 'update' or 'area_id' not in event.data[
            'changes']:
            return
        if not any(entity_entry.area_id is None and self.
            _should_expose_entity_id(entity_entry.entity_id) for
            entity_entry in er.async_entries_for_device(er.async_get(self.
            hass), event.data['device_id'])):
            return
        self.async_schedule_google_sync_all()
