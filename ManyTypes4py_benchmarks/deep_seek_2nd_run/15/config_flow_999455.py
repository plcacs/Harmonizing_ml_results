"""Config flow for MQTT."""
from __future__ import annotations
import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping
import logging
import queue
from ssl import PROTOCOL_TLS_CLIENT, SSLContext, SSLError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, List, Tuple, Set, cast
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate
import voluptuous as vol
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.components.hassio import AddonError, AddonManager, AddonState
from homeassistant.config_entries import SOURCE_RECONFIGURE, ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_CLIENT_ID, CONF_DISCOVERY, CONF_HOST, CONF_PASSWORD, CONF_PAYLOAD, CONF_PORT, CONF_PROTOCOL, CONF_USERNAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.hassio import is_hassio
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.selector import BooleanSelector, FileSelector, FileSelectorConfig, NumberSelector, NumberSelectorConfig, NumberSelectorMode, SelectOptionDict, SelectSelector, SelectSelectorConfig, SelectSelectorMode, TextSelector, TextSelectorConfig, TextSelectorType
from homeassistant.helpers.service_info.hassio import HassioServiceInfo
from homeassistant.util.json import JSON_DECODE_EXCEPTIONS, json_loads
from .addon import get_addon_manager
from .client import MqttClientSetup
from .const import ATTR_PAYLOAD, ATTR_QOS, ATTR_RETAIN, ATTR_TOPIC, CONF_BIRTH_MESSAGE, CONF_BROKER, CONF_CERTIFICATE, CONF_CLIENT_CERT, CONF_CLIENT_KEY, CONF_DISCOVERY_PREFIX, CONF_KEEPALIVE, CONF_TLS_INSECURE, CONF_TRANSPORT, CONF_WILL_MESSAGE, CONF_WS_HEADERS, CONF_WS_PATH, CONFIG_ENTRY_MINOR_VERSION, CONFIG_ENTRY_VERSION, DEFAULT_BIRTH, DEFAULT_DISCOVERY, DEFAULT_ENCODING, DEFAULT_KEEPALIVE, DEFAULT_PORT, DEFAULT_PREFIX, DEFAULT_PROTOCOL, DEFAULT_TRANSPORT, DEFAULT_WILL, DEFAULT_WS_PATH, DOMAIN, SUPPORTED_PROTOCOLS, TRANSPORT_TCP, TRANSPORT_WEBSOCKETS
from .util import async_create_certificate_temp_files, get_file_path, valid_birth_will, valid_publish_topic
_LOGGER = logging.getLogger(__name__)
ADDON_SETUP_TIMEOUT = 5
ADDON_SETUP_TIMEOUT_ROUNDS = 5
MQTT_TIMEOUT = 5
ADVANCED_OPTIONS = 'advanced_options'
SET_CA_CERT = 'set_ca_cert'
SET_CLIENT_CERT = 'set_client_cert'
BOOLEAN_SELECTOR = BooleanSelector()
TEXT_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))
PUBLISH_TOPIC_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))
PORT_SELECTOR = vol.All(NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=1, max=65535)), vol.Coerce(int))
PASSWORD_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
QOS_SELECTOR = vol.All(NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=0, max=2)), vol.Coerce(int))
KEEPALIVE_SELECTOR = vol.All(NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=15, step='any', unit_of_measurement='sec')), vol.Coerce(int))
PROTOCOL_SELECTOR = SelectSelector(SelectSelectorConfig(options=SUPPORTED_PROTOCOLS, mode=SelectSelectorMode.DROPDOWN))
SUPPORTED_TRANSPORTS = [SelectOptionDict(value=TRANSPORT_TCP, label='TCP'), SelectOptionDict(value=TRANSPORT_WEBSOCKETS, label='WebSocket')]
TRANSPORT_SELECTOR = SelectSelector(SelectSelectorConfig(options=SUPPORTED_TRANSPORTS, mode=SelectSelectorMode.DROPDOWN))
WS_HEADERS_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True))
CA_VERIFICATION_MODES = ['off', 'auto', 'custom']
BROKER_VERIFICATION_SELECTOR = SelectSelector(SelectSelectorConfig(options=CA_VERIFICATION_MODES, mode=SelectSelectorMode.DROPDOWN, translation_key=SET_CA_CERT))
CA_CERT_UPLOAD_SELECTOR = FileSelector(FileSelectorConfig(accept='.crt,application/x-x509-ca-cert'))
CERT_UPLOAD_SELECTOR = FileSelector(FileSelectorConfig(accept='.crt,application/x-x509-user-cert'))
KEY_UPLOAD_SELECTOR = FileSelector(FileSelectorConfig(accept='.key,application/pkcs8'))
REAUTH_SCHEMA = vol.Schema({vol.Required(CONF_USERNAME): TEXT_SELECTOR, vol.Required(CONF_PASSWORD): PASSWORD_SELECTOR})
PWD_NOT_CHANGED = '__**password_not_changed**__'

@callback
def update_password_from_user_input(entry_password: Optional[str], user_input: Dict[str, Any]) -> Dict[str, Any]:
    """Update the password if the entry has been updated."""
    substituted_used_data = dict(user_input)
    user_password = substituted_used_data.pop(CONF_PASSWORD, None)
    password_changed = user_password is not None and user_password != PWD_NOT_CHANGED
    password = user_password if password_changed else entry_password
    if password is not None:
        substituted_used_data[CONF_PASSWORD] = password
    return substituted_used_data

class FlowHandler(ConfigFlow, domain=DOMAIN):
    """Handle a config flow."""
    VERSION = CONFIG_ENTRY_VERSION
    MINOR_VERSION = CONFIG_ENTRY_MINOR_VERSION
    _hassio_discovery: Optional[Dict[str, Any]] = None
    _addon_manager: Optional[AddonManager] = None
    install_task: Optional[asyncio.Task[None]] = None
    start_task: Optional[asyncio.Task[None]] = None

    def __init__(self) -> None:
        """Set up flow instance."""
        self.install_task = None
        self.start_task = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow for this handler."""
        return MQTTOptionsFlowHandler()

    async def _async_install_addon(self) -> None:
        """Install the Mosquitto Mqtt broker add-on."""
        addon_manager = get_addon_manager(self.hass)
        await addon_manager.async_schedule_install_addon()

    async def async_step_install_failed(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Add-on installation failed."""
        return self.async_abort(reason='addon_install_failed', description_placeholders={'addon': self._addon_manager.addon_name})

    async def async_step_install_addon(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Install Mosquitto Broker add-on."""
        if self.install_task is None:
            self.install_task = self.hass.async_create_task(self._async_install_addon())
        if not self.install_task.done():
            return self.async_show_progress(step_id='install_addon', progress_action='install_addon', progress_task=self.install_task)
        try:
            await self.install_task
        except AddonError as err:
            _LOGGER.error(err)
            return self.async_show_progress_done(next_step_id='install_failed')
        finally:
            self.install_task = None
        return self.async_show_progress_done(next_step_id='start_addon')

    async def async_step_start_failed(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Add-on start failed."""
        return self.async_abort(reason='addon_start_failed', description_placeholders={'addon': self._addon_manager.addon_name})

    async def async_step_start_addon(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Start Mosquitto Broker add-on."""
        if not self.start_task:
            self.start_task = self.hass.async_create_task(self._async_start_addon())
        if not self.start_task.done():
            return self.async_show_progress(step_id='start_addon', progress_action='start_addon', progress_task=self.start_task)
        try:
            await self.start_task
        except AddonError as err:
            _LOGGER.error(err)
            return self.async_show_progress_done(next_step_id='start_failed')
        finally:
            self.start_task = None
        return self.async_show_progress_done(next_step_id='setup_entry_from_discovery')

    async def _async_get_config_and_try(self) -> Optional[Dict[str, Any]]:
        """Get the MQTT add-on discovery info and try the connection."""
        if self._hassio_discovery is not None:
            return self._hassio_discovery
        addon_manager = get_addon_manager(self.hass)
        try:
            addon_discovery_config = await addon_manager.async_get_addon_discovery_info()
            config = {CONF_BROKER: addon_discovery_config[CONF_HOST], CONF_PORT: addon_discovery_config[CONF_PORT], CONF_USERNAME: addon_discovery_config.get(CONF_USERNAME), CONF_PASSWORD: addon_discovery_config.get(CONF_PASSWORD), CONF_DISCOVERY: DEFAULT_DISCOVERY}
        except AddonError:
            return None
        if await self.hass.async_add_executor_job(try_connection, config):
            self._hassio_discovery = config
            return config
        return None

    async def _async_start_addon(self) -> None:
        """Start the Mosquitto Broker add-on."""
        addon_manager = get_addon_manager(self.hass)
        await addon_manager.async_schedule_start_addon()
        for _ in range(ADDON_SETUP_TIMEOUT_ROUNDS):
            await asyncio.sleep(ADDON_SETUP_TIMEOUT)
            if await self._async_get_config_and_try():
                break
        else:
            raise AddonError(translation_domain=DOMAIN, translation_key='addon_start_failed', translation_placeholders={'addon': addon_manager.addon_name})

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Handle a flow initialized by the user."""
        if is_hassio(self.hass):
            self._addon_manager = get_addon_manager(self.hass)
            return self.async_show_menu(step_id='user', menu_options=['addon', 'broker'], description_placeholders={'addon': self._addon_manager.addon_name})
        return await self.async_step_broker()

    async def async_step_setup_entry_from_discovery(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Set up mqtt entry from discovery info."""
        if (config := (await self._async_get_config_and_try())) is not None:
            return self.async_create_entry(title=self._addon_manager.addon_name, data=config)
        raise AbortFlow('addon_connection_failed', description_placeholders={'addon': self._addon_manager.addon_name})

    async def async_step_addon(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Install and start MQTT broker add-on."""
        addon_manager = self._addon_manager
        try:
            addon_info = await addon_manager.async_get_addon_info()
        except AddonError as err:
            raise AbortFlow('addon_info_failed', description_placeholders={'addon': self._addon_manager.addon_name}) from err
        if addon_info.state == AddonState.RUNNING:
            return await self.async_step_setup_entry_from_discovery()
        if addon_info.state == AddonState.NOT_RUNNING:
            return await self.async_step_start_addon()
        return await self.async_step_install_addon()

    async def async_step_reauth(self, entry_data: Dict[str, Any]) -> ConfigFlowResult:
        """Handle re-authentication with MQTT broker."""
        if is_hassio(self.hass):
            addon_manager = get_addon_manager(self.hass)
            try:
                addon_discovery_config = await addon_manager.async_get_addon_discovery_info()
            except AddonError:
                pass
            else:
                if entry_data[CONF_BROKER] == addon_discovery_config[CONF_HOST] and entry_data[CONF_PORT] == addon_discovery_config[CONF_PORT] and (entry_data.get(CONF_USERNAME) == (username := addon_discovery_config.get(CONF_USERNAME))) and (entry_data.get(CONF_PASSWORD) != (password := addon_discovery_config.get(CONF_PASSWORD))):
                    _LOGGER.info('Executing autorecovery %s add-on secrets', addon_manager.addon_name)
                    return await self.async_step_reauth_confirm(user_input={CONF_USERNAME: username, CONF_PASSWORD: password})
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Confirm re-authentication with MQTT broker."""
        errors: Dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()
        if user_input:
            substituted_used_data = update_password_from_user_input(reauth_entry.data.get(CONF_PASSWORD), user_input)
            new_entry_data = {**reauth_entry.data, **substituted_used_data}
            if await self.hass.async_add_executor_job(try_connection, new_entry_data):
                return self.async_update_reload_and_abort(reauth_entry, data=new_entry_data)
            errors['base'] = 'invalid_auth'
        schema = self.add_suggested_values_to_schema(REAUTH_SCHEMA, {CONF_USERNAME: reauth_entry.data.get(CONF_USERNAME), CONF_PASSWORD: PWD_NOT_CHANGED})
        return self.async_show_form(step_id='reauth_confirm', data_schema=schema, errors=errors)

    async def async_step_broker(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Confirm the setup."""
        errors: Dict[str, str] = {}
        fields = OrderedDict()
        validated_user_input: Dict[str, Any] = {}
        if (is_reconfigure := (self.source == SOURCE_RECONFIGURE)):
            reconfigure_entry = self._get_reconfigure_entry()
        if await async_get_broker_settings(self, fields, reconfigure_entry.data if is_reconfigure else None, user_input, validated_user_input, errors):
            if is_reconfigure:
                validated_user_input = update_password_from_user_input(reconfigure_entry.data.get(CONF_PASSWORD), validated_user_input)
            can_connect = await self.hass.async_add_executor_job(try_connection, validated_user_input)
            if can_connect:
                if is_reconfigure:
                    return self.async_update_reload_and_abort(reconfigure_entry, data=validated_user_input)
                return self.async_create_entry(title=validated_user_input[CONF_BROKER], data=validated_user_input)
            errors['base'] = 'cannot_connect'
        return self.async_show_form(step_id='broker', data_schema=vol.Schema(fields), errors=errors)

    async def async_step_reconfigure(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Handle a reconfiguration flow initialized by the user."""
        return await self.async_step_broker()

    async def async_step_hassio(self, discovery_info: HassioServiceInfo) -> ConfigFlowResult:
        """Receive a Hass.io discovery or process setup after addon install."""
        await self._async_handle_discovery_without_unique_id()
        self._hassio_discovery = discovery_info.config
        return await self.async_step_hassio_confirm()

    async def async_step_hassio_confirm(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Confirm a Hass.io discovery."""
        errors: Dict[str, str] = {}
        if TYPE_CHECKING:
            assert self._hassio_discovery
        if user_input is not None:
            data = self._hassio_discovery.copy()
            data[CONF_BROKER] = data.pop(CONF_HOST)
            can_connect = await self.hass.async_add_executor_job(try_connection, data)
            if can_connect:
                return self.async_create_entry(title=data['addon'], data={CONF_BROKER: data[CONF_BROKER], CONF_PORT: data[CONF_PORT], CONF_USERNAME: data.get(CONF_USERNAME), CONF_PASSWORD: data.get(CONF_PASSWORD), CONF_DISCOVERY: DEFAULT_DISCOVERY})
            errors['base'] = 'cannot_connect'
        return self.async_show_form(step_id='hassio_confirm', description_placeholders={'addon': self._hassio_discovery['addon']}, errors=errors)

class MQTTOptionsFlowHandler(OptionsFlow):
    """Handle MQTT options