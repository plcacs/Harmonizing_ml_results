"""Config flow for MQTT."""
from __future__ import annotations
import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping
import logging
import queue
from ssl import PROTOCOL_TLS_CLIENT, SSLContext, SSLError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
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
from typing import Any, Dict, List, Optional, Union

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
    """Update the password if the entry has been updated.

    As we want to avoid reflecting the stored password in the UI,
    we replace the suggested value in the UI with a sentitel,
    and we change it back here if it was changed.
    """
    substituted_used_data: Dict[str, Any] = dict(user_input)
    user_password = substituted_used_data.pop(CONF_PASSWORD, None)
    password_changed = user_password is not None and user_password != PWD_NOT_CHANGED
    password = user_password if password_changed else entry_password
    if password is not None:
        substituted_used_data[CONF_PASSWORD] = password
    return substituted_used_data

class FlowHandler(ConfigFlow, domain=DOMAIN):
    """Handle a config flow."""
    VERSION: str = CONFIG_ENTRY_VERSION
    MINOR_VERSION: str = CONFIG_ENTRY_MINOR_VERSION
    _hassio_discovery: Optional[Dict[str, Any]] = None

    def __init__(self) -> None:
        """Set up flow instance."""
        self.install_task: Optional[asyncio.Task] = None
        self.start_task: Optional[asyncio.Task] = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> MQTTOptionsFlowHandler:
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
            config: Dict[str, Any] = {CONF_BROKER: addon_discovery_config[CONF_HOST], CONF_PORT: addon_discovery_config[CONF_PORT], CONF_USERNAME: addon_discovery_config.get(CONF_USERNAME), CONF_PASSWORD: addon_discovery_config.get(CONF_PASSWORD), CONF_DISCOVERY: DEFAULT_DISCOVERY}
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
        errors: Dict[str, Any] = {}
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
        errors: Dict[str, Any] = {}
        fields: OrderedDict[str, Any] = OrderedDict()
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

    async def async_step_hassio(self, discovery_info: Dict[str, Any]) -> ConfigFlowResult:
        """Receive a Hass.io discovery or process setup after addon install."""
        await self._async_handle_discovery_without_unique_id()
        self._hassio_discovery = discovery_info.config
        return await self.async_step_hassio_confirm()

    async def async_step_hassio_confirm(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Confirm a Hass.io discovery."""
        errors: Dict[str, Any] = {}
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
    """Handle MQTT options."""

    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Manage the MQTT options."""
        return await self.async_step_options()

    async def async_step_options(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Manage the MQTT options."""
        errors: Dict[str, Any] = {}
        options_config: Dict[str, Any] = dict(self.config_entry.options)
        bad_input = False

        def _birth_will(birt_or_will: str) -> Dict[str, Any]:
            """Return the user input for birth or will."""
            if TYPE_CHECKING:
                assert user_input
            return {ATTR_TOPIC: user_input[f'{birt_or_will}_topic'], ATTR_PAYLOAD: user_input.get(f'{birt_or_will}_payload', ''), ATTR_QOS: user_input[f'{birt_or_will}_qos'], ATTR_RETAIN: user_input[f'{birt_or_will}_retain']}

        def _validate(field: str, values: Dict[str, Any], error_code: str, schema: Callable[[Dict[str, Any]], Any]) -> bool:
            """Validate the user input."""
            nonlocal bad_input
            try:
                option_values = schema(values)
                options_config[field] = option_values
            except vol.Invalid:
                errors['base'] = error_code
                bad_input = True
        if user_input is not None:
            options_config[CONF_DISCOVERY] = user_input[CONF_DISCOVERY]
            _validate(CONF_DISCOVERY_PREFIX, user_input[CONF_DISCOVERY_PREFIX], 'bad_discovery_prefix', valid_publish_topic)
            if 'birth_topic' in user_input:
                _validate(CONF_BIRTH_MESSAGE, _birth_will('birth'), 'bad_birth', valid_birth_will)
            if not user_input['birth_enable']:
                options_config[CONF_BIRTH_MESSAGE] = {}
            if 'will_topic' in user_input:
                _validate(CONF_WILL_MESSAGE, _birth_will('will'), 'bad_will', valid_birth_will)
            if not user_input['will_enable']:
                options_config[CONF_WILL_MESSAGE] = {}
            if not bad_input:
                return self.async_create_entry(data=options_config)
        birth = {**DEFAULT_BIRTH, **options_config.get(CONF_BIRTH_MESSAGE, {})}
        will = {**DEFAULT_WILL, **options_config.get(CONF_WILL_MESSAGE, {})}
        discovery = options_config.get(CONF_DISCOVERY, DEFAULT_DISCOVERY)
        discovery_prefix = options_config.get(CONF_DISCOVERY_PREFIX, DEFAULT_PREFIX)
        fields: OrderedDict[str, Any] = OrderedDict()
        fields[vol.Optional(CONF_DISCOVERY, default=discovery)] = BOOLEAN_SELECTOR
        fields[vol.Optional(CONF_DISCOVERY_PREFIX, default=discovery_prefix)] = PUBLISH_TOPIC_SELECTOR
        fields[vol.Optional('birth_enable', default=CONF_BIRTH_MESSAGE not in options_config or options_config[CONF_BIRTH_MESSAGE] != {})] = BOOLEAN_SELECTOR
        fields[vol.Optional('birth_topic', description={'suggested_value': birth[ATTR_TOPIC]})] = PUBLISH_TOPIC_SELECTOR
        fields[vol.Optional('birth_payload', description={'suggested_value': birth[CONF_PAYLOAD]})] = TEXT_SELECTOR
        fields[vol.Optional('birth_qos', default=birth[ATTR_QOS])] = QOS_SELECTOR
        fields[vol.Optional('birth_retain', default=birth[ATTR_RETAIN])] = BOOLEAN_SELECTOR
        fields[vol.Optional('will_enable', default=CONF_WILL_MESSAGE not in options_config or options_config[CONF_WILL_MESSAGE] != {})] = BOOLEAN_SELECTOR
        fields[vol.Optional('will_topic', description={'suggested_value': will[ATTR_TOPIC]})] = PUBLISH_TOPIC_SELECTOR
        fields[vol.Optional('will_payload', description={'suggested_value': will[CONF_PAYLOAD]})] = TEXT_SELECTOR
        fields[vol.Optional('will_qos', default=will[ATTR_QOS])] = QOS_SELECTOR
        fields[vol.Optional('will_retain', default=will[ATTR_RETAIN])] = BOOLEAN_SELECTOR
        return self.async_show_form(step_id='options', data_schema=vol.Schema(fields), errors=errors, last_step=True)

async def _get_uploaded_file(hass: HomeAssistant, id: str) -> str:
    """Get file content from uploaded file."""

    def _proces_uploaded_file() -> str:
        with process_uploaded_file(hass, id) as file_path:
            return file_path.read_text(encoding=DEFAULT_ENCODING)
    return await hass.async_add_executor_job(_proces_uploaded_file)

async def async_get_broker_settings(flow: FlowHandler, fields: OrderedDict[str, Any], entry_config: Optional[Dict[str, Any]], user_input: Optional[Dict[str, Any]], validated_user_input: Dict[str, Any], errors: Dict[str, Any]) -> bool:
    """Build the config flow schema to collect the broker settings.

    Shows advanced options if one or more are configured
    or when the advanced_broker_options checkbox was selected.
    Returns True when settings are collected successfully.
    """
    hass: HomeAssistant = flow.hass
    advanced_broker_options = False
    user_input_basic: Dict[str, Any] = {}
    current_config: Dict[str, Any] = entry_config.copy() if entry_config is not None else {}

    async def _async_validate_broker_settings(config: Dict[str, Any], user_input: Dict[str, Any], validated_user_input: Dict[str, Any], errors: Dict[str, Any]) -> bool:
        """Additional validation on broker settings for better error messages."""
        certificate = 'auto' if user_input.get(SET_CA_CERT, 'off') == 'auto' else config.get(CONF_CERTIFICATE) if user_input.get(SET_CA_CERT, 'off') == 'custom' else None
        client_certificate = config.get(CONF_CLIENT_CERT) if user_input.get(SET_CLIENT_CERT) else None
        client_key = config.get(CONF_CLIENT_KEY) if user_input.get(SET_CLIENT_CERT) else None
        validated_user_input.update(user_input)
        client_certificate_id = user_input.get(CONF_CLIENT_CERT)
        client_key_id = user_input.get(CONF_CLIENT_KEY)
        if client_certificate_id and (not client_key_id) or (not client_certificate_id and client_key_id):
            errors['base'] = 'invalid_inclusion'
            return False
        certificate_id = user_input.get(CONF_CERTIFICATE)
        if certificate_id:
            certificate = await _get_uploaded_file(hass, certificate_id)
        if not client_certificate and user_input.get(SET_CLIENT_CERT) and (not client_certificate_id) or (not certificate and user_input.get(SET_CA_CERT, 'off') == 'custom' and (not certificate_id)) or (user_input.get(CONF_TRANSPORT) == TRANSPORT_WEBSOCKETS and CONF_WS_PATH not in user_input):
            return False
        if client_certificate_id:
            client_certificate = await _get_uploaded_file(hass, client_certificate_id)
        if client_key_id:
            client_key = await _get_uploaded_file(hass, client_key_id)
        certificate_data: Dict[str, Any] = {}
        if certificate:
            certificate_data[CONF_CERTIFICATE] = certificate
        if client_certificate:
            certificate_data[CONF_CLIENT_CERT] = client_certificate
            certificate_data[CONF_CLIENT_KEY] = client_key
        validated_user_input.update(certificate_data)
        await async_create_certificate_temp_files(hass, certificate_data)
        if (error := (await hass.async_add_executor_job(check_certicate_chain))):
            errors['base'] = error
            return False
        if SET_CA_CERT in validated_user_input:
            del validated_user_input[SET_CA_CERT]
        if SET_CLIENT_CERT in validated_user_input:
            del validated_user_input[SET_CLIENT_CERT]
        if validated_user_input.get(CONF_TRANSPORT, TRANSPORT_TCP) == TRANSPORT_TCP:
            if CONF_WS_PATH in validated_user_input:
                del validated_user_input[CONF_WS_PATH]
            if CONF_WS_HEADERS in validated_user_input:
                del validated_user_input[CONF_WS_HEADERS]
            return True
        try:
            validated_user_input[CONF_WS_HEADERS] = json_loads(validated_user_input.get(CONF_WS_HEADERS, '{}'))
            schema = vol.Schema({cv.string: cv.template})
            schema(validated_user_input[CONF_WS_HEADERS])
        except (*JSON_DECODE_EXCEPTIONS, vol.MultipleInvalid):
            errors['base'] = 'bad_ws_headers'
            return False
        return True
    if user_input:
        user_input_basic = user_input.copy()
        advanced_broker_options = user_input_basic.get(ADVANCED_OPTIONS, False)
        if ADVANCED_OPTIONS not in user_input or advanced_broker_options is False:
            if await _async_validate_broker_settings(current_config, user_input_basic, validated_user_input, errors):
                return True
        current_broker = user_input_basic.get(CONF_BROKER)
        current_port = user_input_basic.get(CONF_PORT, DEFAULT_PORT)
        current_user = user_input_basic.get(CONF_USERNAME)
        current_pass = user_input_basic.get(CONF_PASSWORD)
    else:
        current_broker = current_config.get(CONF_BROKER)
        current_port = current_config.get(CONF_PORT, DEFAULT_PORT)
        current_user = current_config.get(CONF_USERNAME)
        current_entry_pass = current_config.get(CONF_PASSWORD)
        current_pass = PWD_NOT_CHANGED if current_entry_pass else None
    current_config.update(user_input_basic)
    current_client_id = current_config.get(CONF_CLIENT_ID)
    current_keepalive = current_config.get(CONF_KEEPALIVE, DEFAULT_KEEPALIVE)
    current_ca_certificate = current_config.get(CONF_CERTIFICATE)
    current_client_certificate = current_config.get(CONF_CLIENT_CERT)
    current_client_key = current_config.get(CONF_CLIENT_KEY)
    current_tls_insecure = current_config.get(CONF_TLS_INSECURE, False)
    current_protocol = current_config.get(CONF_PROTOCOL, DEFAULT_PROTOCOL)
    current_transport = current_config.get(CONF_TRANSPORT, DEFAULT_TRANSPORT)
    current_ws_path = current_config.get(CONF_WS_PATH, DEFAULT_WS_PATH)
    current_ws_headers = json_dumps(current_config.get(CONF_WS_HEADERS)) if CONF_WS_HEADERS in current_config else None
    advanced_broker_options |= bool(current_client_id or current_keepalive != DEFAULT_KEEPALIVE or current_ca_certificate or current_client_certificate or current_client_key or current_tls_insecure or (current_protocol != DEFAULT_PROTOCOL) or (current_config.get(SET_CA_CERT, 'off') != 'off') or current_config.get(SET_CLIENT_CERT) or (current_transport == TRANSPORT_WEBSOCKETS))
    fields[vol.Required(CONF_BROKER, default=current_broker)] = TEXT_SELECTOR
    fields[vol.Required(CONF_PORT, default=current_port)] = PORT_SELECTOR
    fields[vol.Optional(CONF_USERNAME, description={'suggested_value': current_user})] = TEXT_SELECTOR
    fields[vol.Optional(CONF_PASSWORD, description={'suggested_value': current_pass})] = PASSWORD_SELECTOR
    if not advanced_broker_options:
        if not flow.show_advanced_options:
            return False
        fields[vol.Optional(ADVANCED_OPTIONS)] = BOOLEAN_SELECTOR
        return False
    fields[vol.Optional(CONF_CLIENT_ID, description={'suggested_value': current_client_id})] = TEXT_SELECTOR
    fields[vol.Optional(CONF_KEEPALIVE, description={'suggested_value': current_keepalive})] = KEEPALIVE_SELECTOR
    fields[vol.Optional(SET_CLIENT_CERT, default=current_client_certificate is not None or current_config.get(SET_CLIENT_CERT) is True)] = BOOLEAN_SELECTOR
    if current_client_certificate is not None or current_config.get(SET_CLIENT_CERT) is True:
        fields[vol.Optional(CONF_CLIENT_CERT, description={'suggested_value': user_input_basic.get(CONF_CLIENT_CERT)})] = CERT_UPLOAD_SELECTOR
        fields[vol.Optional(CONF_CLIENT_KEY, description={'suggested_value': user_input_basic.get(CONF_CLIENT_KEY)})] = KEY_UPLOAD_SELECTOR
    verification_mode = current_config.get(SET_CA_CERT) or ('off' if current_ca_certificate is None else 'auto' if current_ca_certificate == 'auto' else 'custom')
    fields[vol.Optional(SET_CA_CERT, default=verification_mode)] = BROKER_VERIFICATION_SELECTOR
    if current_ca_certificate is not None or verification_mode == 'custom':
        fields[vol.Optional(CONF_CERTIFICATE, user_input_basic.get(CONF_CERTIFICATE))] = CA_CERT_UPLOAD_SELECTOR
    fields[vol.Optional(CONF_TLS_INSECURE, description={'suggested_value': current_tls_insecure})] = BOOLEAN_SELECTOR
    fields[vol.Optional(CONF_PROTOCOL, description={'suggested_value': current_protocol})] = PROTOCOL_SELECTOR
    fields[vol.Optional(CONF_TRANSPORT, description={'suggested_value': current_transport})] = TRANSPORT_SELECTOR
    if current_transport == TRANSPORT_WEBSOCKETS:
        fields[vol.Optional(CONF_WS_PATH, description={'suggested_value': current_ws_path})] = TEXT_SELECTOR
        fields[vol.Optional(CONF_WS_HEADERS, description={'suggested_value': current_ws_headers})] = WS_HEADERS_SELECTOR
    return False

def try_connection(user_input: Dict[str, Any]) -> bool:
    """Test if we can connect to an MQTT broker."""
    import paho.mqtt.client as mqtt
    mqtt_client_setup = MqttClientSetup(user_input)
    mqtt_client_setup.setup()
    client = mqtt_client_setup.client
    result = queue.Queue(maxsize=1)

    def on_connect(_mqttc, _userdata, _connect_flags, reason_code, _properties=None) -> None:
        """Handle connection result."""
        result.put(not reason_code.is_failure)
    client.on_connect = on_connect
    client.connect_async(user_input[CONF_BROKER], user_input[CONF_PORT])
    client.loop_start()
    try:
        return result.get(timeout=MQTT_TIMEOUT)
    except queue.Empty:
        return False
    finally:
        client.disconnect()
        client.loop_stop()

def check_certicate_chain() -> Optional[str]:
    """Check the MQTT certificates."""
    if (client_certificate := get_file_path(CONF_CLIENT_CERT)):
        try:
            with open(client_certificate, 'rb') as client_certificate_file:
                load_pem_x509_certificate(client_certificate_file.read())
        except ValueError:
            return 'bad_client_cert'
    if (private_key := get_file_path(CONF_CLIENT_KEY)):
        try:
            with open(private_key, 'rb') as client_key_file:
                load_pem_private_key(client_key_file.read(), password=None)
        except (TypeError, ValueError):
            return 'bad_client_key'
    context = SSLContext(PROTOCOL_TLS_CLIENT)
    if client_certificate and private_key:
        try:
            context.load_cert_chain(client_certificate, private_key)
        except SSLError:
            return 'bad_client_cert_key'
    if (ca_cert := get_file_path(CONF_CERTIFICATE)) is None:
        return None
    try:
        context.load_verify_locations(ca_cert)
    except SSLError:
        return 'bad_certificate'
    return None
