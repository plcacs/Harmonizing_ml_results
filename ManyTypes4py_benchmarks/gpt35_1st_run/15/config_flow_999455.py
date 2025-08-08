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
def update_password_from_user_input(entry_password, user_input) -> dict:
    substituted_used_data: dict = dict(user_input)
    user_password = substituted_used_data.pop(CONF_PASSWORD, None)
    password_changed = user_password is not None and user_password != PWD_NOT_CHANGED
    password = user_password if password_changed else entry_password
    if password is not None:
        substituted_used_data[CONF_PASSWORD] = password
    return substituted_used_data

class FlowHandler(ConfigFlow, domain=DOMAIN):
    VERSION: int = CONFIG_ENTRY_VERSION
    MINOR_VERSION: int = CONFIG_ENTRY_MINOR_VERSION
    _hassio_discovery: dict = None

    def __init__(self) -> None:
        self.install_task = None
        self.start_task = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry) -> MQTTOptionsFlowHandler:
        return MQTTOptionsFlowHandler()

    async def _async_install_addon(self) -> None:
        addon_manager = get_addon_manager(self.hass)
        await addon_manager.async_schedule_install_addon()

    async def async_step_install_failed(self, user_input=None) -> ConfigFlowResult:
        return self.async_abort(reason='addon_install_failed', description_placeholders={'addon': self._addon_manager.addon_name})

    async def async_step_install_addon(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_start_failed(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_start_addon(self, user_input=None) -> ConfigFlowResult:
        ...

    async def _async_get_config_and_try(self) -> dict:
        ...

    async def _async_start_addon(self) -> None:
        ...

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_setup_entry_from_discovery(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_addon(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_reauth(self, entry_data) -> ConfigFlowResult:
        ...

    async def async_step_reauth_confirm(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_broker(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_reconfigure(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_hassio(self, discovery_info) -> ConfigFlowResult:
        ...

    async def async_step_hassio_confirm(self, user_input=None) -> ConfigFlowResult:
        ...

class MQTTOptionsFlowHandler(OptionsFlow):
    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        ...

    async def async_step_options(self, user_input=None) -> ConfigFlowResult:
        ...

async def _get_uploaded_file(hass, id) -> str:
    ...

async def async_get_broker_settings(flow, fields, entry_config, user_input, validated_user_input, errors) -> bool:
    ...

def try_connection(user_input) -> bool:
    ...

def check_certicate_chain() -> str:
    ...
