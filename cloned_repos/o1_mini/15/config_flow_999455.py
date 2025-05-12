"""Config flow for MQTT."""
from __future__ import annotations
import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping
import logging
import queue
from ssl import PROTOCOL_TLS_CLIENT, SSLContext, SSLError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate
import voluptuous as vol
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.components.hassio import AddonError, AddonManager, AddonState
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import (
    CONF_CLIENT_ID,
    CONF_DISCOVERY,
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PAYLOAD,
    CONF_PORT,
    CONF_PROTOCOL,
    CONF_USERNAME,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.hassio import is_hassio
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.selector import (
    BooleanSelector,
    FileSelector,
    FileSelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.service_info.hassio import HassioServiceInfo
from homeassistant.util.json import JSON_DECODE_EXCEPTIONS, json_loads
from .addon import get_addon_manager
from .client import MqttClientSetup
from .const import (
    ATTR_PAYLOAD,
    ATTR_QOS,
    ATTR_RETAIN,
    ATTR_TOPIC,
    CONF_BIRTH_MESSAGE,
    CONF_BROKER,
    CONF_CERTIFICATE,
    CONF_CLIENT_CERT,
    CONF_CLIENT_KEY,
    CONF_DISCOVERY_PREFIX,
    CONF_KEEPALIVE,
    CONF_TLS_INSECURE,
    CONF_TRANSPORT,
    CONF_WILL_MESSAGE,
    CONF_WS_HEADERS,
    CONF_WS_PATH,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DEFAULT_BIRTH,
    DEFAULT_DISCOVERY,
    DEFAULT_ENCODING,
    DEFAULT_KEEPALIVE,
    DEFAULT_PORT,
    DEFAULT_PREFIX,
    DEFAULT_PROTOCOL,
    DEFAULT_TRANSPORT,
    DEFAULT_WILL,
    DEFAULT_WS_PATH,
    DOMAIN,
    SUPPORTED_PROTOCOLS,
    TRANSPORT_TCP,
    TRANSPORT_WEBSOCKETS,
)
from .util import (
    async_create_certificate_temp_files,
    get_file_path,
    valid_birth_will,
    valid_publish_topic,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

ADDON_SETUP_TIMEOUT: int = 5
ADDON_SETUP_TIMEOUT_ROUNDS: int = 5
MQTT_TIMEOUT: int = 5
ADVANCED_OPTIONS: str = 'advanced_options'
SET_CA_CERT: str = 'set_ca_cert'
SET_CLIENT_CERT: str = 'set_client_cert'
BOOLEAN_SELECTOR: BooleanSelector = BooleanSelector()
TEXT_SELECTOR: TextSelector = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))
PUBLISH_TOPIC_SELECTOR: TextSelector = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))
PORT_SELECTOR: vol.Schema = vol.All(
    NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=1, max=65535)),
    vol.Coerce(int),
)
PASSWORD_SELECTOR: TextSelector = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
QOS_SELECTOR: vol.Schema = vol.All(
    NumberSelector(NumberSelectorConfig(mode=NumberSelectorMode.BOX, min=0, max=2)),
    vol.Coerce(int),
)
KEEPALIVE_SELECTOR: vol.Schema = vol.All(
    NumberSelector(
        NumberSelectorConfig(
            mode=NumberSelectorMode.BOX,
            min=15,
            step='any',
            unit_of_measurement='sec',
        )
    ),
    vol.Coerce(int),
)
PROTOCOL_SELECTOR: SelectSelector = SelectSelector(
    SelectSelectorConfig(options=SUPPORTED_PROTOCOLS, mode=SelectSelectorMode.DROPDOWN)
)
SUPPORTED_TRANSPORTS: list[SelectOptionDict] = [
    SelectOptionDict(value=TRANSPORT_TCP, label='TCP'),
    SelectOptionDict(value=TRANSPORT_WEBSOCKETS, label='WebSocket'),
]
TRANSPORT_SELECTOR: SelectSelector = SelectSelector(
    SelectSelectorConfig(options=SUPPORTED_TRANSPORTS, mode=SelectSelectorMode.DROPDOWN)
)
WS_HEADERS_SELECTOR: TextSelector = TextSelector(
    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
)
CA_VERIFICATION_MODES: list[str] = ['off', 'auto', 'custom']
BROKER_VERIFICATION_SELECTOR: SelectSelector = SelectSelector(
    SelectSelectorConfig(
        options=CA_VERIFICATION_MODES,
        mode=SelectSelectorMode.DROPDOWN,
        translation_key=SET_CA_CERT,
    )
)
CA_CERT_UPLOAD_SELECTOR: FileSelector = FileSelector(
    FileSelectorConfig(accept='.crt,application/x-x509-ca-cert')
)
CERT_UPLOAD_SELECTOR: FileSelector = FileSelector(
    FileSelectorConfig(accept='.crt,application/x-x509-user-cert')
)
KEY_UPLOAD_SELECTOR: FileSelector = FileSelector(
    FileSelectorConfig(accept='.key,application/pkcs8')
)
REAUTH_SCHEMA: vol.Schema = vol.Schema(
    {
        vol.Required(CONF_USERNAME): TEXT_SELECTOR,
        vol.Required(CONF_PASSWORD): PASSWORD_SELECTOR,
    }
)
PWD_NOT_CHANGED: str = '__**password_not_changed**__'


@callback
def update_password_from_user_input(
    entry_password: Optional[str], user_input: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Update the password if the entry has been updated.

    As we want to avoid reflecting the stored password in the UI,
    we replace the suggested value in the UI with a sentinel,
    and we change it back here if it was changed.
    """
    substituted_used_data: Dict[str, Any] = dict(user_input)
    user_password: Optional[str] = substituted_used_data.pop(CONF_PASSWORD, None)
    password_changed: bool = user_password is not None and user_password != PWD_NOT_CHANGED
    password: Optional[str] = user_password if password_changed else entry_password
    if password is not None:
        substituted_used_data[CONF_PASSWORD] = password
    return substituted_used_data


class FlowHandler(ConfigFlow, domain=DOMAIN):
    """Handle a config flow."""

    VERSION: int = CONFIG_ENTRY_VERSION
    MINOR_VERSION: int = CONFIG_ENTRY_MINOR_VERSION
    _hassio_discovery: Optional[HassioServiceInfo] = None

    def __init__(self) -> None:
        """Set up flow instance."""
        self.install_task: Optional[asyncio.Task[None]] = None
        self.start_task: Optional[asyncio.Task[None]] = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> MQTTOptionsFlowHandler:
        """Get the options flow for this handler."""
        return MQTTOptionsFlowHandler()

    async def _async_install_addon(self) -> None:
        """Install the Mosquitto Mqtt broker add-on."""
        addon_manager: AddonManager = get_addon_manager(self.hass)
        await addon_manager.async_schedule_install_addon()

    async def async_step_install_failed(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> AbortFlow:
        """Add-on installation failed."""
        return self.async_abort(
            reason='addon_install_failed',
            description_placeholders={'addon': self._addon_manager.addon_name},
        )

    async def async_step_install_addon(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, AbortFlow]:
        """Install Mosquitto Broker add-on."""
        if self.install_task is None:
            self.install_task = self.hass.async_create_task(self._async_install_addon())
        if not self.install_task.done():
            return self.async_show_progress(
                step_id='install_addon',
                progress_action='install_addon',
                progress_task=self.install_task,
            )
        try:
            await self.install_task
        except AddonError as err:
            _LOGGER.error(err)
            return self.async_show_progress_done(next_step_id='install_failed')
        finally:
            self.install_task = None
        return self.async_show_progress_done(next_step_id='start_addon')

    async def async_step_start_failed(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> AbortFlow:
        """Add-on start failed."""
        return self.async_abort(
            reason='addon_start_failed',
            description_placeholders={'addon': self._addon_manager.addon_name},
        )

    async def async_step_start_addon(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, AbortFlow]:
        """Start Mosquitto Broker add-on."""
        if not self.start_task:
            self.start_task = self.hass.async_create_task(self._async_start_addon())
        if not self.start_task.done():
            return self.async_show_progress(
                step_id='start_addon',
                progress_action='start_addon',
                progress_task=self.start_task,
            )
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
            return self._hassio_discovery.config  # type: ignore
        addon_manager: AddonManager = get_addon_manager(self.hass)
        try:
            addon_discovery_config: Dict[str, Any] = await addon_manager.async_get_addon_discovery_info()
            config: Dict[str, Any] = {
                CONF_BROKER: addon_discovery_config[CONF_HOST],
                CONF_PORT: addon_discovery_config[CONF_PORT],
                CONF_USERNAME: addon_discovery_config.get(CONF_USERNAME),
                CONF_PASSWORD: addon_discovery_config.get(CONF_PASSWORD),
                CONF_DISCOVERY: DEFAULT_DISCOVERY,
            }
        except AddonError:
            return None
        if await self.hass.async_add_executor_job(try_connection, config):
            self._hassio_discovery = addon_discovery_config  # type: ignore
            return config
        return None

    async def _async_start_addon(self) -> None:
        """Start the Mosquitto Broker add-on."""
        addon_manager: AddonManager = get_addon_manager(self.hass)
        await addon_manager.async_schedule_start_addon()
        for _ in range(ADDON_SETUP_TIMEOUT_ROUNDS):
            await asyncio.sleep(ADDON_SETUP_TIMEOUT)
            if await self._async_get_config_and_try():
                break
        else:
            raise AddonError(
                translation_domain=DOMAIN,
                translation_key='addon_start_failed',
                translation_placeholders={'addon': addon_manager.addon_name},
            )

    async def async_step_user(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, Awaitable[AbortFlow], ConfigFlowResult]:
        """Handle a flow initialized by the user."""
        if is_hassio(self.hass):
            self._addon_manager = get_addon_manager(self.hass)
            return self.async_show_menu(
                step_id='user',
                menu_options=['addon', 'broker'],
                description_placeholders={'addon': self._addon_manager.addon_name},
            )
        return await self.async_step_broker()

    async def async_step_setup_entry_from_discovery(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, AbortFlow]:
        """Set up mqtt entry from discovery info."""
        config: Optional[Dict[str, Any]] = await self._async_get_config_and_try()
        if config is not None:
            return self.async_create_entry(title=self._addon_manager.addon_name, data=config)
        raise AbortFlow(
            'addon_connection_failed',
            description_placeholders={'addon': self._addon_manager.addon_name},
        )

    async def async_step_addon(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, AbortFlow, ConfigFlowResult]:
        """Install and start MQTT broker add-on."""
        addon_manager: AddonManager = self._addon_manager
        try:
            addon_info = await addon_manager.async_get_addon_info()
        except AddonError as err:
            raise AbortFlow(
                'addon_info_failed',
                description_placeholders={'addon': self._addon_manager.addon_name},
            ) from err
        if addon_info.state == AddonState.RUNNING:
            return await self.async_step_setup_entry_from_discovery()
        if addon_info.state == AddonState.NOT_RUNNING:
            return await self.async_step_start_addon()
        return await self.async_step_install_addon()

    async def async_step_reauth(
        self, entry_data: Dict[str, Any]
    ) -> Union[ConfigFlowResult, ConfigFlowResult]:
        """Handle re-authentication with MQTT broker."""
        if is_hassio(self.hass):
            addon_manager: AddonManager = get_addon_manager(self.hass)
            try:
                addon_discovery_config: Dict[str, Any] = await addon_manager.async_get_addon_discovery_info()
            except AddonError:
                pass
            else:
                if (
                    entry_data[CONF_BROKER] == addon_discovery_config[CONF_HOST]
                    and entry_data[CONF_PORT] == addon_discovery_config[CONF_PORT]
                    and entry_data.get(CONF_USERNAME) == (username := addon_discovery_config.get(CONF_USERNAME))
                    and entry_data.get(CONF_PASSWORD) != (password := addon_discovery_config.get(CONF_PASSWORD))
                ):
                    _LOGGER.info(
                        'Executing autorecovery %s add-on secrets', addon_manager.addon_name
                    )
                    return await self.async_step_reauth_confirm(
                        user_input={CONF_USERNAME: username, CONF_PASSWORD: password}
                    )
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, ConfigFlowResult]:
        """Confirm re-authentication with MQTT broker."""
        errors: Dict[str, str] = {}
        reauth_entry: ConfigEntry = self._get_reauth_entry()
        if user_input:
            substituted_used_data: Mapping[str, Any] = update_password_from_user_input(
                reauth_entry.data.get(CONF_PASSWORD), user_input
            )
            new_entry_data: Dict[str, Any] = {**reauth_entry.data, **substituted_used_data}
            if await self.hass.async_add_executor_job(try_connection, new_entry_data):
                return self.async_update_reload_and_abort(reauth_entry, data=new_entry_data)
            errors['base'] = 'invalid_auth'
        schema: vol.Schema = self.add_suggested_values_to_schema(
            REAUTH_SCHEMA,
            {CONF_USERNAME: reauth_entry.data.get(CONF_USERNAME), CONF_PASSWORD: PWD_NOT_CHANGED},
        )
        return self.async_show_form(
            step_id='reauth_confirm', data_schema=schema, errors=errors
        )

    async def async_step_broker(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, Awaitable[Union[ConfigFlowResult, None]], ConfigFlowResult]:
        """Confirm the setup."""
        errors: Dict[str, str] = {}
        fields: OrderedDict[str, Any] = OrderedDict()
        validated_user_input: Dict[str, Any] = {}
        is_reconfigure: bool = self.source == SOURCE_RECONFIGURE
        reconfigure_entry: Optional[ConfigEntry] = self._get_reconfigure_entry() if is_reconfigure else None
        if await async_get_broker_settings(
            self, fields, reconfigure_entry.data if is_reconfigure else None, user_input, validated_user_input, errors
        ):
            if is_reconfigure and reconfigure_entry:
                validated_user_input = update_password_from_user_input(
                    reconfigure_entry.data.get(CONF_PASSWORD), validated_user_input
                )
            can_connect: bool = await self.hass.async_add_executor_job(
                try_connection, validated_user_input
            )
            if can_connect:
                if is_reconfigure and reconfigure_entry:
                    return self.async_update_reload_and_abort(
                        reconfigure_entry, data=validated_user_input
                    )
                return self.async_create_entry(
                    title=validated_user_input[CONF_BROKER],
                    data=validated_user_input,
                )
            errors['base'] = 'cannot_connect'
        return self.async_show_form(
            step_id='broker', data_schema=vol.Schema(fields), errors=errors
        )

    async def async_step_reconfigure(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, Awaitable[ConfigFlowResult]]:
        """Handle a reconfiguration flow initialized by the user."""
        return await self.async_step_broker()

    async def async_step_hassio(
        self, discovery_info: HassioServiceInfo
    ) -> ConfigFlowResult:
        """Receive a Hass.io discovery or process setup after addon install."""
        await self._async_handle_discovery_without_unique_id()
        self._hassio_discovery = discovery_info
        return await self.async_step_hassio_confirm()

    async def async_step_hassio_confirm(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> Union[ConfigFlowResult, ConfigFlowResult]:
        """Confirm a Hass.io discovery."""
        errors: Dict[str, str] = {}
        if TYPE_CHECKING:
            assert self._hassio_discovery is not None
        if user_input is not None:
            data: Dict[str, Any] = self._hassio_discovery.config.copy()
            data[CONF_BROKER] = data.pop(CONF_HOST)
            can_connect: bool = await self.hass.async_add_executor_job(try_connection, data)
            if can_connect:
                return self.async_create_entry(
                    title=data['addon'],
                    data={
                        CONF_BROKER: data[CONF_BROKER],
                        CONF_PORT: data[CONF_PORT],
                        CONF_USERNAME: data.get(CONF_USERNAME),
                        CONF_PASSWORD: data.get(CONF_PASSWORD),
                        CONF_DISCOVERY: DEFAULT_DISCOVERY,
                    },
                )
            errors['base'] = 'cannot_connect'
        return self.async_show_form(
            step_id='hassio_confirm',
            description_placeholders={'addon': self._hassio_discovery.config.get('addon')},
            errors=errors,
        )


class MQTTOptionsFlowHandler(OptionsFlow):
    """Handle MQTT options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize MQTT options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> ConfigFlowResult:
        """Manage the MQTT options."""
        return await self.async_step_options()

    async def async_step_options(
        self, user_input: Optional[Mapping[str, Any]] = None
    ) -> ConfigFlowResult:
        """Manage the MQTT options."""
        errors: Dict[str, str] = {}
        options_config: Dict[str, Any] = dict(self.config_entry.options)
        bad_input: bool = False

        async def _async_validate_broker_settings(
            config: Dict[str, Any],
            user_input_inner: Dict[str, Any],
            validated_user_input_inner: Dict[str, Any],
            errors_inner: Dict[str, str],
        ) -> bool:
            """Additional validation on broker settings for better error messages."""
            certificate: Optional[str] = (
                'auto'
                if user_input_inner.get(SET_CA_CERT, 'off') == 'auto'
                else config.get(CONF_CERTIFICATE)
                if user_input_inner.get(SET_CA_CERT, 'off') == 'custom'
                else None
            )
            client_certificate: Optional[str] = (
                config.get(CONF_CLIENT_CERT) if user_input_inner.get(SET_CLIENT_CERT) else None
            )
            client_key: Optional[str] = (
                config.get(CONF_CLIENT_KEY) if user_input_inner.get(SET_CLIENT_CERT) else None
            )
            validated_user_input_inner.update(user_input_inner)
            client_certificate_id: Optional[str] = user_input_inner.get(CONF_CLIENT_CERT)
            client_key_id: Optional[str] = user_input_inner.get(CONF_CLIENT_KEY)
            if (client_certificate_id and not client_key_id) or (
                not client_certificate_id and client_key_id
            ):
                errors_inner['base'] = 'invalid_inclusion'
                return False
            certificate_id: Optional[str] = user_input_inner.get(CONF_CERTIFICATE)
            if certificate_id:
                certificate = await _get_uploaded_file(self.hass, certificate_id)
            if (
                not client_certificate
                and user_input_inner.get(SET_CLIENT_CERT)
                and not client_certificate_id
                or (
                    not certificate
                    and user_input_inner.get(SET_CA_CERT, 'off') == 'custom'
                    and not certificate_id
                )
                or (
                    user_input_inner.get(CONF_TRANSPORT) == TRANSPORT_WEBSOCKETS
                    and CONF_WS_PATH not in user_input_inner
                )
            ):
                return False
            if client_certificate_id:
                client_certificate = await _get_uploaded_file(self.hass, client_certificate_id)
            if client_key_id:
                client_key = await _get_uploaded_file(self.hass, client_key_id)
            certificate_data: Dict[str, str] = {}
            if certificate:
                certificate_data[CONF_CERTIFICATE] = certificate
            if client_certificate:
                certificate_data[CONF_CLIENT_CERT] = client_certificate
                certificate_data[CONF_CLIENT_KEY] = client_key
            validated_user_input_inner.update(certificate_data)
            await async_create_certificate_temp_files(self.hass, certificate_data)
            error: Optional[str] = await self.hass.async_add_executor_job(check_certicate_chain)
            if error:
                errors_inner['base'] = error
                return False
            if SET_CA_CERT in validated_user_input_inner:
                del validated_user_input_inner[SET_CA_CERT]
            if SET_CLIENT_CERT in validated_user_input_inner:
                del validated_user_input_inner[SET_CLIENT_CERT]
            if validated_user_input_inner.get(CONF_TRANSPORT, TRANSPORT_TCP) == TRANSPORT_TCP:
                if CONF_WS_PATH in validated_user_input_inner:
                    del validated_user_input_inner[CONF_WS_PATH]
                if CONF_WS_HEADERS in validated_user_input_inner:
                    del validated_user_input_inner[CONF_WS_HEADERS]
                return True
            try:
                validated_user_input_inner[CONF_WS_HEADERS] = json_loads(
                    validated_user_input_inner.get(CONF_WS_HEADERS, '{}')
                )
                schema = vol.Schema({cv.string: cv.template})
                schema(validated_user_input_inner[CONF_WS_HEADERS])
            except (tuple(JSON_DECODE_EXCEPTIONS), vol.MultipleInvalid):
                errors_inner['base'] = 'bad_ws_headers'
                return False
            return True

        if user_input is not None:
            user_input_basic: Dict[str, Any] = dict(user_input)
            advanced_broker_options: bool = user_input_basic.get(ADVANCED_OPTIONS, False)
            if ADVANCED_OPTIONS not in user_input or not advanced_broker_options:
                if await _async_validate_broker_settings(
                    current_config=options_config,
                    user_input_inner=user_input_basic,
                    validated_user_input_inner=validated_user_input,
                    errors_inner=errors,
                ):
                    return self.async_create_entry(data=options_config)
            current_broker: Optional[str] = user_input_basic.get(CONF_BROKER)
            current_port: int = user_input_basic.get(CONF_PORT, DEFAULT_PORT)
            current_user: Optional[str] = user_input_basic.get(CONF_USERNAME)
            current_pass: Optional[str] = user_input_basic.get(CONF_PASSWORD)
        else:
            current_broker: Optional[str] = options_config.get(CONF_BROKER)
            current_port: int = options_config.get(CONF_PORT, DEFAULT_PORT)
            current_user: Optional[str] = options_config.get(CONF_USERNAME)
            current_entry_pass: Optional[str] = options_config.get(CONF_PASSWORD)
            current_pass: Optional[str] = PWD_NOT_CHANGED if current_entry_pass else None

        current_config: Dict[str, Any] = options_config.copy()
        if user_input:
            current_config.update(user_input)
        current_client_id: Optional[str] = current_config.get(CONF_CLIENT_ID)
        current_keepalive: int = current_config.get(CONF_KEEPALIVE, DEFAULT_KEEPALIVE)
        current_ca_certificate: Optional[str] = current_config.get(CONF_CERTIFICATE)
        current_client_certificate: Optional[str] = current_config.get(CONF_CLIENT_CERT)
        current_client_key: Optional[str] = current_config.get(CONF_CLIENT_KEY)
        current_tls_insecure: bool = current_config.get(CONF_TLS_INSECURE, False)
        current_protocol: str = current_config.get(CONF_PROTOCOL, DEFAULT_PROTOCOL)
        current_transport: str = current_config.get(CONF_TRANSPORT, DEFAULT_TRANSPORT)
        current_ws_path: str = current_config.get(CONF_WS_PATH, DEFAULT_WS_PATH)
        current_ws_headers: Optional[str] = (
            json_dumps(current_config.get(CONF_WS_HEADERS)) if CONF_WS_HEADERS in current_config else None
        )
        advanced_broker_options: bool = bool(
            current_client_id
            or current_keepalive != DEFAULT_KEEPALIVE
            or current_ca_certificate
            or current_client_certificate
            or current_client_key
            or current_tls_insecure
            or (current_protocol != DEFAULT_PROTOCOL)
            or (current_config.get(SET_CA_CERT, 'off') != 'off')
            or current_config.get(SET_CLIENT_CERT)
            or (current_transport == TRANSPORT_WEBSOCKETS)
        )
        fields[vol.Required(CONF_BROKER, default=current_broker)] = TEXT_SELECTOR
        fields[vol.Required(CONF_PORT, default=current_port)] = PORT_SELECTOR
        fields[vol.Optional(CONF_USERNAME, description={'suggested_value': current_user})] = TEXT_SELECTOR
        fields[vol.Optional(CONF_PASSWORD, description={'suggested_value': current_pass})] = PASSWORD_SELECTOR
        if not advanced_broker_options:
            if not self.show_advanced_options:
                return False  # type: ignore
            fields[vol.Optional(ADVANCED_OPTIONS)] = BOOLEAN_SELECTOR
            return False  # type: ignore
        fields[vol.Optional(CONF_CLIENT_ID, description={'suggested_value': current_client_id})] = TEXT_SELECTOR
        fields[vol.Optional(CONF_KEEPALIVE, description={'suggested_value': current_keepalive})] = KEEPALIVE_SELECTOR
        fields[vol.Optional(SET_CLIENT_CERT, default=(current_client_certificate is not None or options_config.get(SET_CLIENT_CERT) is True))] = BOOLEAN_SELECTOR
        if current_client_certificate is not None or options_config.get(SET_CLIENT_CERT) is True:
            fields[vol.Optional(CONF_CLIENT_CERT, description={'suggested_value': user_input.get(CONF_CLIENT_CERT) if user_input else None})] = CERT_UPLOAD_SELECTOR
            fields[vol.Optional(CONF_CLIENT_KEY, description={'suggested_value': user_input.get(CONF_CLIENT_KEY) if user_input else None})] = KEY_UPLOAD_SELECTOR
        verification_mode: str = (
            current_config.get(SET_CA_CERT)
            or ('off' if current_ca_certificate is None else 'auto' if current_ca_certificate == 'auto' else 'custom')
        )
        fields[vol.Optional(SET_CA_CERT, default=verification_mode)] = BROKER_VERIFICATION_SELECTOR
        if current_ca_certificate is not None or verification_mode == 'custom':
            fields[vol.Optional(CONF_CERTIFICATE, user_input.get(CONF_CERTIFICATE) if user_input else None)] = CA_CERT_UPLOAD_SELECTOR
        fields[vol.Optional(CONF_TLS_INSECURE, description={'suggested_value': current_tls_insecure})] = BOOLEAN_SELECTOR
        fields[vol.Optional(CONF_PROTOCOL, description={'suggested_value': current_protocol})] = PROTOCOL_SELECTOR
        fields[vol.Optional(CONF_TRANSPORT, description={'suggested_value': current_transport})] = TRANSPORT_SELECTOR
        if current_transport == TRANSPORT_WEBSOCKETS:
            fields[vol.Optional(CONF_WS_PATH, description={'suggested_value': current_ws_path})] = PUBLISH_TOPIC_SELECTOR
            fields[vol.Optional(CONF_WS_HEADERS, description={'suggested_value': current_ws_headers})] = WS_HEADERS_SELECTOR
        return self.async_show_form(
            step_id='options', data_schema=vol.Schema(fields), errors=errors, last_step=True
        )


async def _get_uploaded_file(hass: HomeAssistant, id: str) -> str:
    """Get file content from uploaded file."""

    def _proces_uploaded_file() -> str:
        with process_uploaded_file(hass, id) as file_path:
            return file_path.read_text(encoding=DEFAULT_ENCODING)

    return await hass.async_add_executor_job(_proces_uploaded_file)


async def async_get_broker_settings(
    flow: FlowHandler,
    fields: OrderedDict[str, Any],
    entry_config: Optional[Dict[str, Any]],
    user_input: Optional[Mapping[str, Any]],
    validated_user_input: Dict[str, Any],
    errors: Dict[str, str],
) -> bool:
    """Build the config flow schema to collect the broker settings.

    Shows advanced options if one or more are configured
    or when the advanced_broker_options checkbox was selected.
    Returns True when settings are collected successfully.
    """
    hass: HomeAssistant = flow.hass
    advanced_broker_options: bool = False
    user_input_basic: Dict[str, Any] = {}
    current_config: Dict[str, Any] = entry_config.copy() if entry_config is not None else {}

    async def _async_validate_broker_settings_inner(
        config: Dict[str, Any],
        user_input_inner: Dict[str, Any],
        validated_user_input_inner: Dict[str, Any],
        errors_inner: Dict[str, str],
    ) -> bool:
        """Additional validation on broker settings for better error messages."""
        certificate: Optional[str] = (
            'auto'
            if user_input_inner.get(SET_CA_CERT, 'off') == 'auto'
            else config.get(CONF_CERTIFICATE)
            if user_input_inner.get(SET_CA_CERT, 'off') == 'custom'
            else None
        )
        client_certificate: Optional[str] = (
            config.get(CONF_CLIENT_CERT) if user_input_inner.get(SET_CLIENT_CERT) else None
        )
        client_key: Optional[str] = (
            config.get(CONF_CLIENT_KEY) if user_input_inner.get(SET_CLIENT_CERT) else None
        )
        validated_user_input_inner.update(user_input_inner)
        client_certificate_id: Optional[str] = user_input_inner.get(CONF_CLIENT_CERT)
        client_key_id: Optional[str] = user_input_inner.get(CONF_CLIENT_KEY)
        if (client_certificate_id and not client_key_id) or (not client_certificate_id and client_key_id):
            errors_inner['base'] = 'invalid_inclusion'
            return False
        certificate_id: Optional[str] = user_input_inner.get(CONF_CERTIFICATE)
        if certificate_id:
            certificate = await _get_uploaded_file(hass, certificate_id)
        if (
            not client_certificate
            and user_input_inner.get(SET_CLIENT_CERT)
            and not client_certificate_id
            or (
                not certificate
                and user_input_inner.get(SET_CA_CERT, 'off') == 'custom'
                and not certificate_id
            )
            or (
                user_input_inner.get(CONF_TRANSPORT) == TRANSPORT_WEBSOCKETS
                and CONF_WS_PATH not in user_input_inner
            )
        ):
            return False
        if client_certificate_id:
            client_certificate = await _get_uploaded_file(hass, client_certificate_id)
        if client_key_id:
            client_key = await _get_uploaded_file(hass, client_key_id)
        certificate_data: Dict[str, str] = {}
        if certificate:
            certificate_data[CONF_CERTIFICATE] = certificate
        if client_certificate:
            certificate_data[CONF_CLIENT_CERT] = client_certificate
            certificate_data[CONF_CLIENT_KEY] = client_key
        validated_user_input_inner.update(certificate_data)
        await async_create_certificate_temp_files(hass, certificate_data)
        error: Optional[str] = await hass.async_add_executor_job(check_certicate_chain)
        if error:
            errors_inner['base'] = error
            return False
        if SET_CA_CERT in validated_user_input_inner:
            del validated_user_input_inner[SET_CA_CERT]
        if SET_CLIENT_CERT in validated_user_input_inner:
            del validated_user_input_inner[SET_CLIENT_CERT]
        if validated_user_input_inner.get(CONF_TRANSPORT, TRANSPORT_TCP) == TRANSPORT_TCP:
            if CONF_WS_PATH in validated_user_input_inner:
                del validated_user_input_inner[CONF_WS_PATH]
            if CONF_WS_HEADERS in validated_user_input_inner:
                del validated_user_input_inner[CONF_WS_HEADERS]
            return True
        try:
            validated_user_input_inner[CONF_WS_HEADERS] = json_loads(
                validated_user_input_inner.get(CONF_WS_HEADERS, '{}')
            )
            schema = vol.Schema({cv.string: cv.template})
            schema(validated_user_input_inner[CONF_WS_HEADERS])
        except (tuple(JSON_DECODE_EXCEPTIONS), vol.MultipleInvalid):
            errors_inner['base'] = 'bad_ws_headers'
            return False
        return True

    if user_input:
        user_input_basic = dict(user_input)
        advanced_broker_options = user_input_basic.get(ADVANCED_OPTIONS, False)
        if ADVANCED_OPTIONS not in user_input or not advanced_broker_options:
            if await _async_validate_broker_settings_inner(
                config=current_config,
                user_input_inner=user_input_basic,
                validated_user_input_inner=validated_user_input,
                errors_inner=errors,
            ):
                return True
        current_broker: Optional[str] = user_input_basic.get(CONF_BROKER)
        current_port: int = user_input_basic.get(CONF_PORT, DEFAULT_PORT)
        current_user: Optional[str] = user_input_basic.get(CONF_USERNAME)
        current_pass: Optional[str] = user_input_basic.get(CONF_PASSWORD)
    else:
        current_broker = current_config.get(CONF_BROKER)
        current_port = current_config.get(CONF_PORT, DEFAULT_PORT)
        current_user = current_config.get(CONF_USERNAME)
        current_entry_pass = current_config.get(CONF_PASSWORD)
        current_pass = PWD_NOT_CHANGED if current_entry_pass else None

    current_config.update(user_input_basic)
    current_client_id: Optional[str] = current_config.get(CONF_CLIENT_ID)
    current_keepalive: int = current_config.get(CONF_KEEPALIVE, DEFAULT_KEEPALIVE)
    current_ca_certificate: Optional[str] = current_config.get(CONF_CERTIFICATE)
    current_client_certificate: Optional[str] = current_config.get(CONF_CLIENT_CERT)
    current_client_key: Optional[str] = current_config.get(CONF_CLIENT_KEY)
    current_tls_insecure: bool = current_config.get(CONF_TLS_INSECURE, False)
    current_protocol: str = current_config.get(CONF_PROTOCOL, DEFAULT_PROTOCOL)
    current_transport: str = current_config.get(CONF_TRANSPORT, DEFAULT_TRANSPORT)
    current_ws_path: str = current_config.get(CONF_WS_PATH, DEFAULT_WS_PATH)
    current_ws_headers: Optional[str] = (
        json_dumps(current_config.get(CONF_WS_HEADERS)) if CONF_WS_HEADERS in current_config else None
    )
    advanced_broker_options |= bool(
        current_client_id
        or current_keepalive != DEFAULT_KEEPALIVE
        or current_ca_certificate
        or current_client_certificate
        or current_client_key
        or current_tls_insecure
        or (current_protocol != DEFAULT_PROTOCOL)
        or (current_config.get(SET_CA_CERT, 'off') != 'off')
        or current_config.get(SET_CLIENT_CERT)
        or (current_transport == TRANSPORT_WEBSOCKETS)
    )
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
    fields[
        vol.Optional(
            SET_CLIENT_CERT, default=(current_client_certificate is not None or current_config.get(SET_CLIENT_CERT) is True)
        )
    ] = BOOLEAN_SELECTOR
    if current_client_certificate is not None or current_config.get(SET_CLIENT_CERT) is True:
        fields[
            vol.Optional(
                CONF_CLIENT_CERT,
                description={'suggested_value': user_input_basic.get(CONF_CLIENT_CERT) if user_input else None},
            )
        ] = CERT_UPLOAD_SELECTOR
        fields[
            vol.Optional(
                CONF_CLIENT_KEY,
                description={'suggested_value': user_input_basic.get(CONF_CLIENT_KEY) if user_input else None},
            )
        ] = KEY_UPLOAD_SELECTOR
    verification_mode: str = (
        current_config.get(SET_CA_CERT)
        or ('off' if current_ca_certificate is None else 'auto' if current_ca_certificate == 'auto' else 'custom')
    )
    fields[vol.Optional(SET_CA_CERT, default=verification_mode)] = BROKER_VERIFICATION_SELECTOR
    if current_ca_certificate is not None or verification_mode == 'custom':
        fields[
            vol.Optional(
                CONF_CERTIFICATE,
                user_input_basic.get(CONF_CERTIFICATE) if user_input else None,
            )
        ] = CA_CERT_UPLOAD_SELECTOR
    fields[
        vol.Optional(CONF_TLS_INSECURE, description={'suggested_value': current_tls_insecure})
    ] = BOOLEAN_SELECTOR
    fields[
        vol.Optional(CONF_PROTOCOL, description={'suggested_value': current_protocol})
    ] = PROTOCOL_SELECTOR
    fields[
        vol.Optional(CONF_TRANSPORT, description={'suggested_value': current_transport})
    ] = TRANSPORT_SELECTOR
    if current_transport == TRANSPORT_WEBSOCKETS:
        fields[
            vol.Optional(
                CONF_WS_PATH, description={'suggested_value': current_ws_path}
            )
        ] = PUBLISH_TOPIC_SELECTOR
        fields[
            vol.Optional(
                CONF_WS_HEADERS, description={'suggested_value': current_ws_headers}
            )
        ] = WS_HEADERS_SELECTOR
    return False


def try_connection(user_input: Dict[str, Any]) -> bool:
    """Test if we can connect to an MQTT broker."""
    import paho.mqtt.client as mqtt

    mqtt_client_setup: MqttClientSetup = MqttClientSetup(user_input)
    mqtt_client_setup.setup()
    client: mqtt.Client = mqtt_client_setup.client
    result: queue.Queue[bool] = queue.Queue(maxsize=1)

    def on_connect(
        _mqttc: mqtt.Client,
        _userdata: Any,
        _connect_flags: Any,
        reason_code: mqtt.connack_string,
        _properties: Optional[dict] = None,
    ) -> None:
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
    client_certificate: Optional[str] = get_file_path(CONF_CLIENT_CERT)
    if client_certificate:
        try:
            with open(client_certificate, 'rb') as client_certificate_file:
                load_pem_x509_certificate(client_certificate_file.read())
        except ValueError:
            return 'bad_client_cert'
    client_key: Optional[str] = get_file_path(CONF_CLIENT_KEY)
    if client_key:
        try:
            with open(client_key, 'rb') as client_key_file:
                load_pem_private_key(client_key_file.read(), password=None)
        except (TypeError, ValueError):
            return 'bad_client_key'
    context: SSLContext = SSLContext(PROTOCOL_TLS_CLIENT)
    if client_certificate and client_key:
        try:
            context.load_cert_chain(client_certificate, client_key)
        except SSLError:
            return 'bad_client_cert_key'
    ca_cert: Optional[str] = get_file_path(CONF_CERTIFICATE)
    if ca_cert is None:
        return None
    try:
        context.load_verify_locations(ca_cert)
    except SSLError:
        return 'bad_certificate'
    return None
