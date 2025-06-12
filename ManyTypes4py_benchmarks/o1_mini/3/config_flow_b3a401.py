"""Config flow for DSMR integration."""
from __future__ import annotations
import asyncio
from functools import partial
import os
from typing import Any, Callable, Dict, Optional
from dsmr_parser import obis_references as obis_ref
from dsmr_parser.clients.protocol import (
    create_dsmr_reader,
    create_tcp_dsmr_reader,
)
from dsmr_parser.clients.rfxtrx_protocol import (
    create_rfxtrx_dsmr_reader,
    create_rfxtrx_tcp_dsmr_reader,
)
from dsmr_parser.objects import DSMRObject
import serial
import serial.tools.list_ports
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_PROTOCOL, CONF_TYPE
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from .const import (
    CONF_DSMR_VERSION,
    CONF_SERIAL_ID,
    CONF_SERIAL_ID_GAS,
    CONF_TIME_BETWEEN_UPDATE,
    DEFAULT_TIME_BETWEEN_UPDATE,
    DOMAIN,
    DSMR_PROTOCOL,
    DSMR_VERSIONS,
    LOGGER,
    RFXTRX_DSMR_PROTOCOL,
)

CONF_MANUAL_PATH = 'Enter Manually'


class DSMRConnection:
    """Test the connection to DSMR and receive telegram to read serial ids."""

    def __init__(
        self,
        host: Optional[str],
        port: int,
        dsmr_version: str,
        protocol: str,
    ) -> None:
        """Initialize."""
        self._host: Optional[str] = host
        self._port: int = port
        self._dsmr_version: str = dsmr_version
        self._protocol: str = protocol
        self._telegram: Dict[str, DSMRObject] = {}
        self._equipment_identifier: str = obis_ref.EQUIPMENT_IDENTIFIER
        if dsmr_version == '5B':
            self._equipment_identifier = obis_ref.BELGIUM_EQUIPMENT_IDENTIFIER
        elif dsmr_version in ('5L', '5EONHU'):
            self._equipment_identifier = obis_ref.LUXEMBOURG_EQUIPMENT_IDENTIFIER
        elif dsmr_version == 'Q3D':
            self._equipment_identifier = obis_ref.Q3D_EQUIPMENT_IDENTIFIER

    def equipment_identifier(self) -> Optional[str]:
        """Equipment identifier."""
        if self._equipment_identifier in self._telegram:
            dsmr_object: DSMRObject = self._telegram[self._equipment_identifier]
            identifier: Optional[str] = getattr(dsmr_object, 'value', None)
            return identifier
        return None

    def equipment_identifier_gas(self) -> Optional[str]:
        """Equipment identifier gas."""
        if obis_ref.EQUIPMENT_IDENTIFIER_GAS in self._telegram:
            dsmr_object: DSMRObject = self._telegram[obis_ref.EQUIPMENT_IDENTIFIER_GAS]
            identifier: Optional[str] = getattr(dsmr_object, 'value', None)
            return identifier
        return None

    async def validate_connect(self, hass: HomeAssistant) -> bool:
        """Test if we can validate connection with the device."""

        def update_telegram(telegram: Dict[str, DSMRObject]) -> None:
            if self._equipment_identifier in telegram:
                self._telegram = telegram
                transport.close()
            if self._dsmr_version == '5S' and obis_ref.P1_MESSAGE_TIMESTAMP in telegram:
                self._telegram = telegram
                transport.close()

        if self._host is None:
            if self._protocol == DSMR_PROTOCOL:
                create_reader: Callable = create_dsmr_reader
            else:
                create_reader = create_rfxtrx_dsmr_reader
            reader_factory = partial(
                create_reader,
                self._port,
                self._dsmr_version,
                update_telegram,
                loop=hass.loop,
            )
        else:
            if self._protocol == DSMR_PROTOCOL:
                create_reader = create_tcp_dsmr_reader
            else:
                create_reader = create_rfxtrx_tcp_dsmr_reader
            reader_factory = partial(
                create_reader,
                self._host,
                self._port,
                self._dsmr_version,
                update_telegram,
                loop=hass.loop,
            )
        try:
            transport, protocol = await asyncio.create_task(reader_factory())
        except (serial.SerialException, OSError):
            LOGGER.exception('Error connecting to DSMR')
            return False
        if transport:
            try:
                async with asyncio.timeout(30):
                    await protocol.wait_closed()
            except TimeoutError:
                transport.close()
                await protocol.wait_closed()
        return True


async def _validate_dsmr_connection(
    hass: HomeAssistant, data: Dict[str, Any], protocol: str
) -> Dict[str, Any]:
    """Validate the user input allows us to connect."""
    conn = DSMRConnection(
        host=data.get(CONF_HOST),
        port=data[CONF_PORT],
        dsmr_version=data[CONF_DSMR_VERSION],
        protocol=protocol,
    )
    if not await conn.validate_connect(hass):
        raise CannotConnect
    equipment_identifier = conn.equipment_identifier()
    equipment_identifier_gas = conn.equipment_identifier_gas()
    if equipment_identifier is None and data[CONF_DSMR_VERSION] != '5S':
        raise CannotCommunicate
    return {
        CONF_SERIAL_ID: equipment_identifier,
        CONF_SERIAL_ID_GAS: equipment_identifier_gas,
    }


class DSMRFlowHandler(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for DSMR."""

    VERSION: int = 1
    _dsmr_version: Optional[str] = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> DSMROptionFlowHandler:
        """Get the options flow for this handler."""
        return DSMROptionFlowHandler()

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
        """Step when user initializes a integration."""
        if user_input is not None:
            user_selection: str = user_input[CONF_TYPE]
            if user_selection == 'Serial':
                return await self.async_step_setup_serial()
            return await self.async_step_setup_network()
        list_of_types: list[str] = ['Serial', 'Network']
        schema = vol.Schema({vol.Required(CONF_TYPE): vol.In(list_of_types)})
        return self.async_show_form(step_id='user', data_schema=schema)

    async def async_step_setup_network(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> ConfigFlowResult:
        """Step when setting up network configuration."""
        errors: Dict[str, str] = {}
        if user_input is not None:
            data: Dict[str, Any] = await self.async_validate_dsmr(user_input, errors)
            if not errors:
                return self.async_create_entry(
                    title=f'{data[CONF_HOST]}:{data[CONF_PORT]}', data=data
                )
        schema = vol.Schema(
            {
                vol.Required(CONF_HOST): str,
                vol.Required(CONF_PORT): int,
                vol.Required(CONF_DSMR_VERSION): vol.In(DSMR_VERSIONS),
            }
        )
        return self.async_show_form(
            step_id='setup_network', data_schema=schema, errors=errors
        )

    async def async_step_setup_serial(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> ConfigFlowResult:
        """Step when setting up serial configuration."""
        errors: Dict[str, str] = {}
        if user_input is not None:
            user_selection: str = user_input[CONF_PORT]
            if user_selection == CONF_MANUAL_PATH:
                self._dsmr_version = user_input[CONF_DSMR_VERSION]
                return await self.async_step_setup_serial_manual_path()
            dev_path: str = await self.hass.async_add_executor_job(
                get_serial_by_id, user_selection
            )
            validate_data: Dict[str, Any] = {
                CONF_PORT: dev_path,
                CONF_DSMR_VERSION: user_input[CONF_DSMR_VERSION],
            }
            data: Dict[str, Any] = await self.async_validate_dsmr(validate_data, errors)
            if not errors:
                return self.async_create_entry(title=data[CONF_PORT], data=data)
        ports: list[serial.tools.list_ports_common.ListPortInfo] = await self.hass.async_add_executor_job(
            serial.tools.list_ports.comports
        )
        list_of_ports: Dict[str, str] = {
            port.device: f'{port}, s/n: {port.serial_number or "n/a"}'
            + (f' - {port.manufacturer}' if port.manufacturer else '')
            for port in ports
        }
        list_of_ports[CONF_MANUAL_PATH] = CONF_MANUAL_PATH
        schema = vol.Schema(
            {
                vol.Required(CONF_PORT): vol.In(list_of_ports),
                vol.Required(CONF_DSMR_VERSION): vol.In(DSMR_VERSIONS),
            }
        )
        return self.async_show_form(
            step_id='setup_serial', data_schema=schema, errors=errors
        )

    async def async_step_setup_serial_manual_path(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> ConfigFlowResult:
        """Select path manually."""
        if user_input is not None:
            validate_data: Dict[str, Any] = {
                CONF_PORT: user_input[CONF_PORT],
                CONF_DSMR_VERSION: self._dsmr_version,
            }
            errors: Dict[str, str] = {}
            data: Dict[str, Any] = await self.async_validate_dsmr(validate_data, errors)
            if not errors:
                return self.async_create_entry(title=data[CONF_PORT], data=data)
        schema = vol.Schema({vol.Required(CONF_PORT): str})
        return self.async_show_form(
            step_id='setup_serial_manual_path', data_schema=schema
        )

    async def async_validate_dsmr(
        self, input_data: Dict[str, Any], errors: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate dsmr connection and create data."""
        data: Dict[str, Any] = input_data
        try:
            try:
                protocol: str = DSMR_PROTOCOL
                info: Dict[str, Any] = await _validate_dsmr_connection(
                    self.hass, data, protocol
                )
            except CannotCommunicate:
                protocol = RFXTRX_DSMR_PROTOCOL
                info = await _validate_dsmr_connection(self.hass, data, protocol)
            data = {**data, **info, CONF_PROTOCOL: protocol}
            if info.get(CONF_SERIAL_ID):
                await self.async_set_unique_id(info[CONF_SERIAL_ID])
                self._abort_if_unique_id_configured()
        except CannotConnect:
            errors['base'] = 'cannot_connect'
        except CannotCommunicate:
            errors['base'] = 'cannot_communicate'
        return data


class DSMROptionFlowHandler(OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry: ConfigEntry = config_entry

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title='', data=user_input)
        default_time: int = self.config_entry.options.get(
            CONF_TIME_BETWEEN_UPDATE, DEFAULT_TIME_BETWEEN_UPDATE
        )
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_TIME_BETWEEN_UPDATE,
                    default=default_time,
                ): vol.All(vol.Coerce(int), vol.Range(min=0)),
            }
        )
        return self.async_show_form(step_id='init', data_schema=schema)


def get_serial_by_id(dev_path: str) -> str:
    """Return a /dev/serial/by-id match for given device if available."""
    by_id = '/dev/serial/by-id'
    if not os.path.isdir(by_id):
        return dev_path
    for entry in os.scandir(by_id):
        if entry.is_symlink():
            path = entry.path
            if os.path.realpath(path) == dev_path:
                return path
    return dev_path


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class CannotCommunicate(HomeAssistantError):
    """Error to indicate we cannot communicate."""
