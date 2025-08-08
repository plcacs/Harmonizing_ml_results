from __future__ import annotations
import asyncio
from functools import partial
import os
from typing import Any, Dict, List, Optional
from dsmr_parser import obis_references as obis_ref
from dsmr_parser.clients.protocol import create_dsmr_reader, create_tcp_dsmr_reader
from dsmr_parser.clients.rfxtrx_protocol import create_rfxtrx_dsmr_reader, create_rfxtrx_tcp_dsmr_reader
from dsmr_parser.objects import DSMRObject
import serial
import serial.tools.list_ports
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_PROTOCOL, CONF_TYPE
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from .const import CONF_DSMR_VERSION, CONF_SERIAL_ID, CONF_SERIAL_ID_GAS, CONF_TIME_BETWEEN_UPDATE, DEFAULT_TIME_BETWEEN_UPDATE, DOMAIN, DSMR_PROTOCOL, DSMR_VERSIONS, LOGGER, RFXTRX_DSMR_PROTOCOL
CONF_MANUAL_PATH: str = 'Enter Manually'

class DSMRConnection:
    def __init__(self, host: Optional[str], port: int, dsmr_version: str, protocol: str) -> None:
    def equipment_identifier(self) -> Optional[str]:
    def equipment_identifier_gas(self) -> Optional[str]:
    async def validate_connect(self, hass: HomeAssistant) -> bool:

async def _validate_dsmr_connection(hass: HomeAssistant, data: Dict[str, Any], protocol: str) -> Dict[str, str]:

class DSMRFlowHandler(ConfigFlow, domain=DOMAIN):
    VERSION: int = 1
    _dsmr_version: Optional[str] = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> DSMROptionFlowHandler:

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
    async def async_step_setup_network(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
    async def async_step_setup_serial(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
    async def async_step_setup_serial_manual_path(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:
    async def async_validate_dsmr(self, input_data: Dict[str, Any], errors: Dict[str, str]) -> Dict[str, Any]:

class DSMROptionFlowHandler(OptionsFlow):
    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None) -> ConfigFlowResult:

def get_serial_by_id(dev_path: str) -> str:

class CannotConnect(HomeAssistantError):
class CannotCommunicate(HomeAssistantError):
