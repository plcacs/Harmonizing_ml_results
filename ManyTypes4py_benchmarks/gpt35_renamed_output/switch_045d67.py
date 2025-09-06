from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any
import telnetlib
import voluptuous as vol
from homeassistant.components.switch import ENTITY_ID_FORMAT, PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_COMMAND_OFF, CONF_COMMAND_ON, CONF_COMMAND_STATE, CONF_NAME, CONF_PORT, CONF_RESOURCE, CONF_SWITCHES, CONF_TIMEOUT, CONF_VALUE_TEMPLATE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_PORT: int = 23
DEFAULT_TIMEOUT: float = 0.2
SWITCH_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_COMMAND_OFF): cv.string, vol.
    Required(CONF_COMMAND_ON): cv.string, vol.Required(CONF_RESOURCE): cv.
    string, vol.Optional(CONF_VALUE_TEMPLATE): cv.template, vol.Optional(
    CONF_COMMAND_STATE): cv.string, vol.Optional(CONF_NAME): cv.string, vol
    .Optional(CONF_PORT, default=DEFAULT_PORT): cv.port, vol.Optional(
    CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(float)})
PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend({vol.Required(CONF_SWITCHES
    ): cv.schema_with_slug_keys(SWITCH_SCHEMA)})
SCAN_INTERVAL: timedelta = timedelta(seconds=10)


def func_epc5hp7u(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Find and return switches controlled by telnet commands."""
    devices: dict = config[CONF_SWITCHES]
    switches: list = []
    for object_id, device_config in devices.items():
        switches.append(TelnetSwitch(object_id, device_config[CONF_RESOURCE
            ], device_config[CONF_PORT], device_config.get(CONF_NAME,
            object_id), device_config[CONF_COMMAND_ON], device_config[
            CONF_COMMAND_OFF], device_config.get(CONF_COMMAND_STATE),
            device_config.get(CONF_VALUE_TEMPLATE), device_config[
            CONF_TIMEOUT]))
    if not switches:
        _LOGGER.error('No switches added')
        return
    add_entities(switches)


class TelnetSwitch(SwitchEntity):
    """Representation of a switch that can be toggled using telnet commands."""

    def __init__(self, object_id: str, resource: str, port: int, friendly_name: str, command_on: str,
        command_off: str, command_state: str, value_template: Template, timeout: float) -> None:
        """Initialize the switch."""
        self.entity_id: str = ENTITY_ID_FORMAT.format(object_id)
        self._resource: str = resource
        self._port: int = port
        self._attr_name: str = friendly_name
        self._attr_is_on: bool = False
        self._command_on: str = command_on
        self._command_off: str = command_off
        self._command_state: str = command_state
        self._value_template: Template = value_template
        self._timeout: float = timeout
        self._attr_should_poll: bool = bool(command_state)
        self._attr_assumed_state: bool = bool(command_state is None)

    def func_ts7kqlxq(self, command: str) -> str:
        try:
            telnet: telnetlib.Telnet = telnetlib.Telnet(self._resource, self._port)
            telnet.write(command.encode('ASCII') + b'\r')
            response: bytes = telnet.read_until(b'\r', timeout=self._timeout)
        except OSError as error:
            _LOGGER.error('Command "%s" failed with exception: %s', command,
                repr(error))
            return None
        _LOGGER.debug('telnet response: %s', response.decode('ASCII').strip())
        return response.decode('ASCII').strip()

    def func_aix0lwr1(self) -> None:
        """Update device state."""
        if not self._command_state:
            return
        response: str = self._telnet_command(self._command_state)
        if response and self._value_template:
            rendered: str = self._value_template.render_with_possible_json_value(
                response)
        else:
            _LOGGER.warning('Empty response for command: %s', self.
                _command_state)
            return
        self._attr_is_on: bool = rendered == 'True'

    def func_ry29b0ic(self, **kwargs) -> None:
        """Turn the device on."""
        self._telnet_command(self._command_on)
        if self.assumed_state:
            self._attr_is_on: bool = True
            self.schedule_update_ha_state()

    def func_ovtwj15n(self, **kwargs) -> None:
        """Turn the device off."""
        self._telnet_command(self._command_off)
        if self.assumed_state:
            self._attr_is_on: bool = False
            self.schedule_update_ha_state()
