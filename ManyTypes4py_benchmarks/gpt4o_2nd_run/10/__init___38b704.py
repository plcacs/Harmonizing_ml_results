"""Support for Lutron Homeworks Series 4 and 8 systems."""
from __future__ import annotations
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional
from pyhomeworks import exceptions as hw_exceptions
from pyhomeworks.pyhomeworks import HW_BUTTON_PRESSED, HW_BUTTON_RELEASED, HW_LOGIN_INCORRECT, Homeworks
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_ID, CONF_NAME, CONF_PASSWORD, CONF_PORT, CONF_USERNAME, EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ConfigEntryNotReady, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import slugify
from .const import CONF_ADDR, CONF_CONTROLLER_ID, CONF_KEYPADS, DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS: List[Platform] = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.LIGHT]

CONF_COMMAND = 'command'
EVENT_BUTTON_PRESS = 'homeworks_button_press'
EVENT_BUTTON_RELEASE = 'homeworks_button_release'
KEYPAD_LEDSTATE_POLL_COOLDOWN = 1.0

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

SERVICE_SEND_COMMAND_SCHEMA = vol.Schema({
    vol.Required(CONF_CONTROLLER_ID): str,
    vol.Required(CONF_COMMAND): vol.All(cv.ensure_list, [str])
})

@dataclass
class HomeworksData:
    """Container for config entry data."""
    controller: Homeworks
    controller_id: str
    keypads: Dict[str, 'HomeworksKeypad']

@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for Lutron Homeworks Series 4 and 8 integration."""

    async def async_call_service(service_call: ServiceCall) -> None:
        """Call the service."""
        await async_send_command(hass, service_call.data)

    hass.services.async_register(DOMAIN, 'send_command', async_call_service, schema=SERVICE_SEND_COMMAND_SCHEMA)

async def async_send_command(hass: HomeAssistant, data: Dict[str, Any]) -> None:
    """Send command to a controller."""

    def get_controller_ids() -> List[str]:
        """Get homeworks data for the specified controller ID."""
        return [data.controller_id for data in hass.data[DOMAIN].values()]

    def get_homeworks_data(controller_id: str) -> Optional[HomeworksData]:
        """Get homeworks data for the specified controller ID."""
        for data in hass.data[DOMAIN].values():
            if data.controller_id == controller_id:
                return data
        return None

    homeworks_data = get_homeworks_data(data[CONF_CONTROLLER_ID])
    if not homeworks_data:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key='invalid_controller_id',
            translation_placeholders={
                'controller_id': data[CONF_CONTROLLER_ID],
                'controller_ids': ','.join(get_controller_ids())
            }
        )

    commands = data[CONF_COMMAND]
    _LOGGER.debug('Send commands: %s', commands)
    for command in commands:
        if command.lower().startswith('delay'):
            delay = int(command.partition(' ')[2])
            _LOGGER.debug('Sleeping for %s ms', delay)
            await asyncio.sleep(delay / 1000)
        else:
            _LOGGER.debug("Sending command '%s'", command)
            await hass.async_add_executor_job(homeworks_data.controller._send, command)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Start Homeworks controller."""
    async_setup_services(hass)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Homeworks from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    controller_id = entry.options[CONF_CONTROLLER_ID]

    def hw_callback(msg_type: str, values: List[Any]) -> None:
        """Dispatch state changes."""
        _LOGGER.debug('callback: %s, %s', msg_type, values)
        if msg_type == HW_LOGIN_INCORRECT:
            _LOGGER.debug('login incorrect')
            return
        addr = values[0]
        signal = f'homeworks_entity_{controller_id}_{addr}'
        dispatcher_send(hass, signal, msg_type, values)

    config = entry.options
    controller = Homeworks(
        config[CONF_HOST],
        config[CONF_PORT],
        hw_callback,
        entry.data.get(CONF_USERNAME),
        entry.data.get(CONF_PASSWORD)
    )

    try:
        await hass.async_add_executor_job(controller.connect)
    except hw_exceptions.HomeworksException as err:
        _LOGGER.debug('Failed to connect: %s', err, exc_info=True)
        raise ConfigEntryNotReady from err

    controller.start()

    def cleanup(event: Event) -> None:
        controller.stop()

    entry.async_on_unload(hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, cleanup))

    keypads: Dict[str, HomeworksKeypad] = {}
    for key_config in config.get(CONF_KEYPADS, []):
        addr = key_config[CONF_ADDR]
        name = key_config[CONF_NAME]
        keypads[addr] = HomeworksKeypad(hass, controller, controller_id, addr, name)

    hass.data[DOMAIN][entry.entry_id] = HomeworksData(controller, controller_id, keypads)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(update_listener))
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if (unload_ok := (await hass.config_entries.async_unload_platforms(entry, PLATFORMS))):
        data = hass.data[DOMAIN].pop(entry.entry_id)
        for keypad in data.keypads.values():
            keypad.unsubscribe()
        await hass.async_add_executor_job(data.controller.stop)
    return unload_ok

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)

class HomeworksKeypad:
    """When you want signals instead of entities.

    Stateless sensors such as keypads are expected to generate an event
    instead of a sensor entity in hass.
    """

    def __init__(self, hass: HomeAssistant, controller: Homeworks, controller_id: str, addr: str, name: str) -> None:
        """Register callback that will be used for signals."""
        self._addr = addr
        self._controller = controller
        self._debouncer = Debouncer(
            hass,
            _LOGGER,
            cooldown=KEYPAD_LEDSTATE_POLL_COOLDOWN,
            immediate=False,
            function=self._request_keypad_led_states
        )
        self._hass = hass
        self._name = name
        self._id = slugify(self._name)
        signal = f'homeworks_entity_{controller_id}_{self._addr}'
        _LOGGER.debug('connecting %s', signal)
        self.unsubscribe = async_dispatcher_connect(self._hass, signal, self._update_callback)

    @callback
    def _update_callback(self, msg_type: str, values: List[Any]) -> None:
        """Fire events if button is pressed or released."""
        if msg_type == HW_BUTTON_PRESSED:
            event = EVENT_BUTTON_PRESS
        elif msg_type == HW_BUTTON_RELEASED:
            event = EVENT_BUTTON_RELEASE
        else:
            return
        data = {CONF_ID: self._id, CONF_NAME: self._name, 'button': values[1]}
        self._hass.bus.async_fire(event, data)

    def _request_keypad_led_states(self) -> None:
        """Query keypad led state."""
        self._controller._send(f'RKLS, {self._addr}')

    async def request_keypad_led_states(self) -> None:
        """Query keypad led state.

        Debounced to not storm the controller during setup.
        """
        await self._debouncer.async_call()
