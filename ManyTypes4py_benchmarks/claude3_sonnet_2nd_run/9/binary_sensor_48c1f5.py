"""Binary sensor platform integration for Numato USB GPIO expanders."""
from __future__ import annotations
from functools import partial
import logging
from typing import Any, Callable

from numato_gpio import NumatoGpioError
from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.const import DEVICE_DEFAULT_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import CONF_BINARY_SENSORS, CONF_DEVICES, CONF_ID, CONF_INVERT_LOGIC, CONF_PORTS, DATA_API, DOMAIN

_LOGGER = logging.getLogger(__name__)
NUMATO_SIGNAL = 'numato_signal_{}_{}'

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the configured Numato USB GPIO binary sensor ports."""
    if discovery_info is None:
        return

    def read_gpio(device_id: str, port: str, level: bool) -> None:
        """Send signal to entity to have it update state."""
        dispatcher_send(hass, NUMATO_SIGNAL.format(device_id, port), level)

    api = hass.data[DOMAIN][DATA_API]
    binary_sensors: list[NumatoGpioBinarySensor] = []
    devices = hass.data[DOMAIN][CONF_DEVICES]
    for device in [d for d in devices if CONF_BINARY_SENSORS in d]:
        device_id = device[CONF_ID]
        platform = device[CONF_BINARY_SENSORS]
        invert_logic = platform[CONF_INVERT_LOGIC]
        ports = platform[CONF_PORTS]
        for port, port_name in ports.items():
            try:
                api.setup_input(device_id, port)
            except NumatoGpioError as err:
                _LOGGER.error("Failed to initialize binary sensor '%s' on Numato device %s port %s: %s", port_name, device_id, port, err)
                continue
            try:
                api.edge_detect(device_id, port, partial(read_gpio, device_id))
            except NumatoGpioError as err:
                _LOGGER.error('Notification setup failed on device %s, updates on binary sensor %s only in polling mode: %s', device_id, port_name, err)
            binary_sensors.append(NumatoGpioBinarySensor(port_name, device_id, port, invert_logic, api))
    add_entities(binary_sensors, True)

class NumatoGpioBinarySensor(BinarySensorEntity):
    """Represents a binary sensor (input) port of a Numato GPIO expander."""
    _attr_should_poll = False

    def __init__(self, name: str | None, device_id: str, port: str, invert_logic: bool, api: Any) -> None:
        """Initialize the Numato GPIO based binary sensor object."""
        self._attr_name = name or DEVICE_DEFAULT_NAME
        self._device_id = device_id
        self._port = port
        self._invert_logic = invert_logic
        self._state: bool | None = None
        self._api = api

    async def async_added_to_hass(self) -> None:
        """Connect state update callback."""
        self.async_on_remove(async_dispatcher_connect(self.hass, NUMATO_SIGNAL.format(self._device_id, self._port), self._async_update_state))

    @callback
    def _async_update_state(self, level: bool) -> None:
        """Update entity state."""
        self._state = level
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool | None:
        """Return the state of the entity."""
        if self._state is None:
            return None
        return self._state != self._invert_logic

    def update(self) -> None:
        """Update the GPIO state."""
        try:
            self._state = self._api.read_input(self._device_id, self._port)
        except NumatoGpioError as err:
            self._state = None
            _LOGGER.error('Failed to update Numato device %s port %s: %s', self._device_id, self._port, err)
