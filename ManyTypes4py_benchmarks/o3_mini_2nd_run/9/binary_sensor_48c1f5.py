from __future__ import annotations
from functools import partial
import logging
from typing import Any, Callable, Dict, Optional, List
from numato_gpio import NumatoGpioError
from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.const import DEVICE_DEFAULT_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)
NUMATO_SIGNAL = 'numato_signal_{}_{}'

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the configured Numato USB GPIO binary sensor ports."""
    if discovery_info is None:
        return

    def read_gpio(device_id: str, port: int, level: bool) -> None:
        """Send signal to entity to have it update state."""
        dispatcher_send(hass, NUMATO_SIGNAL.format(device_id, port), level)

    api: Any = hass.data[DOMAIN]["DATA_API"]
    binary_sensors: List[NumatoGpioBinarySensor] = []
    devices: List[Dict[str, Any]] = hass.data[DOMAIN]["CONF_DEVICES"]
    for device in [d for d in devices if "CONF_BINARY_SENSORS" in d]:
        device_id: str = device["CONF_ID"]
        platform: Dict[str, Any] = device["CONF_BINARY_SENSORS"]
        invert_logic: bool = platform["CONF_INVERT_LOGIC"]
        ports: Dict[int, str] = platform["CONF_PORTS"]
        for port, port_name in ports.items():
            try:
                api.setup_input(device_id, port)
            except NumatoGpioError as err:
                _LOGGER.error(
                    "Failed to initialize binary sensor '%s' on Numato device %s port %s: %s",
                    port_name, device_id, port, err
                )
                continue
            try:
                api.edge_detect(device_id, port, partial(read_gpio, device_id))
            except NumatoGpioError as err:
                _LOGGER.error(
                    'Notification setup failed on device %s, updates on binary sensor %s only in polling mode: %s',
                    device_id, port_name, err
                )
            binary_sensors.append(
                NumatoGpioBinarySensor(port_name, device_id, port, invert_logic, api)
            )
    add_entities(binary_sensors, True)

class NumatoGpioBinarySensor(BinarySensorEntity):
    """Represents a binary sensor (input) port of a Numato GPIO expander."""
    _attr_should_poll: bool = False

    def __init__(self, name: str, device_id: str, port: int, invert_logic: bool, api: Any) -> None:
        """Initialize the Numato GPIO based binary sensor object."""
        self._attr_name: str = name or DEVICE_DEFAULT_NAME
        self._device_id: str = device_id
        self._port: int = port
        self._invert_logic: bool = invert_logic
        self._state: Optional[bool] = None
        self._api: Any = api

    async def async_added_to_hass(self) -> None:
        """Connect state update callback."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                NUMATO_SIGNAL.format(self._device_id, self._port),
                self._async_update_state
            )
        )

    @callback
    def _async_update_state(self, level: bool) -> None:
        """Update entity state."""
        self._state = level
        self.async_write_ha_state()

    @property
    def is_on(self) -> Optional[bool]:
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
            _LOGGER.error(
                'Failed to update Numato device %s port %s: %s',
                self._device_id, self._port, err
            )