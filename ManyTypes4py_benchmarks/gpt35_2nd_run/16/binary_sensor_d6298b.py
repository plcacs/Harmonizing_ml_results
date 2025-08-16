from __future__ import annotations
import logging
import voluptuous as vol
from homeassistant.components.binary_sensor import DEVICE_CLASSES_SCHEMA, PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA, BinarySensorEntity
from homeassistant.const import CONF_DEVICE_CLASS, CONF_DEVICES, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, event as evt
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_OFF_DELAY: str = 'off_delay'
PLATFORM_SCHEMA: vol.Schema = BINARY_SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_DEVICES): {cv.string: vol.Schema({vol.Optional(CONF_NAME): cv.string, vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA, vol.Optional(CONF_OFF_DELAY): vol.All(cv.time_period, cv.positive_timedelta)})}}, extra=vol.ALLOW_EXTRA)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    binary_sensors: list[W800rf32BinarySensor] = []
    for device_id, entity in config[CONF_DEVICES].items():
        _LOGGER.debug('Add %s w800rf32.binary_sensor (class %s)', entity[CONF_NAME], entity.get(CONF_DEVICE_CLASS))
        device: W800rf32BinarySensor = W800rf32BinarySensor(device_id, entity.get(CONF_NAME), entity.get(CONF_DEVICE_CLASS), entity.get(CONF_OFF_DELAY))
        binary_sensors.append(device)
    add_entities(binary_sensors)

class W800rf32BinarySensor(BinarySensorEntity):
    """A representation of a w800rf32 binary sensor."""
    _attr_should_poll: bool = False

    def __init__(self, device_id: str, name: str, device_class: str = None, off_delay: timedelta = None) -> None:
        self._signal: str = W800RF32_DEVICE.format(device_id)
        self._name: str = name
        self._device_class: str = device_class
        self._off_delay: timedelta = off_delay
        self._state: bool = False
        self._delay_listener: Optional[Callable] = None

    @callback
    def _off_delay_listener(self, now: datetime) -> None:
        self._delay_listener = None
        self.update_state(False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def device_class(self) -> str:
        return self._device_class

    @property
    def is_on(self) -> bool:
        return self._state

    @callback
    def binary_sensor_update(self, event: w800.W800rf32Event) -> None:
        if not isinstance(event, w800.W800rf32Event):
            return
        dev_id: str = event.device
        command: str = event.command
        _LOGGER.debug('BinarySensor update (Device ID: %s Command %s ...)', dev_id, command)
        if command in ('On', 'Off'):
            is_on: bool = command == 'On'
            self.update_state(is_on)
        if self.is_on and self._off_delay is not None and (self._delay_listener is None):
            self._delay_listener = evt.async_call_later(self.hass, self._off_delay, self._off_delay_listener)

    def update_state(self, state: bool) -> None:
        self._state = state
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(self.hass, self._signal, self.binary_sensor_update)
