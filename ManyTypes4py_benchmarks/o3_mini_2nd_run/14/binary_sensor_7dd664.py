from __future__ import annotations
import datetime
from typing import Any, Callable, Dict
import voluptuous as vol
from homeassistant.components.binary_sensor import PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA, BinarySensorEntity
from homeassistant.const import CONF_DISARM_AFTER_TRIGGER, CONF_NAME, CONF_PAYLOAD, CONF_PAYLOAD_OFF, CONF_PAYLOAD_ON
from homeassistant.core import HomeAssistant, Event
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import track_point_in_time
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from . import EVENT

CONF_VARIABLE = 'variable'
CONF_RESET_DELAY_SEC = 'reset_delay_sec'
DEFAULT_NAME = 'Pilight Binary Sensor'
PLATFORM_SCHEMA = BINARY_SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_VARIABLE): cv.string,
    vol.Required(CONF_PAYLOAD): vol.Schema(dict),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PAYLOAD_ON, default='on'): vol.Any(cv.positive_int, cv.small_float, cv.string),
    vol.Optional(CONF_PAYLOAD_OFF, default='off'): vol.Any(cv.positive_int, cv.small_float, cv.string),
    vol.Optional(CONF_DISARM_AFTER_TRIGGER, default=False): cv.boolean,
    vol.Optional(CONF_RESET_DELAY_SEC, default=30): cv.positive_int
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    if config.get(CONF_DISARM_AFTER_TRIGGER):
        add_entities([
            PilightTriggerSensor(
                hass=hass,
                name=config.get(CONF_NAME),
                variable=config.get(CONF_VARIABLE),
                payload=config.get(CONF_PAYLOAD),
                on_value=config.get(CONF_PAYLOAD_ON),
                off_value=config.get(CONF_PAYLOAD_OFF),
                rst_dly_sec=config.get(CONF_RESET_DELAY_SEC)
            )
        ])
    else:
        add_entities([
            PilightBinarySensor(
                hass=hass,
                name=config.get(CONF_NAME),
                variable=config.get(CONF_VARIABLE),
                payload=config.get(CONF_PAYLOAD),
                on_value=config.get(CONF_PAYLOAD_ON),
                off_value=config.get(CONF_PAYLOAD_OFF)
            )
        ])

class PilightBinarySensor(BinarySensorEntity):
    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        variable: str,
        payload: Dict[str, Any],
        on_value: Any,
        off_value: Any
    ) -> None:
        self._state: bool = False
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._variable: str = variable
        self._payload: Dict[str, Any] = payload
        self._on_value: Any = on_value
        self._off_value: Any = off_value
        hass.bus.listen(EVENT, self._handle_code)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_on(self) -> bool:
        return self._state

    def _handle_code(self, call: Event) -> None:
        payload_ok: bool = True
        for key in self._payload:
            if key not in call.data:
                payload_ok = False
                continue
            if self._payload[key] != call.data[key]:
                payload_ok = False
        if payload_ok:
            if self._variable not in call.data:
                return
            value: Any = call.data[self._variable]
            self._state = value == self._on_value
            self.schedule_update_ha_state()

class PilightTriggerSensor(BinarySensorEntity):
    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        variable: str,
        payload: Dict[str, Any],
        on_value: Any,
        off_value: Any,
        rst_dly_sec: int = 30
    ) -> None:
        self._state: bool = False
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._variable: str = variable
        self._payload: Dict[str, Any] = payload
        self._on_value: Any = on_value
        self._off_value: Any = off_value
        self._reset_delay_sec: int = rst_dly_sec
        self._delay_after: datetime.datetime | None = None
        hass.bus.listen(EVENT, self._handle_code)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_on(self) -> bool:
        return self._state

    def _reset_state(self, call: Any) -> None:
        self._state = False
        self._delay_after = None
        self.schedule_update_ha_state()

    def _handle_code(self, call: Event) -> None:
        payload_ok: bool = True
        for key in self._payload:
            if key not in call.data:
                payload_ok = False
                continue
            if self._payload[key] != call.data[key]:
                payload_ok = False
        if payload_ok:
            if self._variable not in call.data:
                return
            value: Any = call.data[self._variable]
            self._state = value == self._on_value
            if self._delay_after is None:
                self._delay_after = dt_util.utcnow() + datetime.timedelta(seconds=self._reset_delay_sec)
                track_point_in_time(self._hass, self._reset_state, self._delay_after)
            self.schedule_update_ha_state()