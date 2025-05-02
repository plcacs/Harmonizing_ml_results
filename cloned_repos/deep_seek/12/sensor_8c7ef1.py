"""Support for LaCrosse sensor components."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Final, Optional, TypedDict, cast
import pylacrosse
from serial import SerialException
import voluptuous as vol
from homeassistant.components.sensor import (
    ENTITY_ID_FORMAT,
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import (
    CONF_DEVICE,
    CONF_ID,
    CONF_NAME,
    CONF_SENSORS,
    CONF_TYPE,
    EVENT_HOMEASSISTANT_STOP,
    PERCENTAGE,
    UnitOfTemperature,
)
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import async_generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER: Final = logging.getLogger(__name__)

CONF_BAUD: Final = 'baud'
CONF_DATARATE: Final = 'datarate'
CONF_EXPIRE_AFTER: Final = 'expire_after'
CONF_FREQUENCY: Final = 'frequency'
CONF_JEELINK_LED: Final = 'led'
CONF_TOGGLE_INTERVAL: Final = 'toggle_interval'
CONF_TOGGLE_MASK: Final = 'toggle_mask'
DEFAULT_DEVICE: Final = '/dev/ttyUSB0'
DEFAULT_BAUD: Final = 57600
DEFAULT_EXPIRE_AFTER: Final = 300
TYPES: Final = ['battery', 'humidity', 'temperature']

class SensorConfig(TypedDict, total=False):
    """TypedDict for sensor configuration."""
    id: int
    type: str
    expire_after: int
    name: str

SENSOR_SCHEMA: Final = vol.Schema({
    vol.Required(CONF_ID): cv.positive_int,
    vol.Required(CONF_TYPE): vol.In(TYPES),
    vol.Optional(CONF_EXPIRE_AFTER): cv.positive_int,
    vol.Optional(CONF_NAME): cv.string,
})

PLATFORM_SCHEMA: Final = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_SENSORS): cv.schema_with_slug_keys(SENSOR_SCHEMA),
    vol.Optional(CONF_BAUD, default=DEFAULT_BAUD): cv.positive_int,
    vol.Optional(CONF_DATARATE): cv.positive_int,
    vol.Optional(CONF_DEVICE, default=DEFAULT_DEVICE): cv.string,
    vol.Optional(CONF_FREQUENCY): cv.positive_int,
    vol.Optional(CONF_JEELINK_LED): cv.boolean,
    vol.Optional(CONF_TOGGLE_INTERVAL): cv.positive_int,
    vol.Optional(CONF_TOGGLE_MASK): cv.positive_int,
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the LaCrosse sensors."""
    usb_device: str = config[CONF_DEVICE]
    baud: int = config[CONF_BAUD]
    expire_after: Optional[int] = config.get(CONF_EXPIRE_AFTER)
    _LOGGER.debug('%s %s', usb_device, baud)
    
    try:
        lacrosse = pylacrosse.LaCrosse(usb_device, baud)
        lacrosse.open()
    except SerialException as exc:
        _LOGGER.warning('Unable to open serial port: %s', exc)
        return

    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, lambda event: lacrosse.close())

    if CONF_JEELINK_LED in config:
        lacrosse.led_mode_state(config[CONF_JEELINK_LED])
    if CONF_FREQUENCY in config:
        lacrosse.set_frequency(config[CONF_FREQUENCY])
    if CONF_DATARATE in config:
        lacrosse.set_datarate(config[CONF_DATARATE])
    if CONF_TOGGLE_INTERVAL in config:
        lacrosse.set_toggle_interval(config[CONF_TOGGLE_INTERVAL])
    if CONF_TOGGLE_MASK in config:
        lacrosse.set_toggle_mask(config[CONF_TOGGLE_MASK])

    lacrosse.start_scan()
    sensors: list[LaCrosseSensor] = []
    
    for device, device_config in config[CONF_SENSORS].items():
        _LOGGER.debug('%s %s', device, device_config)
        typ: str = device_config[CONF_TYPE]
        sensor_class = TYPE_CLASSES[typ]
        name: str = device_config.get(CONF_NAME, device)
        sensors.append(sensor_class(hass, lacrosse, device, name, expire_after, device_config))
    
    add_entities(sensors)

class LaCrosseSensor(SensorEntity):
    """Implementation of a Lacrosse sensor."""
    _temperature: Optional[float]
    _humidity: Optional[float]
    _low_battery: Optional[bool]
    _new_battery: Optional[bool]

    def __init__(
        self,
        hass: HomeAssistant,
        lacrosse: pylacrosse.LaCrosse,
        device_id: str,
        name: str,
        expire_after: Optional[int],
        config: SensorConfig,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.entity_id = async_generate_entity_id(ENTITY_ID_FORMAT, device_id, hass=hass)
        self._config = config
        self._expire_after = expire_after
        self._expiration_trigger: Optional[CALLBACK_TYPE] = None
        self._attr_name = name
        lacrosse.register_callback(int(self._config['id']), self._callback_lacrosse, None)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return {
            'low_battery': self._low_battery,
            'new_battery': self._new_battery,
        }

    def _callback_lacrosse(self, lacrosse_sensor: Any, user_data: Any) -> None:
        """Handle a function that is called from pylacrosse with new values."""
        if self._expire_after is not None and self._expire_after > 0:
            if self._expiration_trigger:
                self._expiration_trigger()
                self._expiration_trigger = None
            expiration_at = dt_util.utcnow() + timedelta(seconds=self._expire_after)
            self._expiration_trigger = async_track_point_in_utc_time(
                self.hass, self.value_is_expired, expiration_at
            )
        
        self._temperature = lacrosse_sensor.temperature
        self._humidity = lacrosse_sensor.humidity
        self._low_battery = lacrosse_sensor.low_battery
        self._new_battery = lacrosse_sensor.new_battery

    @callback
    def value_is_expired(self, *_: Any) -> None:
        """Triggered when value is expired."""
        self._expiration_trigger = None
        self.async_write_ha_state()

class LaCrosseTemperature(LaCrosseSensor):
    """Implementation of a Lacrosse temperature sensor."""
    _attr_device_class: Final = SensorDeviceClass.TEMPERATURE
    _attr_state_class: Final = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement: Final = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        return self._temperature

class LaCrosseHumidity(LaCrosseSensor):
    """Implementation of a Lacrosse humidity sensor."""
    _attr_native_unit_of_measurement: Final = PERCENTAGE
    _attr_state_class: Final = SensorStateClass.MEASUREMENT
    _attr_device_class: Final = SensorDeviceClass.HUMIDITY

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        return self._humidity

class LaCrosseBattery(LaCrosseSensor):
    """Implementation of a Lacrosse battery sensor."""

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the sensor."""
        if self._low_battery is None:
            return None
        return 'low' if self._low_battery else 'ok'

    @property
    def icon(self) -> str:
        """Icon to use in the frontend."""
        if self._low_battery is None:
            return 'mdi:battery-unknown'
        return 'mdi:battery-alert' if self._low_battery else 'mdi:battery'

TYPE_CLASSES: Final[Dict[str, type[LaCrosseSensor]]] = {
    'temperature': LaCrosseTemperature,
    'humidity': LaCrosseHumidity,
    'battery': LaCrosseBattery,
}
