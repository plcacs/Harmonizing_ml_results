"""Support for LaCrosse sensor components."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Callable, Dict, Optional
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

_LOGGER = logging.getLogger(__name__)

CONF_BAUD = "baud"
CONF_DATARATE = "datarate"
CONF_EXPIRE_AFTER = "expire_after"
CONF_FREQUENCY = "frequency"
CONF_JEELINK_LED = "led"
CONF_TOGGLE_INTERVAL = "toggle_interval"
CONF_TOGGLE_MASK = "toggle_mask"

DEFAULT_DEVICE = "/dev/ttyUSB0"
DEFAULT_BAUD = 57600
DEFAULT_EXPIRE_AFTER = 300

TYPES = ["battery", "humidity", "temperature"]

SENSOR_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ID): cv.positive_int,
        vol.Required(CONF_TYPE): vol.In(TYPES),
        vol.Optional(CONF_EXPIRE_AFTER): cv.positive_int,
        vol.Optional(CONF_NAME): cv.string,
    }
)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_SENSORS): cv.schema_with_slug_keys(SENSOR_SCHEMA),
        vol.Optional(CONF_BAUD, default=DEFAULT_BAUD): cv.positive_int,
        vol.Optional(CONF_DATARATE): cv.positive_int,
        vol.Optional(CONF_DEVICE, default=DEFAULT_DEVICE): cv.string,
        vol.Optional(CONF_FREQUENCY): cv.positive_int,
        vol.Optional(CONF_JEELINK_LED): cv.boolean,
        vol.Optional(CONF_TOGGLE_INTERVAL): cv.positive_int,
        vol.Optional(CONF_TOGGLE_MASK): cv.positive_int,
    }
)


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
    _LOGGER.debug("%s %s", usb_device, baud)
    try:
        lacrosse: pylacrosse.LaCrosse = pylacrosse.LaCrosse(usb_device, baud)
        lacrosse.open()
    except SerialException as exc:
        _LOGGER.warning("Unable to open serial port: %s", exc)
        return

    hass.bus.listen_once(
        EVENT_HOMEASSISTANT_STOP, lambda event: lacrosse.close()  # type: ignore
    )

    if CONF_JEELINK_LED in config:
        lacrosse.led_mode_state(config.get(CONF_JEELINK_LED))  # type: ignore
    if CONF_FREQUENCY in config:
        lacrosse.set_frequency(config.get(CONF_FREQUENCY))  # type: ignore
    if CONF_DATARATE in config:
        lacrosse.set_datarate(config.get(CONF_DATARATE))  # type: ignore
    if CONF_TOGGLE_INTERVAL in config:
        lacrosse.set_toggle_interval(config.get(CONF_TOGGLE_INTERVAL))  # type: ignore
    if CONF_TOGGLE_MASK in config:
        lacrosse.set_toggle_mask(config.get(CONF_TOGGLE_MASK))  # type: ignore

    lacrosse.start_scan()

    sensors: list[LaCrosseSensor] = []
    for device, device_config in config[CONF_SENSORS].items():  # type: ignore
        _LOGGER.debug("%s %s", device, device_config)
        typ: str = device_config[CONF_TYPE]
        sensor_class = TYPE_CLASSES[typ]
        name: str = device_config.get(CONF_NAME, device)
        sensors.append(
            sensor_class(
                hass,
                lacrosse,
                device,
                name,
                expire_after,
                device_config,
            )
        )
    add_entities(sensors)


class LaCrosseSensor(SensorEntity):
    """Implementation of a Lacrosse sensor."""

    _temperature: Optional[float] = None
    _humidity: Optional[float] = None
    _low_battery: Optional[bool] = None
    _new_battery: Optional[bool] = None

    def __init__(
        self,
        hass: HomeAssistant,
        lacrosse: pylacrosse.LaCrosse,
        device_id: str,
        name: str,
        expire_after: Optional[int],
        config: Dict[str, Any],
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.entity_id: str = async_generate_entity_id(
            ENTITY_ID_FORMAT, device_id, hass=hass
        )
        self._config: Dict[str, Any] = config
        self._expire_after: Optional[int] = expire_after
        self._expiration_trigger: Optional[Callable[..., None]] = None
        self._attr_name: str = name
        lacrosse.register_callback(
            int(self._config["id"]), self._callback_lacrosse, None
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Optional[bool]]:
        """Return the state attributes."""
        return {"low_battery": self._low_battery, "new_battery": self._new_battery}

    def _callback_lacrosse(
        self, lacrosse_sensor: pylacrosse.LaCrosseSensor, user_data: Any
    ) -> None:
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
    def value_is_expired(self, _: datetime) -> None:
        """Triggered when value is expired."""
        self._expiration_trigger = None
        self.async_write_ha_state()


class LaCrosseTemperature(LaCrosseSensor):
    """Implementation of a Lacrosse temperature sensor."""

    _attr_device_class: Optional[SensorDeviceClass] = SensorDeviceClass.TEMPERATURE
    _attr_state_class: Optional[SensorStateClass] = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement: Optional[str] = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        return self._temperature


class LaCrosseHumidity(LaCrosseSensor):
    """Implementation of a Lacrosse humidity sensor."""

    _attr_native_unit_of_measurement: Optional[str] = PERCENTAGE
    _attr_state_class: Optional[SensorStateClass] = SensorStateClass.MEASUREMENT
    _attr_device_class: Optional[SensorDeviceClass] = SensorDeviceClass.HUMIDITY

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
        if self._low_battery is True:
            return "low"
        return "ok"

    @property
    def icon(self) -> str:
        """Icon to use in the frontend."""
        if self._low_battery is None:
            return "mdi:battery-unknown"
        if self._low_battery is True:
            return "mdi:battery-alert"
        return "mdi:battery"


TYPE_CLASSES: Dict[str, Any] = {
    "temperature": LaCrosseTemperature,
    "humidity": LaCrosseHumidity,
    "battery": LaCrosseBattery,
}
