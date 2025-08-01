from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, List
from enocean.utils import combine_hex
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    RestoreSensor,
    SensorDeviceClass,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    CONF_DEVICE_CLASS,
    CONF_ID,
    CONF_NAME,
    PERCENTAGE,
    STATE_CLOSED,
    STATE_OPEN,
    UnitOfPower,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .entity import EnOceanEntity

CONF_MAX_TEMP = "max_temp"
CONF_MIN_TEMP = "min_temp"
CONF_RANGE_FROM = "range_from"
CONF_RANGE_TO = "range_to"
DEFAULT_NAME = "EnOcean sensor"
SENSOR_TYPE_HUMIDITY = "humidity"
SENSOR_TYPE_POWER = "powersensor"
SENSOR_TYPE_TEMPERATURE = "temperature"
SENSOR_TYPE_WINDOWHANDLE = "windowhandle"


@dataclass(frozen=True, kw_only=True)
class EnOceanSensorEntityDescription(SensorEntityDescription):
    """Describes EnOcean sensor entity."""


SENSOR_DESC_TEMPERATURE = EnOceanSensorEntityDescription(
    key=SENSOR_TYPE_TEMPERATURE,
    name="Temperature",
    native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    device_class=SensorDeviceClass.TEMPERATURE,
    state_class=SensorStateClass.MEASUREMENT,
    unique_id=lambda dev_id: f"{combine_hex(dev_id)}-{SENSOR_TYPE_TEMPERATURE}",
)
SENSOR_DESC_HUMIDITY = EnOceanSensorEntityDescription(
    key=SENSOR_TYPE_HUMIDITY,
    name="Humidity",
    native_unit_of_measurement=PERCENTAGE,
    device_class=SensorDeviceClass.HUMIDITY,
    state_class=SensorStateClass.MEASUREMENT,
    unique_id=lambda dev_id: f"{combine_hex(dev_id)}-{SENSOR_TYPE_HUMIDITY}",
)
SENSOR_DESC_POWER = EnOceanSensorEntityDescription(
    key=SENSOR_TYPE_POWER,
    name="Power",
    native_unit_of_measurement=UnitOfPower.WATT,
    device_class=SensorDeviceClass.POWER,
    state_class=SensorStateClass.MEASUREMENT,
    unique_id=lambda dev_id: f"{combine_hex(dev_id)}-{SENSOR_TYPE_POWER}",
)
SENSOR_DESC_WINDOWHANDLE = EnOceanSensorEntityDescription(
    key=SENSOR_TYPE_WINDOWHANDLE,
    name="WindowHandle",
    translation_key="window_handle",
    unique_id=lambda dev_id: f"{combine_hex(dev_id)}-{SENSOR_TYPE_WINDOWHANDLE}",
)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_ID): vol.All(cv.ensure_list, [vol.Coerce(int)]),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_DEVICE_CLASS, default=SENSOR_TYPE_POWER): cv.string,
        vol.Optional(CONF_MAX_TEMP, default=40): vol.Coerce(int),
        vol.Optional(CONF_MIN_TEMP, default=0): vol.Coerce(int),
        vol.Optional(CONF_RANGE_FROM, default=255): cv.positive_int,
        vol.Optional(CONF_RANGE_TO, default=0): cv.positive_int,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up an EnOcean sensor device."""
    dev_id: List[int] = config[CONF_ID]
    dev_name: str = config[CONF_NAME]
    sensor_type: str = config[CONF_DEVICE_CLASS]
    entities: List[EnOceanSensor] = []
    if sensor_type == SENSOR_TYPE_TEMPERATURE:
        temp_min: int = config[CONF_MIN_TEMP]
        temp_max: int = config[CONF_MAX_TEMP]
        range_from: int = config[CONF_RANGE_FROM]
        range_to: int = config[CONF_RANGE_TO]
        entities = [
            EnOceanTemperatureSensor(
                dev_id, dev_name, SENSOR_DESC_TEMPERATURE, scale_min=temp_min, scale_max=temp_max, range_from=range_from, range_to=range_to
            )
        ]
    elif sensor_type == SENSOR_TYPE_HUMIDITY:
        entities = [EnOceanHumiditySensor(dev_id, dev_name, SENSOR_DESC_HUMIDITY)]
    elif sensor_type == SENSOR_TYPE_POWER:
        entities = [EnOceanPowerSensor(dev_id, dev_name, SENSOR_DESC_POWER)]
    elif sensor_type == SENSOR_TYPE_WINDOWHANDLE:
        entities = [EnOceanWindowHandle(dev_id, dev_name, SENSOR_DESC_WINDOWHANDLE)]
    add_entities(entities)


class EnOceanSensor(EnOceanEntity, RestoreSensor):
    """Representation of an EnOcean sensor device such as a power meter."""

    def __init__(self, dev_id: List[int], dev_name: str, description: EnOceanSensorEntityDescription) -> None:
        """Initialize the EnOcean sensor device."""
        super().__init__(dev_id)
        self.entity_description: EnOceanSensorEntityDescription = description
        self._attr_name: str = f"{description.name} {dev_name}"
        self._attr_unique_id: str = description.unique_id(dev_id)

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        await super().async_added_to_hass()
        if self._attr_native_value is not None:
            return
        sensor_data: Optional[Any] = await self.async_get_last_sensor_data()
        if sensor_data is not None:
            self._attr_native_value = sensor_data.native_value  # type: ignore

    def value_changed(self, packet: Any) -> None:
        """Update the internal state of the sensor."""
        pass


class EnOceanPowerSensor(EnOceanSensor):
    """Representation of an EnOcean power sensor.

    EEPs (EnOcean Equipment Profiles):
    - A5-12-01 (Automated Meter Reading, Electricity)
    """

    def value_changed(self, packet: Any) -> None:
        """Update the internal state of the sensor."""
        if packet.rorg != 165:
            return
        packet.parse_eep(18, 1)
        if packet.parsed["DT"]["raw_value"] == 1:
            raw_val: float = packet.parsed["MR"]["raw_value"]
            divisor: float = packet.parsed["DIV"]["raw_value"]
            self._attr_native_value = raw_val / 10 ** divisor  # type: ignore
            self.schedule_update_ha_state()


class EnOceanTemperatureSensor(EnOceanSensor):
    """Representation of an EnOcean temperature sensor device.

    EEPs (EnOcean Equipment Profiles):
    - A5-02-01 to A5-02-1B All 8 Bit Temperature Sensors of A5-02
    - A5-10-01 to A5-10-14 (Room Operating Panels)
    - A5-04-01 (Temp. and Humidity Sensor, Range 0°C to +40°C and 0% to 100%)
    - A5-04-02 (Temp. and Humidity Sensor, Range -20°C to +60°C and 0% to 100%)
    - A5-10-10 (Temp. and Humidity Sensor and Set Point)
    - A5-10-12 (Temp. and Humidity Sensor, Set Point and Occupancy Control)
    - 10 Bit Temp. Sensors are not supported (A5-02-20, A5-02-30)

    For the following EEPs the scales must be set to "0 to 250":
    - A5-04-01
    - A5-04-02
    - A5-10-10 to A5-10-14
    """

    def __init__(
        self,
        dev_id: List[int],
        dev_name: str,
        description: EnOceanSensorEntityDescription,
        *,
        scale_min: int,
        scale_max: int,
        range_from: int,
        range_to: int,
    ) -> None:
        """Initialize the EnOcean temperature sensor device."""
        super().__init__(dev_id, dev_name, description)
        self._scale_min: int = scale_min
        self._scale_max: int = scale_max
        self.range_from: int = range_from
        self.range_to: int = range_to

    def value_changed(self, packet: Any) -> None:
        """Update the internal state of the sensor."""
        if packet.data[0] != 165:
            return
        temp_scale: int = self._scale_max - self._scale_min
        temp_range: int = self.range_to - self.range_from
        raw_val: int = packet.data[3]
        temperature: float = temp_scale / temp_range * (raw_val - self.range_from)
        temperature += self._scale_min
        self._attr_native_value = round(temperature, 1)  # type: ignore
        self.schedule_update_ha_state()


class EnOceanHumiditySensor(EnOceanSensor):
    """Representation of an EnOcean humidity sensor device.

    EEPs (EnOcean Equipment Profiles):
    - A5-04-01 (Temp. and Humidity Sensor, Range 0°C to +40°C and 0% to 100%)
    - A5-04-02 (Temp. and Humidity Sensor, Range -20°C to +60°C and 0% to 100%)
    - A5-10-10 to A5-10-14 (Room Operating Panels)
    """

    def value_changed(self, packet: Any) -> None:
        """Update the internal state of the sensor."""
        if packet.rorg != 165:
            return
        humidity: float = packet.data[2] * 100 / 250
        self._attr_native_value = round(humidity, 1)  # type: ignore
        self.schedule_update_ha_state()


class EnOceanWindowHandle(EnOceanSensor):
    """Representation of an EnOcean window handle device.

    EEPs (EnOcean Equipment Profiles):
    - F6-10-00 (Mechanical handle / Hoppe AG)
    """

    def value_changed(self, packet: Any) -> None:
        """Update the internal state of the sensor."""
        action: int = (packet.data[1] & 112) >> 4
        if action == 7:
            self._attr_native_value = STATE_CLOSED  # type: ignore
        if action in (4, 6):
            self._attr_native_value = STATE_OPEN  # type: ignore
        if action == 5:
            self._attr_native_value = "tilt"  # type: ignore
        self.schedule_update_ha_state()