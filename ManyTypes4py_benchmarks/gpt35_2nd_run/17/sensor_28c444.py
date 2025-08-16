from __future__ import annotations
from typing import Any, Dict, List, Optional
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import CONCENTRATION_PARTS_PER_MILLION, CONF_DEVICES, CONF_NAME, CONF_SENSOR_TYPE, CONF_UNIT_OF_MEASUREMENT, DEGREE, LIGHT_LUX, PERCENTAGE, UV_INDEX, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfLength, UnitOfPower, UnitOfPrecipitationDepth, UnitOfPressure, UnitOfSpeed, UnitOfTemperature, UnitOfVolumetricFlux
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_ALIASES, CONF_AUTOMATIC_ADD, DATA_DEVICE_REGISTER, DATA_ENTITY_LOOKUP, EVENT_KEY_ID, EVENT_KEY_SENSOR, EVENT_KEY_UNIT, SIGNAL_AVAILABILITY, SIGNAL_HANDLE_EVENT, TMP_ENTITY
from .entity import RflinkDevice

SENSOR_TYPES: List[SensorEntityDescription] = [
    SensorEntityDescription(key='average_windspeed', name='Average windspeed', device_class=SensorDeviceClass.WIND_SPEED, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfSpeed.KILOMETERS_PER_HOUR),
    SensorEntityDescription(key='barometric_pressure', name='Barometric pressure', device_class=SensorDeviceClass.PRESSURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfPressure.HPA),
    # Add other SensorEntityDescription objects here
]

SENSOR_TYPES_DICT: Dict[str, SensorEntityDescription] = {desc.key: desc for desc in SENSOR_TYPES}

PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_AUTOMATIC_ADD, default=True): cv.boolean,
    vol.Optional(CONF_DEVICES, default={}): {
        cv.string: vol.Schema({
            vol.Optional(CONF_NAME): cv.string,
            vol.Required(CONF_SENSOR_TYPE): cv.string,
            vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string,
            vol.Optional(CONF_ALIASES, default=[]): vol.All(cv.ensure_list, [cv.string])
        })
    }
}, extra=vol.ALLOW_EXTRA)

def lookup_unit_for_sensor_type(sensor_type: str) -> Optional[str]:
    field_abbrev: Dict[str, str] = {v: k for k, v in PACKET_FIELDS.items()}
    return UNITS.get(field_abbrev.get(sensor_type))

def devices_from_config(domain_config: Dict[str, Any]) -> List[RflinkSensor]:
    devices: List[RflinkSensor] = []
    for device_id, config in domain_config[CONF_DEVICES].items():
        device = RflinkSensor(device_id, **config)
        devices.append(device)
    return devices

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    async_add_entities(devices_from_config(config))

    async def add_new_device(event: Dict[str, Any]) -> None:
        device_id = event[EVENT_KEY_ID]
        device = RflinkSensor(device_id, event[EVENT_KEY_SENSOR], event[EVENT_KEY_UNIT], initial_event=event)
        async_add_entities([device])

    if config[CONF_AUTOMATIC_ADD]:
        hass.data[DATA_DEVICE_REGISTER][EVENT_KEY_SENSOR] = add_new_device

class RflinkSensor(RflinkDevice, SensorEntity):
    def __init__(self, device_id: str, sensor_type: str, unit_of_measurement: Optional[str] = None, initial_event: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._sensor_type = sensor_type
        self._unit_of_measurement = unit_of_measurement
        if sensor_type in SENSOR_TYPES_DICT:
            self.entity_description = SENSOR_TYPES_DICT[sensor_type]
        elif not unit_of_measurement:
            self._unit_of_measurement = lookup_unit_for_sensor_type(sensor_type)
        super().__init__(device_id, initial_event=initial_event, **kwargs)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        self._state = event['value']

    async def async_added_to_hass(self) -> None:
        tmp_entity = TMP_ENTITY.format(self._device_id)
        if tmp_entity in self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_SENSOR][self._device_id]:
            self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_SENSOR][self._device_id].remove(tmp_entity)
        self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_SENSOR][self._device_id].append(self.entity_id)
        if self._aliases:
            for _id in self._aliases:
                self.hass.data[DATA_ENTITY_LOOKUP][EVENT_KEY_SENSOR][_id].append(self.entity_id)
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_AVAILABILITY, self._availability_callback))
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_HANDLE_EVENT.format(self.entity_id), self.handle_event_callback))
        if self._initial_event:
            self.handle_event_callback(self._initial_event)

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        if self._unit_of_measurement:
            return self._unit_of_measurement
        if hasattr(self, 'entity_description'):
            return self.entity_description.native_unit_of_measurement
        return None

    @property
    def native_value(self) -> Any:
        return self._state
