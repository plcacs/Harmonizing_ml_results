from __future__ import annotations
import logging
import time
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import UNDEFINED, ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from . import REPETIER_API, SENSOR_TYPES, UPDATE_SIGNAL, RepetierSensorEntityDescription
_LOGGER: logging.Logger = logging.getLogger(__name__)

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    if discovery_info is None:
        return
    sensor_map: dict[str, type] = {'bed_temperature': RepetierTempSensor, 'extruder_temperature': RepetierTempSensor, 'chamber_temperature': RepetierTempSensor, 'current_state': RepetierSensor, 'current_job': RepetierJobSensor, 'job_end': RepetierJobEndSensor, 'job_start': RepetierJobStartSensor}
    sensors_info: list[dict[str, str]] = discovery_info['sensors']
    entities: list[SensorEntity] = []
    for info in sensors_info:
        printer_name: str = info['printer_name']
        api = hass.data[REPETIER_API][printer_name]
        printer_id: str = info['printer_id']
        sensor_type: str = info['sensor_type']
        temp_id: str = info['temp_id']
        description: RepetierSensorEntityDescription = SENSOR_TYPES[sensor_type]
        name_suffix: str = '' if description.name is UNDEFINED else description.name
        name: str = f'{info["name"]}{name_suffix}'
        if temp_id is not None:
            _LOGGER.debug('%s Temp_id: %s', sensor_type, temp_id)
            name = f'{name}{temp_id}'
        sensor_class: type = sensor_map[sensor_type]
        entity: SensorEntity = sensor_class(api, temp_id, name, printer_id, description)
        entities.append(entity)
    add_entities(entities, True)

class RepetierSensor(SensorEntity):
    _attr_should_poll: bool = False

    def __init__(self, api, temp_id, name, printer_id, description) -> None:
        self.entity_description: RepetierSensorEntityDescription = description
        self._api = api
        self._attributes: dict[str, str] = {}
        self._temp_id: str = temp_id
        self._printer_id: str = printer_id
        self._state: str | None = None
        self._attr_name: str = name
        self._attr_available: bool = False

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        return self._attributes

    @property
    def native_value(self) -> str | None:
        return self._state

    @callback
    def update_callback(self) -> None:
        self.async_schedule_update_ha_state(True)

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(async_dispatcher_connect(self.hass, UPDATE_SIGNAL, self.update_callback))

    def _get_data(self) -> dict[str, str] | None:
        sensor_type: str = self.entity_description.key
        data: dict[str, str] | None = self._api.get_data(self._printer_id, sensor_type, self._temp_id)
        if data is None:
            _LOGGER.debug('Data not found for %s and %s', sensor_type, self._temp_id)
            self._attr_available = False
            return None
        self._attr_available = True
        return data

    def update(self) -> None:
        if (data := self._get_data()) is None:
            return
        state: str = data.pop('state')
        _LOGGER.debug('Printer %s State %s', self.name, state)
        self._attributes.update(data)
        self._state = state

class RepetierTempSensor(RepetierSensor):
    @property
    def native_value(self) -> float | None:
        if self._state is None:
            return None
        return round(self._state, 2)

    def update(self) -> None:
        if (data := self._get_data()) is None:
            return
        state: float = data.pop('state')
        temp_set: float = data['temp_set']
        _LOGGER.debug('Printer %s Setpoint: %s, Temp: %s', self.name, temp_set, state)
        self._attributes.update(data)
        self._state = state

class RepetierJobSensor(RepetierSensor):
    @property
    def native_value(self) -> float | None:
        if self._state is None:
            return None
        return round(self._state, 2)

class RepetierJobEndSensor(RepetierSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def update(self) -> None:
        if (data := self._get_data()) is None:
            return
        job_name: str = data['job_name']
        start: float = data['start']
        print_time: float = data['print_time']
        from_start: float = data['from_start']
        time_end: float = start + round(print_time, 0)
        self._state = dt_util.utc_from_timestamp(time_end)
        remaining: float = print_time - from_start
        remaining_secs: int = int(round(remaining, 0))
        _LOGGER.debug('Job %s remaining %s', job_name, time.strftime('%H:%M:%S', time.gmtime(remaining_secs)))

class RepetierJobStartSensor(RepetierSensor):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def update(self) -> None:
        if (data := self._get_data()) is None:
            return
        job_name: str = data['job_name']
        start: float = data['start']
        from_start: float = data['from_start']
        self._state = dt_util.utc_from_timestamp(start)
        elapsed_secs: int = int(round(from_start, 0))
        _LOGGER.debug('Job %s elapsed %s', job_name, time.strftime('%H:%M:%S', time.gmtime(elapsed_secs)))
