"""Support for monitoring Repetier Server Sensors."""
from __future__ import annotations
import logging
import time
from typing import Any, Optional, List, Dict
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from . import (
    REPETIER_API,
    SENSOR_TYPES,
    UPDATE_SIGNAL,
    RepetierSensorEntityDescription,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the available Repetier Server sensors."""
    if discovery_info is None:
        return
    sensor_map: Dict[str, type[RepetierSensor]] = {
        "bed_temperature": RepetierTempSensor,
        "extruder_temperature": RepetierTempSensor,
        "chamber_temperature": RepetierTempSensor,
        "current_state": RepetierSensor,
        "current_job": RepetierJobSensor,
        "job_end": RepetierJobEndSensor,
        "job_start": RepetierJobStartSensor,
    }
    sensors_info: List[Dict[str, Any]] = discovery_info["sensors"]
    entities: List[RepetierSensor] = []
    for info in sensors_info:
        printer_name: str = info["printer_name"]
        api: Any = hass.data[REPETIER_API][printer_name]
        printer_id: str = info["printer_id"]
        sensor_type: str = info["sensor_type"]
        temp_id: Optional[str] = info.get("temp_id")
        description: RepetierSensorEntityDescription = SENSOR_TYPES[sensor_type]
        name_suffix: str = "" if description.name is None else description.name
        name: str = f"{info['name']}{name_suffix}"
        if temp_id is not None:
            _LOGGER.debug("%s Temp_id: %s", sensor_type, temp_id)
            name = f"{name}{temp_id}"
        sensor_class: type[RepetierSensor] = sensor_map[sensor_type]
        entity: RepetierSensor = sensor_class(api, temp_id, name, printer_id, description)
        entities.append(entity)
    add_entities(entities, True)


class RepetierSensor(SensorEntity):
    """Class to create and populate a Repetier Sensor."""

    _attr_should_poll: bool = False

    def __init__(
        self,
        api: Any,
        temp_id: Optional[str],
        name: str,
        printer_id: str,
        description: RepetierSensorEntityDescription,
    ) -> None:
        """Init new sensor."""
        self.entity_description: RepetierSensorEntityDescription = description
        self._api: Any = api
        self._attributes: Dict[str, Any] = {}
        self._temp_id: Optional[str] = temp_id
        self._printer_id: str = printer_id
        self._state: Optional[Any] = None
        self._attr_name: str = name
        self._attr_available: bool = False

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return sensor attributes."""
        return self._attributes

    @property
    def native_value(self) -> Optional[Any]:
        """Return sensor state."""
        return self._state

    @callback
    def update_callback(self) -> None:
        """Get new data and update state."""
        self.async_schedule_update_ha_state(True)

    async def async_added_to_hass(self) -> None:
        """Connect update callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(self.hass, UPDATE_SIGNAL, self.update_callback)
        )

    def _get_data(self) -> Optional[Dict[str, Any]]:
        """Return new data from the api cache."""
        sensor_type: str = self.entity_description.key
        data: Optional[Dict[str, Any]] = self._api.get_data(
            self._printer_id, sensor_type, self._temp_id
        )
        if data is None:
            _LOGGER.debug(
                "Data not found for %s and %s", sensor_type, self._temp_id
            )
            self._attr_available = False
            return None
        self._attr_available = True
        return data

    def update(self) -> None:
        """Update the sensor."""
        data: Optional[Dict[str, Any]] = self._get_data()
        if data is None:
            return
        state: Any = data.pop("state")
        _LOGGER.debug("Printer %s State %s", self.name, state)
        self._attributes.update(data)
        self._state = state


class RepetierTempSensor(RepetierSensor):
    """Represent a Repetier temp sensor."""

    @property
    def native_value(self) -> Optional[float]:
        """Return sensor state."""
        if self._state is None:
            return None
        return round(float(self._state), 2)

    def update(self) -> None:
        """Update the sensor."""
        data: Optional[Dict[str, Any]] = self._get_data()
        if data is None:
            return
        state: Any = data.pop("state")
        temp_set: Any = data["temp_set"]
        _LOGGER.debug(
            "Printer %s Setpoint: %s, Temp: %s", self.name, temp_set, state
        )
        self._attributes.update(data)
        self._state = state


class RepetierJobSensor(RepetierSensor):
    """Represent a Repetier job sensor."""

    @property
    def native_value(self) -> Optional[float]:
        """Return sensor state."""
        if self._state is None:
            return None
        return round(float(self._state), 2)


class RepetierJobEndSensor(RepetierSensor):
    """Class to create and populate a Repetier Job End timestamp Sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def update(self) -> None:
        """Update the sensor."""
        data: Optional[Dict[str, Any]] = self._get_data()
        if data is None:
            return
        job_name: str = data["job_name"]
        start: float = data["start"]
        print_time: float = data["print_time"]
        from_start: float = data["from_start"]
        time_end: float = start + round(print_time, 0)
        self._state: Any = dt_util.utc_from_timestamp(time_end)
        remaining: float = print_time - from_start
        remaining_secs: int = int(round(remaining, 0))
        _LOGGER.debug(
            "Job %s remaining %s",
            job_name,
            time.strftime("%H:%M:%S", time.gmtime(remaining_secs)),
        )


class RepetierJobStartSensor(RepetierSensor):
    """Class to create and populate a Repetier Job Start timestamp Sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def update(self) -> None:
        """Update the sensor."""
        data: Optional[Dict[str, Any]] = self._get_data()
        if data is None:
            return
        job_name: str = data["job_name"]
        start: float = data["start"]
        from_start: float = data["from_start"]
        self._state: Any = dt_util.utc_from_timestamp(start)
        elapsed_secs: int = int(round(from_start, 0))
        _LOGGER.debug(
            "Job %s elapsed %s",
            job_name,
            time.strftime("%H:%M:%S", time.gmtime(elapsed_secs)),
        )
