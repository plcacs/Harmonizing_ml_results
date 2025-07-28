"""Support for collecting data from the ARWN project."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from homeassistant.components import mqtt
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import DEGREE, UnitOfPrecipitationDepth, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import slugify
from homeassistant.util.json import json_loads_object

_LOGGER = logging.getLogger(__name__)
DOMAIN: str = 'arwn'
DATA_ARWN: str = 'arwn'
TOPIC: str = 'arwn/#'

def discover_sensors(topic: str, payload: Dict[str, Any]) -> Optional[List[ArwnSensor]]:
    """Given a topic, dynamically create the right sensor type.

    Async friendly.
    """
    parts: List[str] = topic.split('/')
    unit: str = payload.get('units', '')
    domain: str = parts[1]
    if domain == 'temperature':
        name: str = parts[2]
        if unit == 'F':
            unit = UnitOfTemperature.FAHRENHEIT  # type: ignore
        else:
            unit = UnitOfTemperature.CELSIUS  # type: ignore
        return [ArwnSensor(topic, name, 'temp', unit, device_class=SensorDeviceClass.TEMPERATURE)]
    if domain == 'moisture':
        name: str = f'{parts[2]} Moisture'
        return [ArwnSensor(topic, name, 'moisture', unit, icon='mdi:water-percent')]
    if domain == 'rain':
        if len(parts) >= 3 and parts[2] == 'today':
            return [ArwnSensor(topic, 'Rain Since Midnight', 'since_midnight', UnitOfPrecipitationDepth.INCHES, device_class=SensorDeviceClass.PRECIPITATION)]
        return [ArwnSensor(topic + '/total', 'Total Rainfall', 'total', unit, device_class=SensorDeviceClass.PRECIPITATION),
                ArwnSensor(topic + '/rate', 'Rainfall Rate', 'rate', unit, device_class=SensorDeviceClass.PRECIPITATION)]
    if domain == 'barometer':
        return [ArwnSensor(topic, 'Barometer', 'pressure', unit, icon='mdi:thermometer-lines')]
    if domain == 'wind':
        return [ArwnSensor(topic + '/speed', 'Wind Speed', 'speed', unit, device_class=SensorDeviceClass.WIND_SPEED),
                ArwnSensor(topic + '/gust', 'Wind Gust', 'gust', unit, device_class=SensorDeviceClass.WIND_SPEED),
                ArwnSensor(topic + '/dir', 'Wind Direction', 'direction', DEGREE, icon='mdi:compass')]
    return None

def _slug(name: str) -> str:
    return f'sensor.arwn_{slugify(name)}'

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the ARWN platform."""
    if not await mqtt.async_wait_for_mqtt_client(hass):
        _LOGGER.error('MQTT integration is not available')
        return

    @callback
    def async_sensor_event_received(msg: mqtt.Message) -> None:
        """Process events as sensors.

        When a new event on our topic (arwn/#) is received we map it
        into a known kind of sensor based on topic name. If we've
        never seen this before, we keep this sensor around in a global
        cache. If we have seen it before, we update the values of the
        existing sensor. Either way, we push an ha state update at the
        end for the new event we've seen.

        This lets us dynamically incorporate sensors without any
        configuration on our side.
        """
        event: Dict[str, Any] = json_loads_object(msg.payload)
        sensors: Optional[List[ArwnSensor]] = discover_sensors(msg.topic, event)
        if not sensors:
            return
        store: Dict[str, ArwnSensor] = hass.data.get(DATA_ARWN, {})
        if 'timestamp' in event:
            del event['timestamp']
        for sensor in sensors:
            if sensor.name not in store:
                sensor.hass = hass
                sensor.set_event(event)
                store[sensor.name] = sensor
                _LOGGER.debug('Registering sensor %(name)s => %(event)s', {'name': sensor.name, 'event': event})
                async_add_entities((sensor,), True)
            else:
                _LOGGER.debug('Recording sensor %(name)s => %(event)s', {'name': sensor.name, 'event': event})
                store[sensor.name].set_event(event)
        hass.data[DATA_ARWN] = store

    await mqtt.async_subscribe(hass, TOPIC, async_sensor_event_received, 0)

class ArwnSensor(SensorEntity):
    """Representation of an ARWN sensor."""
    _attr_should_poll: bool = False

    def __init__(
        self,
        topic: str,
        name: str,
        state_key: str,
        units: Union[str, UnitOfTemperature, UnitOfPrecipitationDepth],
        icon: Optional[str] = None,
        device_class: Optional[SensorDeviceClass] = None
    ) -> None:
        """Initialize the sensor."""
        self.entity_id: str = _slug(name)
        self._attr_name: str = name
        self._attr_unique_id: str = topic
        self._state_key: str = state_key
        self._attr_native_unit_of_measurement: Union[str, UnitOfTemperature, UnitOfPrecipitationDepth] = units
        self._attr_icon: Optional[str] = icon
        self._attr_device_class: Optional[SensorDeviceClass] = device_class

    def set_event(self, event: Dict[str, Any]) -> None:
        """Update the sensor with the most recent event."""
        ev: Dict[str, Any] = {}
        ev.update(event)
        self._attr_extra_state_attributes = ev
        self._attr_native_value = ev.get(self._state_key, None)
        self.async_write_ha_state()