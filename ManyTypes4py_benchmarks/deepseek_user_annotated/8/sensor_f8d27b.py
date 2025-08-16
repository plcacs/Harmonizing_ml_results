"""Support for MQTT room presence detection."""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
import logging
from typing import Any, Optional, Dict

import voluptuous as vol

from homeassistant.components import mqtt
from homeassistant.components.mqtt import CONF_STATE_TOPIC
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import (
    ATTR_DEVICE_ID,
    ATTR_ID,
    CONF_DEVICE_ID,
    CONF_NAME,
    CONF_TIMEOUT,
    CONF_UNIQUE_ID,
    STATE_NOT_HOME,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util, slugify
from homeassistant.util.json import json_loads

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_DISTANCE: str = "distance"
ATTR_ROOM: str = "room"

CONF_AWAY_TIMEOUT: str = "away_timeout"

DEFAULT_AWAY_TIMEOUT: int = 0
DEFAULT_NAME: str = "Room Sensor"
DEFAULT_TIMEOUT: int = 5
DEFAULT_TOPIC: str = "room_presence"

PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_DEVICE_ID): cv.string,
        vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_AWAY_TIMEOUT, default=DEFAULT_AWAY_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
    }
).extend(mqtt.MQTT_RO_SCHEMA.schema)


@lru_cache(maxsize=256)
def _slugify_upper(string: str) -> str:
    """Return a slugified version of string, uppercased."""
    return slugify(string).upper()


MQTT_PAYLOAD: vol.Schema = vol.Schema(
    vol.All(
        json_loads,
        vol.Schema(
            {
                vol.Required(ATTR_ID): cv.string,
                vol.Required(ATTR_DISTANCE): vol.Coerce(float),
            },
            extra=vol.ALLOW_EXTRA,
        ),
    )
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up MQTT room Sensor."""
    if not await mqtt.async_wait_for_mqtt_client(hass):
        _LOGGER.error("MQTT integration is not available")
        return
    async_add_entities(
        [
            MQTTRoomSensor(
                config.get(CONF_NAME),
                config[CONF_STATE_TOPIC],
                config[CONF_DEVICE_ID],
                config[CONF_TIMEOUT],
                config[CONF_AWAY_TIMEOUT],
                config.get(CONF_UNIQUE_ID),
            )
        ]
    )


class MQTTRoomSensor(SensorEntity):
    """Representation of a room sensor that is updated via MQTT."""

    _attr_unique_id: Optional[str]
    _state: str
    _name: Optional[str]
    _state_topic: str
    _device_id: str
    _timeout: int
    _consider_home: Optional[timedelta]
    _distance: Optional[float]
    _updated: Optional[dt_util.datetime]

    def __init__(
        self,
        name: Optional[str],
        state_topic: str,
        device_id: str,
        timeout: int,
        consider_home: int,
        unique_id: Optional[str],
    ) -> None:
        """Initialize the sensor."""
        self._attr_unique_id = unique_id
        self._state = STATE_NOT_HOME
        self._name = name
        self._state_topic = f"{state_topic}/+"
        self._device_id = _slugify_upper(device_id)
        self._timeout = timeout
        self._consider_home = (
            timedelta(seconds=consider_home) if consider_home else None
        )
        self._distance = None
        self._updated = None

    async def async_added_to_hass(self) -> None:
        """Subscribe to MQTT events."""

        @callback
        def update_state(device_id: str, room: str, distance: float) -> None:
            """Update the sensor state."""
            self._state = room
            self._distance = distance
            self._updated = dt_util.utcnow()
            self.async_write_ha_state()

        @callback
        def message_received(msg: mqtt.MQTTMessage) -> None:
            """Handle new MQTT messages."""
            try:
                data: Dict[str, Any] = MQTT_PAYLOAD(msg.payload)
            except vol.MultipleInvalid as error:
                _LOGGER.debug("Skipping update because of malformatted data: %s", error)
                return

            device: Dict[str, Any] = _parse_update_data(msg.topic, data)
            if device.get(CONF_DEVICE_ID) == self._device_id:
                if self._distance is None or self._updated is None:
                    update_state(**device)
                else:
                    timediff: timedelta = dt_util.utcnow() - self._updated
                    if (
                        device.get(ATTR_ROOM) == self._state
                        or device.get(ATTR_DISTANCE) < self._distance
                        or timediff.total_seconds() >= self._timeout
                    ):
                        update_state(**device)

        await mqtt.async_subscribe(self.hass, self._state_topic, message_received, 1)

    @property
    def name(self) -> Optional[str]:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> Dict[str, Optional[float]]:
        """Return the state attributes."""
        return {ATTR_DISTANCE: self._distance}

    @property
    def native_value(self) -> str:
        """Return the current room of the entity."""
        return self._state

    def update(self) -> None:
        """Update the state for absent devices."""
        if (
            self._updated
            and self._consider_home
            and dt_util.utcnow() - self._updated > self._consider_home
        ):
            self._state = STATE_NOT_HOME


def _parse_update_data(topic: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the room presence update."""
    parts: List[str] = topic.split("/")
    room: str = parts[-1]
    device_id: str = _slugify_upper(data.get(ATTR_ID))
    distance: float = data.get("distance")
    return {ATTR_DEVICE_ID: device_id, ATTR_ROOM: room, ATTR_DISTANCE: distance}
