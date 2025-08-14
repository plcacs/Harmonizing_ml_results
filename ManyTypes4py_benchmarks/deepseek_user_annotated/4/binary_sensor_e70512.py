"""Support for Envisalink zone states- represented as binary sensors."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.const import ATTR_LAST_TRIP_TIME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

from . import CONF_ZONENAME, CONF_ZONETYPE, DATA_EVL, SIGNAL_ZONE_UPDATE, ZONE_SCHEMA
from .entity import EnvisalinkEntity

_LOGGER: logging.Logger = logging.getLogger(__name__)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Envisalink binary sensor entities."""
    if not discovery_info:
        return
    configured_zones: Dict[str, Any] = discovery_info["zones"]

    entities: list[EnvisalinkBinarySensor] = []
    for zone_num in configured_zones:
        entity_config_data: Dict[str, Any] = ZONE_SCHEMA(configured_zones[zone_num])
        entity: EnvisalinkBinarySensor = EnvisalinkBinarySensor(
            hass,
            zone_num,
            entity_config_data[CONF_ZONENAME],
            entity_config_data[CONF_ZONETYPE],
            hass.data[DATA_EVL].alarm_state["zone"][zone_num],
            hass.data[DATA_EVL],
        )
        entities.append(entity)

    async_add_entities(entities)


class EnvisalinkBinarySensor(EnvisalinkEntity, BinarySensorEntity):
    """Representation of an Envisalink binary sensor."""

    def __init__(
        self,
        hass: HomeAssistant,
        zone_number: str,
        zone_name: str,
        zone_type: str,
        info: Dict[str, Any],
        controller: Any,
    ) -> None:
        """Initialize the binary_sensor."""
        self._zone_type: str = zone_type
        self._zone_number: str = zone_number

        _LOGGER.debug("Setting up zone: %s", zone_name)
        super().__init__(zone_name, info, controller)

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_ZONE_UPDATE, self.async_update_callback
            )
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        attr: Dict[str, Any] = {}

        seconds_ago: int = self._info["last_fault"]
        if seconds_ago < 65536 * 5:
            now: datetime.datetime = dt_util.now().replace(microsecond=0)
            delta: datetime.timedelta = datetime.timedelta(seconds=seconds_ago)
            last_trip_time: str = (now - delta).isoformat()
        else:
            last_trip_time: Optional[str] = None

        attr[ATTR_LAST_TRIP_TIME] = last_trip_time
        attr["zone"] = self._zone_number

        return attr

    @property
    def is_on(self) -> bool:
        """Return true if sensor is on."""
        return bool(self._info["status"]["open"])

    @property
    def device_class(self) -> str:
        """Return the class of this sensor, from DEVICE_CLASSES."""
        return self._zone_type

    @callback
    def async_update_callback(self, zone: Optional[str]) -> None:
        """Update the zone's state, if needed."""
        if zone is None or int(zone) == int(self._zone_number):
            self.async_write_ha_state()
