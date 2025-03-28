"""Support for interfacing with Russound via RNET Protocol."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from russound import russound
import voluptuous as vol

from homeassistant.components.media_player import (
    PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_ZONES: str = "zones"
CONF_SOURCES: str = "sources"

ZONE_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_NAME): cv.string})

SOURCE_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_NAME): cv.string})

PLATFORM_SCHEMA: vol.Schema = MEDIA_PLAYER_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HOST): cv.string,
        vol.Required(CONF_NAME): cv.string,
        vol.Required(CONF_PORT): cv.port,
        vol.Required(CONF_ZONES): vol.Schema({cv.positive_int: ZONE_SCHEMA}),
        vol.Required(CONF_SOURCES): vol.All(cv.ensure_list, [SOURCE_SCHEMA]),
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Russound RNET platform."""
    host: Optional[str] = config.get(CONF_HOST)
    port: Optional[int] = config.get(CONF_PORT)

    if host is None or port is None:
        _LOGGER.error("Invalid config. Expected %s and %s", CONF_HOST, CONF_PORT)
        return

    russ: russound.Russound = russound.Russound(host, port)
    russ.connect()

    sources: List[str] = [source["name"] for source in config[CONF_SOURCES]]

    if russ.is_connected():
        for zone_id, extra in config[CONF_ZONES].items():
            add_entities(
                [RussoundRNETDevice(hass, russ, sources, zone_id, extra)], True
            )
    else:
        _LOGGER.error("Not connected to %s:%s", host, port)


class RussoundRNETDevice(MediaPlayerEntity):
    """Representation of a Russound RNET device."""

    _attr_supported_features: MediaPlayerEntityFeature = (
        MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.SELECT_SOURCE
    )

    def __init__(
        self,
        hass: HomeAssistant,
        russ: russound.Russound,
        sources: List[str],
        zone_id: int,
        extra: Dict[str, Any],
    ) -> None:
        """Initialise the Russound RNET device."""
        self._attr_name: str = extra["name"]
        self._russ: russound.Russound = russ
        self._attr_source_list: List[str] = sources
        self._controller_id: str = str(math.ceil(zone_id / 6))
        self._zone_id: int = (zone_id - 1) % 6 + 1

    def update(self) -> None:
        """Retrieve latest state."""
        try:
            ret: Optional[List[int]] = self._russ.get_zone_info(self._controller_id, self._zone_id, 4)
        except BrokenPipeError:
            _LOGGER.error("Broken Pipe Error, trying to reconnect to Russound RNET")
            self._russ.connect()
            ret = self._russ.get_zone_info(self._controller_id, self._zone_id, 4)

        _LOGGER.debug("ret= %s", ret)
        if ret is not None:
            _LOGGER.debug(
                "Updating status for RNET zone %s on controller %s",
                self._zone_id,
                self._controller_id,
            )
            if ret[0] == 0:
                self._attr_state: MediaPlayerState = MediaPlayerState.OFF
            else:
                self._attr_state = MediaPlayerState.ON
            self._attr_volume_level: float = ret[2] * 2 / 100.0
            index: int = ret[1]
            if self.source_list and 0 <= index < len(self.source_list):
                self._attr_source: Optional[str] = self.source_list[index]
        else:
            _LOGGER.error("Could not update status for zone %s", self._zone_id)

    def set_volume_level(self, volume: float) -> None:
        """Set volume level.  Volume has a range (0..1).

        Translate this to a range of (0..100) as expected
        by _russ.set_volume()
        """
        self._russ.set_volume(self._controller_id, self._zone_id, volume * 100)

    def turn_on(self) -> None:
        """Turn the media player on."""
        self._russ.set_power(self._controller_id, self._zone_id, "1")

    def turn_off(self) -> None:
        """Turn off media player."""
        self._russ.set_power(self._controller_id, self._zone_id, "0")

    def mute_volume(self, mute: bool) -> None:
        """Send mute command."""
        self._russ.toggle_mute(self._controller_id, self._zone_id)

    def select_source(self, source: str) -> None:
        """Set the input source."""
        if self.source_list and source in self.source_list:
            index: int = self.source_list.index(source)
            self._russ.set_source(self._controller_id, self._zone_id, index)
