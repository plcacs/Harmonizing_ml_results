"""Support for Roborock image."""
import asyncio
from collections.abc import Callable
from datetime import datetime
import io
from typing import Any, Dict, List, Optional, Set

from roborock import RoborockCommand
from vacuum_map_parser_base.config.color import ColorsPalette
from vacuum_map_parser_base.config.image_config import ImageConfig
from vacuum_map_parser_base.config.size import Sizes
from vacuum_map_parser_roborock.map_data_parser import RoborockMapDataParser

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DEFAULT_DRAWABLES, DOMAIN, DRAWABLES, IMAGE_CACHE_INTERVAL, MAP_FILE_FORMAT, MAP_SLEEP
from .coordinator import RoborockConfigEntry, RoborockDataUpdateCoordinator
from .entity import RoborockCoordinatedEntityV1


async def async_setup_entry(hass: HomeAssistant, config_entry: RoborockConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up Roborock image platform."""
    drawables: List[str] = [drawable for (drawable, default_value) in DEFAULT_DRAWABLES.items() if config_entry.options.get(DRAWABLES, {}).get(drawable, default_value)]
    parser: RoborockMapDataParser = RoborockMapDataParser(ColorsPalette(), Sizes(), drawables, ImageConfig(), [])

    def parse_image(map_bytes: bytes) -> Optional[bytes]:
        parsed_map = parser.parse(map_bytes)
        if parsed_map.image is None:
            return None
        img_byte_arr = io.BytesIO()
        parsed_map.image.data.save(img_byte_arr, format=MAP_FILE_FORMAT)
        return img_byte_arr.getvalue()
    
    await asyncio.gather(*(refresh_coordinators(hass, coord) for coord in config_entry.runtime_data.v1))
    async_add_entities((RoborockMap(config_entry, f'{coord.duid_slug}_map_{map_info.name}', coord, map_info.flag, map_info.name, parse_image) for coord in config_entry.runtime_data.v1 for map_info in coord.maps.values()))


class RoborockMap(RoborockCoordinatedEntityV1, ImageEntity):
    """A class to let you visualize the map."""
    _attr_has_entity_name: bool = True
    image_last_updated: datetime
    _attr_name: str

    def __init__(self, config_entry: RoborockConfigEntry, unique_id: str, coordinator: RoborockDataUpdateCoordinator, map_flag: int, map_name: str, parser: Callable[[bytes], Optional[bytes]]) -> None:
        """Initialize a Roborock map."""
        RoborockCoordinatedEntityV1.__init__(self, unique_id, coordinator)
        ImageEntity.__init__(self, coordinator.hass)
        self.config_entry: RoborockConfigEntry = config_entry
        self._attr_name = map_name
        self.parser: Callable[[bytes], Optional[bytes]] = parser
        self.map_flag: int = map_flag
        self.cached_map: bytes = b''
        self._attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC

    @property
    def is_selected(self) -> bool:
        """Return if this map is the currently selected map."""
        return self.map_flag == self.coordinator.current_map

    def is_map_valid(self) -> bool:
        """Update the map if it is valid.

        Update this map if it is the currently active map, and the
        vacuum is cleaning, or if it has never been set at all.
        """
        return self.cached_map == b'' or (self.is_selected and self.image_last_updated is not None and (self.coordinator.roborock_device_info.props.status is not None) and bool(self.coordinator.roborock_device_info.props.status.in_cleaning))

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass load any previously cached maps from disk."""
        await super().async_added_to_hass()
        content: Optional[bytes] = await self.coordinator.map_storage.async_load_map(self.map_flag)
        self.cached_map = content or b''
        self._attr_image_last_updated = dt_util.utcnow()
        self.async_write_ha_state()

    def _handle_coordinator_update(self) -> None:
        if (dt_util.utcnow() - self.image_last_updated).total_seconds() > IMAGE_CACHE_INTERVAL and self.is_map_valid():
            self._attr_image_last_updated = dt_util.utcnow()
        super()._handle_coordinator_update()

    async def async_image(self) -> Optional[bytes]:
        """Update the image if it is not cached."""
        if self.is_map_valid():
            response: List[Any] = await asyncio.gather(*(self.cloud_api.get_map_v1(), self.coordinator.set_current_map_rooms()), return_exceptions=True)
            if not isinstance(response[0], bytes) or (content := self.parser(response[0])) is None:
                raise HomeAssistantError(translation_domain=DOMAIN, translation_key='map_failure')
            if self.cached_map != content:
                self.cached_map = content
                await self.coordinator.map_storage.async_save_map(self.map_flag, content)
        return self.cached_map


async def refresh_coordinators(hass: HomeAssistant, coord: RoborockDataUpdateCoordinator) -> None:
    """Get the starting map information for all maps for this device.

    The following steps must be done synchronously.
    Only one map can be loaded at a time per device.
    """
    cur_map: Optional[int] = coord.current_map
    assert cur_map is not None
    map_flags: List[int] = sorted(coord.maps, key=lambda data: data == cur_map, reverse=True)
    for map_flag in map_flags:
        if map_flag != cur_map:
            await coord.api.send_command(RoborockCommand.LOAD_MULTI_MAP, [map_flag])
            coord.current_map = map_flag
            await asyncio.sleep(MAP_SLEEP)
        await coord.set_current_map_rooms()
    if len(coord.maps) != 1:
        await coord.cloud_api.send_command(RoborockCommand.LOAD_MULTI_MAP, [cur_map])
        coord.current_map = cur_map
