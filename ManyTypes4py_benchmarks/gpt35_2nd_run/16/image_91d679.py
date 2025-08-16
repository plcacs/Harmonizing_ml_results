import asyncio
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
from vacuum_map_parser_base.config.color import ColorsPalette
from vacuum_map_parser_base.config.image_config import ImageConfig
from vacuum_map_parser_base.config.size import Sizes
from vacuum_map_parser_roborock.map_data_parser import RoborockMapDataParser

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class RoborockMap(RoborockCoordinatedEntityV1, ImageEntity):
    ...

    def __init__(self, config_entry: ConfigEntry, unique_id: str, coordinator: RoborockDataUpdateCoordinator, map_flag: str, map_name: str, parser: Callable) -> None:
        ...

    @property
    def is_selected(self) -> bool:
        ...

    def is_map_valid(self) -> bool:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def _handle_coordinator_update(self) -> None:
        ...

    async def async_image(self) -> bytes:
        ...

async def refresh_coordinators(hass: HomeAssistant, coord: RoborockDataUpdateCoordinator) -> None:
    ...
