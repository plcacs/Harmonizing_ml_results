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
from typing import List, Callable, Optional

async def func_iw7cguk8(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class RoborockMap(RoborockCoordinatedEntityV1, ImageEntity):
    def __init__(self, config_entry: ConfigEntry, unique_id: str, coordinator: RoborockDataUpdateCoordinator, map_flag: str, map_name: str, parser: Callable) -> None:
        ...

    @property
    def func_gpt1cvag(self) -> bool:
        ...

    def func_yf94tu6q(self) -> bool:
        ...

    async def func_1bgt5rjm(self) -> None:
        ...

    def func_mbdk9lru(self) -> None:
        ...

    async def func_i66aclsb(self) -> bytes:
        ...

async def func_si990qea(hass: HomeAssistant, coord: RoborockDataUpdateCoordinator) -> None:
    ...
