from __future__ import annotations
from asyncio import Task
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, List, Dict, Union
import voluptuous as vol
from homeassistant.components import media_source
from homeassistant.components.media_player import BrowseMedia, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType, async_process_play_media_url
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo, format_mac
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from .const import ATTR_BLUESOUND_GROUP, ATTR_MASTER, DOMAIN
from .coordinator import BluesoundCoordinator
from .utils import dispatcher_join_signal, dispatcher_unjoin_signal, format_unique_id

if TYPE_CHECKING:
    from . import BluesoundConfigEntry

_LOGGER: logging.Logger
SCAN_INTERVAL: timedelta
DATA_BLUESOUND: str
DEFAULT_PORT: int
SERVICE_CLEAR_TIMER: str
SERVICE_JOIN: str
SERVICE_SET_TIMER: str
SERVICE_UNJOIN: str
POLL_TIMEOUT: int

async def async_setup_entry(hass: HomeAssistant, config_entry: BluesoundConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class BluesoundPlayer(CoordinatorEntity[BluesoundCoordinator], MediaPlayerEntity):

    def __init__(self, coordinator: BluesoundCoordinator, host: str, port: int, player: Player) -> None:

    async def async_added_to_hass(self) -> None:

    async def async_will_remove_from_hass(self) -> None:

    @callback
    def _handle_coordinator_update(self) -> None:

    @property
    def state(self) -> MediaPlayerState:

    @property
    def media_title(self) -> str:

    @property
    def media_artist(self) -> str:

    @property
    def media_album_name(self) -> str:

    @property
    def media_image_url(self) -> str:

    @property
    def media_position(self) -> Union[int, None]:

    @property
    def media_duration(self) -> Union[int, None]:

    @property
    def media_position_updated_at(self) -> datetime:

    @property
    def volume_level(self) -> float:

    @property
    def is_volume_muted(self) -> bool:

    @property
    def id(self) -> str:

    @property
    def bluesound_device_name(self) -> str:

    @property
    def sync_status(self) -> SyncStatus:

    @property
    def source_list(self) -> Union[List[str], None]:

    @property
    def source(self) -> Union[str, None]:

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:

    @property
    def is_leader(self) -> bool:

    @property
    def is_grouped(self) -> bool:

    @property
    def shuffle(self) -> bool:

    async def async_join(self, master: str) -> None:

    async def async_unjoin(self) -> None:

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

    def rebuild_bluesound_group(self) -> List[str]:

    async def async_add_follower(self, host: str, port: int) -> None:

    async def async_remove_follower(self, host: str, port: int) -> None:

    async def async_increase_timer(self) -> None:

    async def async_clear_timer(self) -> None:

    async def async_set_shuffle(self, shuffle: bool) -> None:

    async def async_select_source(self, source: str) -> None:

    async def async_clear_playlist(self) -> None:

    async def async_media_next_track(self) -> None:

    async def async_media_previous_track(self) -> None:

    async def async_media_play(self) -> None:

    async def async_media_pause(self) -> None:

    async def async_media_stop(self) -> None:

    async def async_media_seek(self, position: int) -> None:

    async def async_play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:

    async def async_volume_up(self) -> None:

    async def async_volume_down(self) -> None:

    async def async_set_volume_level(self, volume: float) -> None:

    async def async_mute_volume(self, mute: bool) -> None:

    async def async_browse_media(self, media_content_type: str = None, media_content_id: str = None) -> BrowseMedia:
