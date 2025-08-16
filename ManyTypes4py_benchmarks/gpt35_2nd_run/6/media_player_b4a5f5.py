from __future__ import annotations
import datetime
from functools import partial
import logging
from typing import Any
from soco import SoCo, alarms
from soco.core import MUSIC_SRC_LINE_IN, MUSIC_SRC_RADIO, PLAY_MODE_BY_MEANING, PLAY_MODES
from soco.data_structures import DidlFavorite, DidlMusicTrack
from soco.ms_data_structures import MusicServiceItem
from sonos_websocket.exception import SonosWebsocketError
import voluptuous as vol
from homeassistant.components import media_source, spotify
from homeassistant.components.media_player import ATTR_INPUT_SOURCE, ATTR_MEDIA_ALBUM_NAME, ATTR_MEDIA_ANNOUNCE, ATTR_MEDIA_ARTIST, ATTR_MEDIA_CONTENT_ID, ATTR_MEDIA_ENQUEUE, ATTR_MEDIA_TITLE, BrowseMedia, MediaPlayerDeviceClass, MediaPlayerEnqueue, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType, RepeatMode, async_process_play_media_url
from homeassistant.components.plex import PLEX_URI_SCHEME
from homeassistant.components.plex.services import process_plex_payload
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TIME
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, entity_platform, service
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from . import UnjoinData, media_browser
from .const import DATA_SONOS, DOMAIN as SONOS_DOMAIN, MEDIA_TYPES_TO_SONOS, MODELS_LINEIN_AND_TV, MODELS_LINEIN_ONLY, MODELS_TV_ONLY, PLAYABLE_MEDIA_TYPES, SONOS_CREATE_MEDIA_PLAYER, SONOS_MEDIA_UPDATED, SONOS_STATE_PLAYING, SONOS_STATE_TRANSITIONING, SOURCE_LINEIN, SOURCE_TV
from .entity import SonosEntity
from .helpers import soco_error
from .speaker import SonosMedia, SonosSpeaker

_LOGGER: logging.Logger
LONG_SERVICE_TIMEOUT: float
UNJOIN_SERVICE_TIMEOUT: float
VOLUME_INCREMENT: int
REPEAT_TO_SONOS: dict[RepeatMode, Any]
SONOS_TO_REPEAT: dict[Any, RepeatMode]
UPNP_ERRORS_TO_IGNORE: list[str]
ANNOUNCE_NOT_SUPPORTED_ERRORS: list[str]
SERVICE_SNAPSHOT: str
SERVICE_RESTORE: str
SERVICE_SET_TIMER: str
SERVICE_CLEAR_TIMER: str
SERVICE_UPDATE_ALARM: str
SERVICE_PLAY_QUEUE: str
SERVICE_REMOVE_FROM_QUEUE: str
SERVICE_GET_QUEUE: str
ATTR_SLEEP_TIME: str
ATTR_ALARM_ID: str
ATTR_VOLUME: str
ATTR_ENABLED: str
ATTR_INCLUDE_LINKED_ZONES: str
ATTR_MASTER: str
ATTR_WITH_GROUP: str
ATTR_QUEUE_POSITION: str

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()

    @callback
    def async_create_entities(speaker: SonosSpeaker) -> None:
        ...

    @service.verify_domain_control(hass, SONOS_DOMAIN)
    async def async_service_handle(service_call: ServiceCall) -> None:
        ...

    def async_process_unjoin(now: datetime.datetime) -> None:
        ...

class SonosMediaPlayerEntity(SonosEntity, MediaPlayerEntity):
    _attr_name: None
    _attr_supported_features: MediaPlayerEntityFeature
    _attr_media_content_type: MediaType
    _attr_device_class: MediaPlayerDeviceClass
    _attr_unique_id: str

    def __init__(self, speaker: SonosSpeaker) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def async_write_media_state(self, uid: str) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def coordinator(self) -> SonosSpeaker:
        ...

    @property
    def group_members(self) -> list[str]:
        ...

    def __hash__(self) -> int:
        ...

    @property
    def state(self) -> MediaPlayerState:
        ...

    async def _async_fallback_poll(self) -> None:
        ...

    def _update(self) -> None:
        ...

    @property
    def volume_level(self) -> float:
        ...

    @property
    def is_volume_muted(self) -> bool:
        ...

    @property
    def shuffle(self) -> bool:
        ...

    @property
    def repeat(self) -> RepeatMode:
        ...

    @property
    def media(self) -> SonosMedia:
        ...

    @property
    def media_content_id(self) -> str:
        ...

    @property
    def media_duration(self) -> int:
        ...

    @property
    def media_position(self) -> int:
        ...

    @property
    def media_position_updated_at(self) -> datetime.datetime:
        ...

    @property
    def media_image_url(self) -> str:
        ...

    @property
    def media_channel(self) -> str:
        ...

    @property
    def media_playlist(self) -> str:
        ...

    @property
    def media_artist(self) -> str:
        ...

    @property
    def media_album_name(self) -> str:
        ...

    @property
    def media_title(self) -> str:
        ...

    @property
    def source(self) -> str:
        ...

    @soco_error()
    def volume_up(self) -> None:
        ...

    @soco_error()
    def volume_down(self) -> None:
        ...

    @soco_error()
    def set_volume_level(self, volume: float) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def set_shuffle(self, shuffle: bool) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def set_repeat(self, repeat: RepeatMode) -> None:
        ...

    @soco_error()
    def mute_volume(self, mute: bool) -> None:
        ...

    @soco_error()
    def select_source(self, source: str) -> None:
        ...

    def _play_favorite_by_name(self, name: str) -> None:
        ...

    def _play_favorite(self, favorite: DidlFavorite) -> None:
        ...

    @property
    def source_list(self) -> list[str]:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_play(self) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_stop(self) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_pause(self) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_next_track(self) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_previous_track(self) -> None:
        ...

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_seek(self, position: int) -> None:
        ...

    @soco_error()
    def clear_playlist(self) -> None:
        ...

    async def async_play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:
        ...

    @soco_error()
    def _play_media(self, media_type: str, media_id: str, is_radio: bool, **kwargs: Any) -> None:
        ...

    def _play_media_queue(self, soco: SoCo, item: MusicServiceItem, enqueue: str) -> None:
        ...

    @soco_error()
    def set_sleep_timer(self, sleep_time: int) -> None:
        ...

    @soco_error()
    def clear_sleep_timer(self) -> None:
        ...

    @soco_error()
    def set_alarm(self, alarm_id: int, time: datetime.time = None, volume: float = None, enabled: bool = None, include_linked_zones: bool = None) -> None:
        ...

    @soco_error()
    def play_queue(self, queue_position: int = 0) -> None:
        ...

    @soco_error()
    def remove_from_queue(self, queue_position: int = 0) -> None:
        ...

    @soco_error()
    def get_queue(self) -> list[dict[str, Any]]:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    async def async_get_browse_image(self, media_content_type: str, media_content_id: str, media_image_id: str = None) -> tuple[str, bytes]:
        ...

    async def async_browse_media(self, media_content_type: str = None, media_content_id: str = None) -> BrowseMedia:
        ...

    async def async_join_players(self, group_members: list[str]) -> None:
        ...

    async def async_unjoin_player(self) -> None:
        ...
