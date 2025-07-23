"""Support to interface with Sonos players."""
from __future__ import annotations
import datetime
from functools import partial
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from soco import SoCo, alarms
from soco.core import MUSIC_SRC_LINE_IN, MUSIC_SRC_RADIO, PLAY_MODE_BY_MEANING, PLAY_MODES
from soco.data_structures import DidlFavorite, DidlMusicTrack
from soco.ms_data_structures import MusicServiceItem
from sonos_websocket.exception import SonosWebsocketError
import voluptuous as vol
from homeassistant.components import media_source, spotify
from homeassistant.components.media_player import (
    ATTR_INPUT_SOURCE,
    ATTR_MEDIA_ALBUM_NAME,
    ATTR_MEDIA_ANNOUNCE,
    ATTR_MEDIA_ARTIST,
    ATTR_MEDIA_CONTENT_ID,
    ATTR_MEDIA_ENQUEUE,
    ATTR_MEDIA_TITLE,
    BrowseMedia,
    MediaPlayerDeviceClass,
    MediaPlayerEnqueue,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
    RepeatMode,
    async_process_play_media_url,
)
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
from .const import (
    DATA_SONOS,
    DOMAIN as SONOS_DOMAIN,
    MEDIA_TYPES_TO_SONOS,
    MODELS_LINEIN_AND_TV,
    MODELS_LINEIN_ONLY,
    MODELS_TV_ONLY,
    PLAYABLE_MEDIA_TYPES,
    SONOS_CREATE_MEDIA_PLAYER,
    SONOS_MEDIA_UPDATED,
    SONOS_STATE_PLAYING,
    SONOS_STATE_TRANSITIONING,
    SOURCE_LINEIN,
    SOURCE_TV,
)
from .entity import SonosEntity
from .helpers import soco_error
from .speaker import SonosMedia, SonosSpeaker

_LOGGER = logging.getLogger(__name__)
LONG_SERVICE_TIMEOUT = 30.0
UNJOIN_SERVICE_TIMEOUT = 0.1
VOLUME_INCREMENT = 2
REPEAT_TO_SONOS = {RepeatMode.OFF: False, RepeatMode.ALL: True, RepeatMode.ONE: "ONE"}
SONOS_TO_REPEAT = {meaning: mode for mode, meaning in REPEAT_TO_SONOS.items()}
UPNP_ERRORS_TO_IGNORE = ["701", "711", "712"]
ANNOUNCE_NOT_SUPPORTED_ERRORS = ["globalError"]
SERVICE_SNAPSHOT = "snapshot"
SERVICE_RESTORE = "restore"
SERVICE_SET_TIMER = "set_sleep_timer"
SERVICE_CLEAR_TIMER = "clear_sleep_timer"
SERVICE_UPDATE_ALARM = "update_alarm"
SERVICE_PLAY_QUEUE = "play_queue"
SERVICE_REMOVE_FROM_QUEUE = "remove_from_queue"
SERVICE_GET_QUEUE = "get_queue"
ATTR_SLEEP_TIME = "sleep_time"
ATTR_ALARM_ID = "alarm_id"
ATTR_VOLUME = "volume"
ATTR_ENABLED = "enabled"
ATTR_INCLUDE_LINKED_ZONES = "include_linked_zones"
ATTR_MASTER = "master"
ATTR_WITH_GROUP = "with_group"
ATTR_QUEUE_POSITION = "queue_position"

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Sonos from a config entry."""
    platform = entity_platform.async_get_current_platform()

    @callback
    def async_create_entities(speaker: SonosSpeaker) -> None:
        """Handle device discovery and create entities."""
        _LOGGER.debug("Creating media_player on %s", speaker.zone_name)
        async_add_entities([SonosMediaPlayerEntity(speaker)])

    @service.verify_domain_control(hass, SONOS_DOMAIN)
    async def async_service_handle(service_call: ServiceCall) -> None:
        """Handle dispatched services."""
        assert platform is not None
        entities = await platform.async_extract_from_service(service_call)
        if not entities:
            return
        speakers: List[SonosSpeaker] = []
        for entity in entities:
            assert isinstance(entity, SonosMediaPlayerEntity)
            speakers.append(entity.speaker)
        if service_call.service == SERVICE_SNAPSHOT:
            await SonosSpeaker.snapshot_multi(
                hass, speakers, service_call.data[ATTR_WITH_GROUP]
            )
        elif service_call.service == SERVICE_RESTORE:
            await SonosSpeaker.restore_multi(
                hass, speakers, service_call.data[ATTR_WITH_GROUP]
            )

    config_entry.async_on_unload(
        async_dispatcher_connect(hass, SONOS_CREATE_MEDIA_PLAYER, async_create_entities)
    )
    join_unjoin_schema = cv.make_entity_service_schema(
        {vol.Optional(ATTR_WITH_GROUP, default=True): cv.boolean}
    )
    hass.services.async_register(
        SONOS_DOMAIN, SERVICE_SNAPSHOT, async_service_handle, join_unjoin_schema
    )
    hass.services.async_register(
        SONOS_DOMAIN, SERVICE_RESTORE, async_service_handle, join_unjoin_schema
    )
    platform.async_register_entity_service(
        SERVICE_SET_TIMER,
        {
            vol.Required(ATTR_SLEEP_TIME): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=86399)
            )
        },
        "set_sleep_timer",
    )
    platform.async_register_entity_service(
        SERVICE_CLEAR_TIMER, None, "clear_sleep_timer"
    )
    platform.async_register_entity_service(
        SERVICE_UPDATE_ALARM,
        {
            vol.Required(ATTR_ALARM_ID): cv.positive_int,
            vol.Optional(ATTR_TIME): cv.time,
            vol.Optional(ATTR_VOLUME): cv.small_float,
            vol.Optional(ATTR_ENABLED): cv.boolean,
            vol.Optional(ATTR_INCLUDE_LINKED_ZONES): cv.boolean,
        },
        "set_alarm",
    )
    platform.async_register_entity_service(
        SERVICE_PLAY_QUEUE,
        {vol.Optional(ATTR_QUEUE_POSITION): cv.positive_int},
        "play_queue",
    )
    platform.async_register_entity_service(
        SERVICE_REMOVE_FROM_QUEUE,
        {vol.Optional(ATTR_QUEUE_POSITION): cv.positive_int},
        "remove_from_queue",
    )
    platform.async_register_entity_service(
        SERVICE_GET_QUEUE,
        None,
        "get_queue",
        supports_response=SupportsResponse.ONLY,
    )


class SonosMediaPlayerEntity(SonosEntity, MediaPlayerEntity):
    """Representation of a Sonos entity."""

    _attr_name: None = None
    _attr_supported_features: MediaPlayerEntityFeature = (
        MediaPlayerEntityFeature.BROWSE_MEDIA
        | MediaPlayerEntityFeature.CLEAR_PLAYLIST
        | MediaPlayerEntityFeature.GROUPING
        | MediaPlayerEntityFeature.MEDIA_ANNOUNCE
        | MediaPlayerEntityFeature.MEDIA_ENQUEUE
        | MediaPlayerEntityFeature.NEXT_TRACK
        | MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.PLAY
        | MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.PREVIOUS_TRACK
        | MediaPlayerEntityFeature.REPEAT_SET
        | MediaPlayerEntityFeature.SEEK
        | MediaPlayerEntityFeature.SELECT_SOURCE
        | MediaPlayerEntityFeature.SHUFFLE_SET
        | MediaPlayerEntityFeature.STOP
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )
    _attr_media_content_type: str = MediaType.MUSIC
    _attr_device_class: MediaPlayerDeviceClass = MediaPlayerDeviceClass.SPEAKER

    def __init__(self, speaker: SonosSpeaker) -> None:
        """Initialize the media player entity."""
        super().__init__(speaker)
        self._attr_unique_id: str = self.soco.uid

    async def async_added_to_hass(self) -> None:
        """Handle common setup when added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SONOS_MEDIA_UPDATED, self.async_write_media_state
            )
        )

    @callback
    def async_write_media_state(self, uid: str) -> None:
        """Write media state if the provided UID is coordinator of this speaker."""
        if self.coordinator.uid == uid:
            self.async_write_ha_state()

    @property
    def available(self) -> bool:
        """Return if the media_player is available."""
        return (
            self.speaker.available
            and bool(self.speaker.sonos_group_entities)
            and (self.media.playback_status is not None)
        )

    @property
    def coordinator(self) -> SonosSpeaker:
        """Return the current coordinator SonosSpeaker."""
        return self.speaker.coordinator or self.speaker

    @property
    def group_members(self) -> List[str]:
        """List of entity_ids which are currently grouped together."""
        return self.speaker.sonos_group_entities

    def __hash__(self) -> int:
        """Return a hash of self."""
        return hash(self.unique_id)

    @property
    def state(self) -> MediaPlayerState:
        """Return the state of the entity."""
        if self.media.playback_status in ("PAUSED_PLAYBACK", "STOPPED"):
            if self.media.title is None:
                return MediaPlayerState.IDLE
            return MediaPlayerState.PAUSED
        if self.media.playback_status in (SONOS_STATE_PLAYING, SONOS_STATE_TRANSITIONING):
            return MediaPlayerState.PLAYING
        return MediaPlayerState.IDLE

    async def _async_fallback_poll(self) -> None:
        """Retrieve latest state by polling."""
        await self.hass.data[DATA_SONOS].favorites[
            self.speaker.household_id
        ].async_poll()
        await self.hass.async_add_executor_job(self._update)

    def _update(self) -> None:
        """Retrieve latest state by polling."""
        self.speaker.update_groups()
        self.speaker.update_volume()
        if self.speaker.is_coordinator:
            self.media.poll_media()

    @property
    def volume_level(self) -> Optional[float]:
        """Volume level of the media player (0..1)."""
        return self.speaker.volume and self.speaker.volume / 100

    @property
    def is_volume_muted(self) -> bool:
        """Return true if volume is muted."""
        return self.speaker.muted

    @property
    def shuffle(self) -> bool:
        """Shuffling state."""
        return PLAY_MODES[self.media.play_mode][0]

    @property
    def repeat(self) -> RepeatMode:
        """Return current repeat mode."""
        sonos_repeat = PLAY_MODES[self.media.play_mode][1]
        return SONOS_TO_REPEAT[sonos_repeat]

    @property
    def media(self) -> SonosMedia:
        """Return the SonosMedia object from the coordinator speaker."""
        return self.coordinator.media

    @property
    def media_content_id(self) -> Optional[str]:
        """Content id of current playing media."""
        return self.media.uri

    @property
    def media_duration(self) -> Optional[int]:
        """Duration of current playing media in seconds."""
        return int(self.media.duration) if self.media.duration else None

    @property
    def media_position(self) -> Optional[float]:
        """Position of current playing media in seconds."""
        return self.media.position

    @property
    def media_position_updated_at(self) -> Optional[datetime.datetime]:
        """When was the position of the current playing media valid."""
        return self.media.position_updated_at

    @property
    def media_image_url(self) -> Optional[str]:
        """Image url of current playing media."""
        return self.media.image_url or None

    @property
    def media_channel(self) -> Optional[str]:
        """Channel currently playing."""
        return self.media.channel or None

    @property
    def media_playlist(self) -> Optional[str]:
        """Title of playlist currently playing."""
        return self.media.playlist_name

    @property
    def media_artist(self) -> Optional[str]:
        """Artist of current playing media, music track only."""
        return self.media.artist or None

    @property
    def media_album_name(self) -> Optional[str]:
        """Album name of current playing media, music track only."""
        return self.media.album_name or None

    @property
    def media_title(self) -> Optional[str]:
        """Title of current playing media."""
        return self.media.title or None

    @property
    def source(self) -> Optional[str]:
        """Name of the current input source."""
        return self.media.source_name or None

    @soco_error()
    def volume_up(self) -> None:
        """Volume up media player."""
        self.soco.volume += VOLUME_INCREMENT

    @soco_error()
    def volume_down(self) -> None:
        """Volume down media player."""
        self.soco.volume -= VOLUME_INCREMENT

    @soco_error()
    def set_volume_level(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        self.soco.volume = int(volume * 100)

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def set_shuffle(self, shuffle: bool) -> None:
        """Enable/Disable shuffle mode."""
        sonos_shuffle = shuffle
        sonos_repeat = PLAY_MODES[self.media.play_mode][1]
        self.coordinator.soco.play_mode = PLAY_MODE_BY_MEANING[
            (sonos_shuffle, sonos_repeat)
        ]

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def set_repeat(self, repeat: RepeatMode) -> None:
        """Set repeat mode."""
        sonos_shuffle = PLAY_MODES[self.media.play_mode][0]
        sonos_repeat = REPEAT_TO_SONOS[repeat]
        self.coordinator.soco.play_mode = PLAY_MODE_BY_MEANING[
            (sonos_shuffle, sonos_repeat)
        ]

    @soco_error()
    def mute_volume(self, mute: bool) -> None:
        """Mute (true) or unmute (false) media player."""
        self.soco.mute = mute

    @soco_error()
    def select_source(self, source: str) -> None:
        """Select input source."""
        soco = self.coordinator.soco
        if source == SOURCE_LINEIN:
            soco.switch_to_line_in()
            return
        if source == SOURCE_TV:
            soco.switch_to_tv()
            return
        self._play_favorite_by_name(source)

    def _play_favorite_by_name(self, name: str) -> None:
        """Play a favorite by name."""
        fav = [fav for fav in self.speaker.favorites if fav.title == name]
        if len(fav) != 1:
            raise ServiceValidationError(
                translation_domain=SONOS_DOMAIN,
                translation_key="invalid_favorite",
                translation_placeholders={"name": name},
            )
        src = fav.pop()
        self._play_favorite(src)

    def _play_favorite(self, favorite: DidlFavorite) -> None:
        """Play a favorite."""
        uri = favorite.reference.get_uri()
        soco = self.coordinator.soco
        if soco.music_source_from_uri(uri) in [MUSIC_SRC_RADIO, MUSIC_SRC_LINE_IN]:
            soco.play_uri(uri, title=favorite.title, timeout=LONG_SERVICE_TIMEOUT)
        else:
            soco.clear_queue()
            soco.add_to_queue(favorite.reference, timeout=LONG_SERVICE_TIMEOUT)
            soco.play_from_queue(0)

    @property
    def source_list(self) -> List[str]:
        """List of available input sources."""
        model = self.coordinator.model_name.split()[-1].upper()
        if model in MODELS_LINEIN_ONLY:
            return [SOURCE_LINEIN]
        if model in MODELS_TV_ONLY:
            return [SOURCE_TV]
        if model in MODELS_LINEIN_AND_TV:
            return [SOURCE_LINEIN, SOURCE_TV]
        return []

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_play(self) -> None:
        """Send play command."""
        self.coordinator.soco.play()

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_stop(self) -> None:
        """Send stop command."""
        self.coordinator.soco.stop()

    @soco_error(UPNP_ERRORS_TO_IGNORE)
    def media_pause(self) -> None:
        """Send pause command."""
