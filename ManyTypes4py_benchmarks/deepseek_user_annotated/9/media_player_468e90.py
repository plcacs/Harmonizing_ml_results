"""Implementation of the musiccast media player."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional, Union, List, Dict, Set, cast

from aiomusiccast import MusicCastGroupException, MusicCastMediaContent
from aiomusiccast.features import ZoneFeature

from homeassistant.components import media_source
from homeassistant.components.media_player import (
    BrowseMedia,
    MediaClass,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
    RepeatMode,
    async_process_play_media_url,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import uuid as uuid_util

from .const import (
    ATTR_MAIN_SYNC,
    ATTR_MC_LINK,
    DEFAULT_ZONE,
    DOMAIN,
    HA_REPEAT_MODE_TO_MC_MAPPING,
    MC_REPEAT_MODE_TO_HA_MAPPING,
    MEDIA_CLASS_MAPPING,
    NULL_GROUP,
)
from .coordinator import MusicCastDataUpdateCoordinator
from .entity import MusicCastDeviceEntity

_LOGGER = logging.getLogger(__name__)

MUSIC_PLAYER_BASE_SUPPORT = (
    MediaPlayerEntityFeature.SHUFFLE_SET
    | MediaPlayerEntityFeature.REPEAT_SET
    | MediaPlayerEntityFeature.SELECT_SOUND_MODE
    | MediaPlayerEntityFeature.SELECT_SOURCE
    | MediaPlayerEntityFeature.GROUPING
    | MediaPlayerEntityFeature.PLAY_MEDIA
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up MusicCast sensor based on a config entry."""
    coordinator: MusicCastDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]

    name = coordinator.data.network_name

    media_players: List[Entity] = []

    for zone in coordinator.data.zones:
        zone_name = name if zone == DEFAULT_ZONE else f"{name} {zone}"

        media_players.append(
            MusicCastMediaPlayer(zone, zone_name, entry.entry_id, coordinator)
        )

    async_add_entities(media_players)


class MusicCastMediaPlayer(MusicCastDeviceEntity, MediaPlayerEntity):
    """The musiccast media player."""

    _attr_media_content_type = MediaType.MUSIC
    _attr_should_poll = False

    def __init__(
        self,
        zone_id: str,
        name: str,
        entry_id: str,
        coordinator: MusicCastDataUpdateCoordinator,
    ) -> None:
        """Initialize the musiccast device."""
        self._player_state = MediaPlayerState.PLAYING
        self._volume_muted = False
        self._shuffle = False
        self._zone_id = zone_id

        super().__init__(
            name=name,
            icon="mdi:speaker",
            coordinator=coordinator,
        )

        self._volume_min = self.coordinator.data.zones[self._zone_id].min_volume
        self._volume_max = self.coordinator.data.zones[self._zone_id].max_volume

        self._cur_track = 0
        self._repeat = RepeatMode.OFF

    async def async_added_to_hass(self) -> None:
        """Run when this Entity has been added to HA."""
        await super().async_added_to_hass()
        self.coordinator.entities.append(self)
        self.coordinator.musiccast.register_group_update_callback(
            self.update_all_mc_entities
        )
        self.async_on_remove(
            self.coordinator.async_add_listener(self.async_schedule_check_client_list)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Entity being removed from hass."""
        await super().async_will_remove_from_hass()
        self.coordinator.entities.remove(self)
        self.coordinator.musiccast.remove_group_update_callback(
            self.update_all_mc_entities
        )

    @property
    def ip_address(self) -> str:
        """Return the ip address of the musiccast device."""
        return self.coordinator.musiccast.ip

    @property
    def zone_id(self) -> str:
        """Return the zone id of the musiccast device."""
        return self._zone_id

    @property
    def _is_netusb(self) -> bool:
        return self.coordinator.data.netusb_input == self.source_id

    @property
    def _is_tuner(self) -> bool:
        return self.source_id == "tuner"

    @property
    def media_content_id(self) -> Optional[str]:
        """Return the content ID of current playing media."""
        return None

    @property
    def state(self) -> MediaPlayerState:
        """Return the state of the player."""
        if self.coordinator.data.zones[self._zone_id].power == "on":
            if self._is_netusb and self.coordinator.data.netusb_playback == "pause":
                return MediaPlayerState.PAUSED
            if self._is_netusb and self.coordinator.data.netusb_playback == "stop":
                return MediaPlayerState.IDLE
            return MediaPlayerState.PLAYING
        return MediaPlayerState.OFF

    @property
    def source_mapping(self) -> Dict[str, str]:
        """Return a mapping of the actual source names to their labels configured in the MusicCast App."""
        ret: Dict[str, str] = {}
        for inp in self.coordinator.data.zones[self._zone_id].input_list:
            label = self.coordinator.data.input_names.get(inp, "")
            if inp != label and (
                label in self.coordinator.data.zones[self._zone_id].input_list
                or list(self.coordinator.data.input_names.values()).count(label) > 1
            ):
                label += f" ({inp})"
            if label == "":
                label = inp
            ret[inp] = label
        return ret

    @property
    def reverse_source_mapping(self) -> Dict[str, str]:
        """Return a mapping from the source label to the source name."""
        return {v: k for k, v in self.source_mapping.items()}

    @property
    def volume_level(self) -> Optional[float]:
        """Return the volume level of the media player (0..1)."""
        if ZoneFeature.VOLUME in self.coordinator.data.zones[self._zone_id].features:
            volume = self.coordinator.data.zones[self._zone_id].current_volume
            return (volume - self._volume_min) / (self._volume_max - self._volume_min)
        return None

    @property
    def is_volume_muted(self) -> Optional[bool]:
        """Return boolean if volume is currently muted."""
        if ZoneFeature.VOLUME in self.coordinator.data.zones[self._zone_id].features:
            return self.coordinator.data.zones[self._zone_id].mute
        return None

    @property
    def shuffle(self) -> bool:
        """Boolean if shuffling is enabled."""
        return (
            self.coordinator.data.netusb_shuffle == "on" if self._is_netusb else False
        )

    @property
    def sound_mode(self) -> Optional[str]:
        """Return the current sound mode."""
        return self.coordinator.data.zones[self._zone_id].sound_program

    @property
    def sound_mode_list(self) -> List[str]:
        """Return a list of available sound modes."""
        return self.coordinator.data.zones[self._zone_id].sound_program_list

    @property
    def zone(self) -> str:
        """Return the zone of the media player."""
        return self._zone_id

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this media_player."""
        return f"{self.coordinator.data.device_id}_{self._zone_id}"

    async def async_turn_on(self) -> None:
        """Turn the media player on."""
        await self.coordinator.musiccast.turn_on(self._zone_id)
        self.async_write_ha_state()

    async def async_turn_off(self) -> None:
        """Turn the media player off."""
        await self.coordinator.musiccast.turn_off(self._zone_id)
        self.async_write_ha_state()

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute the volume."""
        await self.coordinator.musiccast.mute_volume(self._zone_id, mute)
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume: float) -> None:
        """Set the volume level, range 0..1."""
        await self.coordinator.musiccast.set_volume_level(self._zone_id, volume)
        self.async_write_ha_state()

    async def async_volume_up(self) -> None:
        """Turn volume up for media player."""
        await self.coordinator.musiccast.volume_up(self._zone_id)

    async def async_volume_down(self) -> None:
        """Turn volume down for media player."""
        await self.coordinator.musiccast.volume_down(self._zone_id)

    async def async_media_play(self) -> None:
        """Send play command."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_play()
        else:
            raise HomeAssistantError(
                "Service play is not supported for non NetUSB sources."
            )

    async def async_media_pause(self) -> None:
        """Send pause command."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_pause()
        else:
            raise HomeAssistantError(
                "Service pause is not supported for non NetUSB sources."
            )

    async def async_media_stop(self) -> None:
        """Send stop command."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_stop()
        else:
            raise HomeAssistantError(
                "Service stop is not supported for non NetUSB sources."
            )

    async def async_set_shuffle(self, shuffle: bool) -> None:
        """Enable/disable shuffle mode."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_shuffle(shuffle)
        else:
            raise HomeAssistantError(
                "Service shuffle is not supported for non NetUSB sources."
            )

    async def async_play_media(
        self, media_type: Union[MediaType, str], media_id: str, **kwargs: Any
    ) -> None:
        """Play media."""
        if media_source.is_media_source_id(media_id):
            play_item = await media_source.async_resolve_media(
                self.hass, media_id, self.entity_id
            )
            media_id = play_item.url

        if self.state == MediaPlayerState.OFF:
            await self.async_turn_on()

        if media_id:
            parts = media_id.split(":")

            if parts[0] == "list":
                index = parts[3] if parts[3] != "-1" else "0"
                await self.coordinator.musiccast.play_list_media(index, self._zone_id)
                return

            if parts[0] == "presets":
                index = parts[1]
                await self.coordinator.musiccast.recall_netusb_preset(
                    self._zone_id, index
                )
                return

            if parts[0] in ("http", "https") or media_id.startswith("/"):
                media_id = async_process_play_media_url(self.hass, media_id)
                await self.coordinator.musiccast.play_url_media(
                    self._zone_id, media_id, "HomeAssistant"
                )
                return

        raise HomeAssistantError(
            "Only presets, media from media browser and http URLs are supported"
        )

    async def async_browse_media(
        self, media_content_type: Optional[str] = None, media_content_id: Optional[str] = None
    ) -> BrowseMedia:
        """Implement the websocket media browsing helper."""
        if media_content_id and media_source.is_media_source_id(media_content_id):
            return await media_source.async_browse_media(
                self.hass,
                media_content_id,
                content_filter=lambda item: item.media_content_type.startswith(
                    "audio/"
                ),
            )

        if self.state == MediaPlayerState.OFF:
            raise HomeAssistantError(
                "The device has to be turned on to be able to browse media."
            )

        if media_content_id:
            media_content_path = media_content_id.split(":")
            media_content_provider = await MusicCastMediaContent.browse_media(
                self.coordinator.musiccast, self._zone_id, media_content_path, 24
            )
            add_media_source = False
        else:
            media_content_provider = MusicCastMediaContent.categories(
                self.coordinator.musiccast, self._zone_id
            )
            add_media_source = True

        def get_content_type(item: Any) -> str:
            if item.can_play:
                return MediaClass.TRACK
            return MediaClass.DIRECTORY

        children: List[BrowseMedia] = [
            BrowseMedia(
                title=child.title,
                media_class=MEDIA_CLASS_MAPPING.get(child.content_type),
                media_content_id=child.content_id,
                media_content_type=get_content_type(child),
                can_play=child.can_play,
                can_expand=child.can_browse,
                thumbnail=child.thumbnail,
            )
            for child in media_content_provider.children
        ]

        if add_media_source:
            with contextlib.suppress(media_source.BrowseError):
                item = await media_source.async_browse_media(
                    self.hass,
                    None,
                    content_filter=lambda item: item.media_content_type.startswith(
                        "audio/"
                    ),
                )
                if item.domain is None:
                    children.extend(item.children)
                else:
                    children.append(item)

        return BrowseMedia(
            title=media_content_provider.title,
            media_class=MEDIA_CLASS_MAPPING.get(media_content_provider.content_type),
            media_content_id=media_content_provider.content_id,
            media_content_type=get_content_type(media_content_provider),
            can_play=False,
            can_expand=media_content_provider.can_browse,
            children=children,
        )

    async def async_select_sound_mode(self, sound_mode: str) -> None:
        """Select sound mode."""
        await self.coordinator.musiccast.select_sound_mode(self._zone_id, sound_mode)

    @property
    def media_image_url(self) -> Optional[str]:
        """Return the image url of current playing media."""
        if self.is_client and self.group_server != self:
            return self.group_server.coordinator.musiccast.media_image_url
        return self.coordinator.musiccast.media_image_url if self._is_netusb else None

    @property
    def media_title(self) -> Optional[str]:
        """Return the title of current playing media."""
        if self._is_netusb:
            return self.coordinator.data.netusb_track
        if self._is_tuner:
            return self.coordinator.musiccast.tuner_media_title
        return None

    @property
    def media_artist(self) -> Optional[str]:
        """Return the artist of current playing media (Music track only)."""
        if self._is_netusb:
            return self.coordinator.data.netusb_artist
        if self._is_tuner:
            return self.coordinator.musiccast.tuner_media_artist
        return None

    @property
    def media_album_name(self) -> Optional[str]:
        """Return the album of current playing media (Music track only)."""
        return self.coordinator.data.netusb_album if self._is_netusb else None

    @property
    def repeat(self) -> RepeatMode:
        """Return current repeat mode."""
        return (
            MC_REPEAT_MODE_TO_HA_MAPPING.get(self.coordinator.data.netusb_repeat)
            if self._is_netusb
            else RepeatMode.OFF
        )

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        """Flag media player features that are supported."""
        supported_features = MUSIC_PLAYER_BASE_SUPPORT
        zone = self.coordinator.data.zones[self._zone_id]

        if ZoneFeature.POWER in zone.features:
            supported_features |= (
                MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.TURN_OFF
            )
        if ZoneFeature.VOLUME in zone.features:
            supported_features |= (
                MediaPlayerEntityFeature.VOLUME_SET
                | MediaPlayerEntityFeature.VOLUME_STEP
            )
        if ZoneFeature.MUTE in zone.features:
            supported_features |= MediaPlayerEntityFeature.VOLUME_MUTE

        if self._is_netusb or self._is_tuner:
            supported_features |= MediaPlayerEntityFeature.PREVIOUS_TRACK
            supported_features |= MediaPlayerEntityFeature.NEXT_TRACK

        if self._is_netusb:
            supported_features |= MediaPlayerEntityFeature.PAUSE
            supported_features |= MediaPlayerEntityFeature.PLAY
            supported_features |= MediaPlayerEntityFeature.STOP

        if self.state != MediaPlayerState.OFF:
            supported_features |= MediaPlayerEntityFeature.BROWSE_MEDIA

        return supported_features

    async def async_media_previous_track(self) -> None:
        """Send previous track command."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_previous_track()
        elif self._is_tuner:
            await self.coordinator.musiccast.tuner_previous_station()
        else:
            raise HomeAssistantError(
                "Service previous track is not supported for non NetUSB or Tuner"
                " sources."
            )

    async def async_media_next_track(self) -> None:
        """Send next track command."""
        if self._is_netusb:
            await self.coordinator.musiccast.netusb_next_track()
        elif self._