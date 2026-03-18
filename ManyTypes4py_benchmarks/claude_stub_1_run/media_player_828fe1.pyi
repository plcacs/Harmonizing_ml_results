```pyi
from datetime import datetime
from typing import Any

from homeassistant.components.media_player import (
    MediaPlayerDeviceClass,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
    RepeatMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None: ...

SOUND_MODE_LIST: list[str]
DEFAULT_SOUND_MODE: str
YOUTUBE_PLAYER_SUPPORT: MediaPlayerEntityFeature
MUSIC_PLAYER_SUPPORT: MediaPlayerEntityFeature
NETFLIX_PLAYER_SUPPORT: MediaPlayerEntityFeature
BROWSE_PLAYER_SUPPORT: MediaPlayerEntityFeature

class AbstractDemoPlayer(MediaPlayerEntity):
    _attr_should_poll: bool
    _attr_sound_mode_list: list[str]
    _attr_name: str
    _attr_state: MediaPlayerState
    _attr_volume_level: float
    _attr_is_volume_muted: bool
    _attr_shuffle: bool
    _attr_sound_mode: str
    _attr_device_class: MediaPlayerDeviceClass | None

    def __init__(self, name: str, device_class: MediaPlayerDeviceClass | None = None) -> None: ...
    def turn_on(self) -> None: ...
    def turn_off(self) -> None: ...
    def mute_volume(self, mute: bool) -> None: ...
    def volume_up(self) -> None: ...
    def volume_down(self) -> None: ...
    def set_volume_level(self, volume: float) -> None: ...
    def media_play(self) -> None: ...
    def media_pause(self) -> None: ...
    def media_stop(self) -> None: ...
    def set_shuffle(self, shuffle: bool) -> None: ...
    def select_sound_mode(self, sound_mode: str) -> None: ...

class DemoYoutubePlayer(AbstractDemoPlayer):
    _attr_app_name: str
    _attr_media_content_type: MediaType
    _attr_supported_features: MediaPlayerEntityFeature
    _attr_media_content_id: str
    _attr_media_title: str
    _attr_media_duration: int
    _progress: int | None
    _progress_updated_at: datetime

    def __init__(
        self, name: str, youtube_id: str, media_title: str, duration: int
    ) -> None: ...
    @property
    def media_image_url(self) -> str: ...
    @property
    def media_position(self) -> int | None: ...
    @property
    def media_position_updated_at(self) -> datetime | None: ...
    def play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None: ...
    def media_pause(self) -> None: ...

class DemoMusicPlayer(AbstractDemoPlayer):
    _attr_media_album_name: str
    _attr_media_content_id: str
    _attr_media_content_type: MediaType
    _attr_media_duration: int
    _attr_media_image_url: str
    _attr_supported_features: MediaPlayerEntityFeature
    tracks: list[tuple[str, str]]
    _cur_track: int
    _attr_group_members: list[str]
    _attr_repeat: RepeatMode

    def __init__(self, name: str = ...) -> None: ...
    @property
    def media_title(self) -> str: ...
    @property
    def media_artist(self) -> str: ...
    @property
    def media_track(self) -> int: ...
    def media_previous_track(self) -> None: ...
    def media_next_track(self) -> None: ...
    def clear_playlist(self) -> None: ...
    def set_repeat(self, repeat: RepeatMode) -> None: ...
    def join_players(self, group_members: list[str]) -> None: ...
    def unjoin_player(self) -> None: ...

class DemoTVShowPlayer(AbstractDemoPlayer):
    _attr_app_name: str
    _attr_media_content_id: str
    _attr_media_content_type: MediaType
    _attr_media_duration: int
    _attr_media_image_url: str
    _attr_media_season: str
    _attr_media_series_title: str
    _attr_source_list: list[str]
    _attr_supported_features: MediaPlayerEntityFeature
    _cur_episode: int
    _episode_count: int
    _attr_source: str

    def __init__(self) -> None: ...
    @property
    def media_title(self) -> str: ...
    @property
    def media_episode(self) -> str: ...
    def media_previous_track(self) -> None: ...
    def media_next_track(self) -> None: ...
    def select_source(self, source: str) -> None: ...

class DemoBrowsePlayer(AbstractDemoPlayer):
    _attr_supported_features: MediaPlayerEntityFeature

class DemoGroupPlayer(AbstractDemoPlayer):
    _attr_supported_features: MediaPlayerEntityFeature
```