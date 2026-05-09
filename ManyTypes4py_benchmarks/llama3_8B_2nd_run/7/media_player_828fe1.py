from __future__ import annotations
from datetime import datetime
from typing import Any

class AbstractDemoPlayer(MediaPlayerEntity):
    """A demo media players."""
    _attr_sound_mode_list: list[str]
    _attr_volume_level: float
    _attr_is_volume_muted: bool
    _attr_shuffle: bool
    _attr_sound_mode: str
    _attr_device_class: MediaPlayerDeviceClass | None

    def __init__(self, name: str, device_class: MediaPlayerDeviceClass | None = None):
        """Initialize the demo device."""
        self._attr_name = name
        self._attr_state = MediaPlayerState.PLAYING
        self._attr_volume_level = 1.0
        self._attr_is_volume_muted = False
        self._attr_shuffle = False
        self._attr_sound_mode = DEFAULT_SOUND_MODE
        self._attr_device_class = device_class

class DemoYoutubePlayer(AbstractDemoPlayer):
    """A Demo media player that only supports YouTube."""
    _attr_app_name: str
    _attr_media_content_type: MediaType
    _attr_supported_features: int

    def __init__(self, name: str, youtube_id: str, media_title: str, duration: int):
        """Initialize the demo device."""
        super().__init__(name)
        self._attr_media_content_id = youtube_id
        self._attr_media_title = media_title
        self._attr_media_duration = duration
        self._progress: int | None
        self._progress_updated_at: datetime | None

    @property
    def media_image_url(self) -> str:
        """Return the image url of current playing media."""
        return f'https://img.youtube.com/vi/{self.media_content_id}/hqdefault.jpg'

    @property
    def media_position(self) -> int | None:
        """Position of current playing media in seconds."""
        if self._progress is None:
            return None
        position = self._progress
        if self.state == MediaPlayerState.PLAYING:
            position += int((dt_util.utcnow() - self._progress_updated_at).total_seconds())
        return position

class DemoMusicPlayer(AbstractDemoPlayer):
    """A Demo media player."""
    _attr_media_album_name: str
    _attr_media_content_id: str
    _attr_media_content_type: MediaType
    _attr_media_duration: int
    _attr_media_image_url: str
    _attr_supported_features: int
    tracks: list[tuple[str, str]]

    def __init__(self, name: str = 'Walkman'):
        """Initialize the demo device."""
        super().__init__(name)
        self._cur_track: int
        self._attr_group_members: list[str]
        self._attr_repeat: RepeatMode

class DemoTVShowPlayer(AbstractDemoPlayer):
    """A Demo media player that only supports Netflix."""
    _attr_app_name: str
    _attr_media_content_id: str
    _attr_media_content_type: MediaType
    _attr_media_duration: int
    _attr_media_image_url: str
    _attr_media_season: str
    _attr_media_series_title: str
    _attr_source_list: list[str]
    _attr_supported_features: int

    def __init__(self):
        """Initialize the demo device."""
        super().__init__('Lounge room', MediaPlayerDeviceClass.TV)
        self._cur_episode: int
        self._episode_count: int
        self._attr_source: str

class DemoBrowsePlayer(AbstractDemoPlayer):
    """A Demo media player that supports browse."""
    _attr_supported_features: int

class DemoGroupPlayer(AbstractDemoPlayer):
    """A Demo media player that supports grouping."""
    _attr_supported_features: int | MediaPlayerEntityFeature.GROUPING | MediaPlayerEntityFeature.TURN_OFF
