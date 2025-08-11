"""Volumio Platform.

Volumio rest API: https://volumio.github.io/docs/API/REST_API.html
"""
from __future__ import annotations
from datetime import timedelta
import json
from typing import Any
from homeassistant.components.media_player import BrowseMedia, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType, RepeatMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ID, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import Throttle
from .browse_media import browse_node, browse_top_level
from .const import DATA_INFO, DATA_VOLUMIO, DOMAIN
PLAYLIST_UPDATE_INTERVAL = timedelta(seconds=15)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Volumio media player platform."""
    data = hass.data[DOMAIN][config_entry.entry_id]
    volumio = data[DATA_VOLUMIO]
    info = data[DATA_INFO]
    uid = config_entry.data[CONF_ID]
    name = config_entry.data[CONF_NAME]
    entity = Volumio(volumio, uid, name, info)
    async_add_entities([entity])

class Volumio(MediaPlayerEntity):
    """Volumio Player Object."""
    _attr_has_entity_name = True
    _attr_name = None
    _attr_media_content_type = MediaType.MUSIC
    _attr_supported_features = MediaPlayerEntityFeature.PAUSE | MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.VOLUME_MUTE | MediaPlayerEntityFeature.PREVIOUS_TRACK | MediaPlayerEntityFeature.NEXT_TRACK | MediaPlayerEntityFeature.SEEK | MediaPlayerEntityFeature.STOP | MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PLAY_MEDIA | MediaPlayerEntityFeature.VOLUME_STEP | MediaPlayerEntityFeature.SELECT_SOURCE | MediaPlayerEntityFeature.REPEAT_SET | MediaPlayerEntityFeature.SHUFFLE_SET | MediaPlayerEntityFeature.CLEAR_PLAYLIST | MediaPlayerEntityFeature.BROWSE_MEDIA
    _attr_source_list = []

    def __init__(self, volumio: Union[str, None, tuple[str]], uid: Union[str, tuple[typing.Union[int,str]], typing.Mapping], name: Union[str, None, dict[str, typing.Any]], info: Union[str, dict[str, typing.Any]]) -> None:
        """Initialize the media player."""
        self._volumio = volumio
        unique_id = uid
        self._state = {}
        self.thumbnail_cache = {}
        self._attr_unique_id = unique_id
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, unique_id)}, manufacturer='Volumio', model=info['hardware'], name=name, sw_version=info['systemversion'])

    async def async_update(self):
        """Update state."""
        self._state = await self._volumio.get_state()
        await self._async_update_playlists()

    @property
    def state(self):
        """Return the state of the device."""
        status = self._state.get('status', None)
        if status == 'pause':
            return MediaPlayerState.PAUSED
        if status == 'play':
            return MediaPlayerState.PLAYING
        return MediaPlayerState.IDLE

    @property
    def media_title(self) -> Union[str, None]:
        """Title of current playing media."""
        return self._state.get('title', None)

    @property
    def media_artist(self) -> Union[str, None]:
        """Artist of current playing media (Music track only)."""
        return self._state.get('artist', None)

    @property
    def media_album_name(self) -> Union[str, None, dict]:
        """Artist of current playing media (Music track only)."""
        return self._state.get('album', None)

    @property
    def media_image_url(self) -> Union[str, dict[str, str]]:
        """Image url of current playing media."""
        url = self._state.get('albumart', None)
        return self._volumio.canonic_url(url)

    @property
    def media_seek_position(self) -> Union[bool, str, None]:
        """Time in seconds of current seek position."""
        return self._state.get('seek', None)

    @property
    def media_duration(self) -> Union[int, None, float, str]:
        """Time in seconds of current song duration."""
        return self._state.get('duration', None)

    @property
    def volume_level(self) -> float:
        """Volume level of the media player (0..1)."""
        volume = self._state.get('volume', None)
        if volume is not None and volume != '':
            volume = int(volume) / 100
        return volume

    @property
    def is_volume_muted(self) -> Union[str, None, bool]:
        """Boolean if volume is currently muted."""
        return self._state.get('mute', None)

    @property
    def shuffle(self) -> Union[bool, str]:
        """Boolean if shuffle is enabled."""
        return self._state.get('random', False)

    @property
    def repeat(self):
        """Return current repeat mode."""
        if self._state.get('repeat', None):
            return RepeatMode.ALL
        return RepeatMode.OFF

    async def async_media_next_track(self):
        """Send media_next command to media player."""
        await self._volumio.next()

    async def async_media_previous_track(self):
        """Send media_previous command to media player."""
        await self._volumio.previous()

    async def async_media_play(self):
        """Send media_play command to media player."""
        await self._volumio.play()

    async def async_media_pause(self):
        """Send media_pause command to media player."""
        if self._state.get('trackType') == 'webradio':
            await self._volumio.stop()
        else:
            await self._volumio.pause()

    async def async_media_stop(self):
        """Send media_stop command to media player."""
        await self._volumio.stop()

    async def async_set_volume_level(self, volume):
        """Send volume_up command to media player."""
        await self._volumio.set_volume_level(int(volume * 100))

    async def async_volume_up(self):
        """Service to send the Volumio the command for volume up."""
        await self._volumio.volume_up()

    async def async_volume_down(self):
        """Service to send the Volumio the command for volume down."""
        await self._volumio.volume_down()

    async def async_mute_volume(self, mute):
        """Send mute command to media player."""
        if mute:
            await self._volumio.mute()
        else:
            await self._volumio.unmute()

    async def async_set_shuffle(self, shuffle):
        """Enable/disable shuffle mode."""
        await self._volumio.set_shuffle(shuffle)

    async def async_set_repeat(self, repeat):
        """Set repeat mode."""
        if repeat == RepeatMode.OFF:
            await self._volumio.repeatAll('false')
        else:
            await self._volumio.repeatAll('true')

    async def async_select_source(self, source):
        """Choose an available playlist and play it."""
        await self._volumio.play_playlist(source)
        self._attr_source = source

    async def async_clear_playlist(self):
        """Clear players playlist."""
        await self._volumio.clear_playlist()
        self._attr_source = None

    @Throttle(PLAYLIST_UPDATE_INTERVAL)
    async def _async_update_playlists(self, **kwargs):
        """Update available Volumio playlists."""
        self._attr_source_list = await self._volumio.get_playlists()

    async def async_play_media(self, media_type, media_id, **kwargs):
        """Send the play_media command to the media player."""
        await self._volumio.replace_and_play(json.loads(media_id))

    async def async_browse_media(self, media_content_type=None, media_content_id=None):
        """Implement the websocket media browsing helper."""
        self.thumbnail_cache = {}
        if media_content_type in (None, 'library'):
            return await browse_top_level(self._volumio)
        return await browse_node(self, self._volumio, media_content_type, media_content_id)

    async def async_get_browse_image(self, media_content_type, media_content_id, media_image_id=None):
        """Get album art from Volumio."""
        cached_url = self.thumbnail_cache.get(media_content_id)
        image_url = self._volumio.canonic_url(cached_url)
        return await self._async_fetch_image(image_url)