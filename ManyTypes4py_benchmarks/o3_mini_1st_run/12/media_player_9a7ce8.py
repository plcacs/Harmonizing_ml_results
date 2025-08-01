"""Support to interact with a Music Player Daemon."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
import hashlib
import logging
import os
from socket import gaierror
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import mpd
from mpd.asyncio import MPDClient
import voluptuous as vol
from homeassistant.components import media_source
from homeassistant.components.media_player import (
    PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA,
    BrowseMedia,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
    RepeatMode,
    async_process_play_media_url,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import Throttle, dt as dt_util
from .const import DOMAIN, LOGGER

DEFAULT_NAME: str = 'MPD'
DEFAULT_PORT: int = 6600
PLAYLIST_UPDATE_INTERVAL: timedelta = timedelta(seconds=120)
SUPPORT_MPD: MediaPlayerEntityFeature = (
    MediaPlayerEntityFeature.PAUSE
    | MediaPlayerEntityFeature.PREVIOUS_TRACK
    | MediaPlayerEntityFeature.NEXT_TRACK
    | MediaPlayerEntityFeature.PLAY_MEDIA
    | MediaPlayerEntityFeature.PLAY
    | MediaPlayerEntityFeature.CLEAR_PLAYLIST
    | MediaPlayerEntityFeature.REPEAT_SET
    | MediaPlayerEntityFeature.SHUFFLE_SET
    | MediaPlayerEntityFeature.SEEK
    | MediaPlayerEntityFeature.STOP
    | MediaPlayerEntityFeature.TURN_OFF
    | MediaPlayerEntityFeature.TURN_ON
    | MediaPlayerEntityFeature.BROWSE_MEDIA
)
PLATFORM_SCHEMA = MEDIA_PLAYER_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PASSWORD): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
})


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up media player from config_entry."""
    async_add_entities(
        [MpdDevice(entry.data[CONF_HOST], entry.data[CONF_PORT], entry.data.get(CONF_PASSWORD), entry.entry_id)],
        True,
    )


class MpdDevice(MediaPlayerEntity):
    """Representation of a MPD server."""

    _attr_media_content_type: str = MediaType.MUSIC
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None

    def __init__(self, server: str, port: int, password: Optional[str], unique_id: str) -> None:
        """Initialize the MPD device."""
        self.server: str = server
        self.port: int = port
        self._attr_unique_id: str = unique_id
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, unique_id)}, entry_type=DeviceEntryType.SERVICE)
        self.password: Optional[str] = password
        self._status: Dict[str, Any] = {}
        self._currentsong: Optional[Dict[str, Any]] = None
        self._current_playlist: Optional[str] = None
        self._muted_volume: Optional[float] = None
        self._media_image_hash: Optional[str] = None
        self._media_image_file: Optional[str] = None
        self._client: MPDClient = MPDClient()
        self._client.timeout = 30
        self._client.idletimeout = 10
        self._client_lock: asyncio.Lock = asyncio.Lock()

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[None, None]:
        """Handle MPD connect and disconnect."""
        async with self._client_lock:
            try:
                try:
                    async with asyncio.timeout(self._client.timeout + 5):
                        await self._client.connect(self.server, self.port)
                except TimeoutError as error:
                    raise TimeoutError('Connection attempt timed out') from error
                if self.password is not None:
                    await self._client.password(self.password)
                self._attr_available = True
                yield
            except (TimeoutError, gaierror, mpd.ConnectionError, OSError) as error:
                log_level = logging.DEBUG
                if self._attr_available is not False:
                    log_level = logging.WARNING
                LOGGER.log(log_level, "Error connecting to '%s': %s", self.server, error)
                self._attr_available = False
                self._status = {}
                yield
            finally:
                with suppress(mpd.ConnectionError):
                    self._client.disconnect()

    async def async_update(self) -> None:
        """Get the latest data from MPD and update the state."""
        async with self.connection():
            try:
                self._status = await self._client.status()
                self._currentsong = await self._client.currentsong()
                await self._async_update_media_image_hash()
                position: Optional[Any] = self._status.get('elapsed')
                if position is None:
                    position = self._status.get('time')
                    if isinstance(position, str) and ':' in position:
                        position = position.split(':')[0]
                if position is not None and self._attr_media_position != position:
                    self._attr_media_position_updated_at = dt_util.utcnow()
                    self._attr_media_position = int(float(position))
                await self._update_playlists()
            except (mpd.ConnectionError, ValueError) as error:
                LOGGER.debug('Error updating status: %s', error)

    @property
    def state(self) -> MediaPlayerState:
        """Return the media state."""
        if not self._status:
            return MediaPlayerState.OFF
        if self._status.get('state') == 'play':
            return MediaPlayerState.PLAYING
        if self._status.get('state') == 'pause':
            return MediaPlayerState.PAUSED
        if self._status.get('state') == 'stop':
            return MediaPlayerState.OFF
        return MediaPlayerState.OFF

    @property
    def media_content_id(self) -> Optional[str]:
        """Return the content ID of current playing media."""
        if self._currentsong:
            return self._currentsong.get('file')
        return None

    @property
    def media_duration(self) -> Optional[str]:
        """Return the duration of current playing media in seconds."""
        if self._currentsong is not None and (currentsong_time := self._currentsong.get('time')):
            return currentsong_time
        time_from_status: Any = self._status.get('time')
        if isinstance(time_from_status, str) and ':' in time_from_status:
            return time_from_status.split(':')[1]
        return None

    @property
    def media_title(self) -> str:
        """Return the title of current playing media."""
        if self._currentsong is None:
            return "None"
        name: Optional[str] = self._currentsong.get('name', None)
        title: Optional[str] = self._currentsong.get('title', None)
        file_name: Optional[str] = self._currentsong.get('file', None)
        if name is None and title is None:
            if file_name is None:
                return 'None'
            return os.path.basename(file_name)
        if name is None:
            return title  # type: ignore
        if title is None:
            return name
        return f'{name}: {title}'

    @property
    def media_artist(self) -> Optional[str]:
        """Return the artist of current playing media (Music track only)."""
        if self._currentsong is None:
            return None
        artists: Any = self._currentsong.get('artist')
        if isinstance(artists, list):
            return ', '.join(artists)
        return artists

    @property
    def media_album_name(self) -> Optional[str]:
        """Return the album of current playing media (Music track only)."""
        if self._currentsong is None:
            return None
        return self._currentsong.get('album')

    @property
    def media_image_hash(self) -> Optional[str]:
        """Hash value for media image."""
        return self._media_image_hash

    async def async_get_media_image(self) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch media image of current playing track."""
        async with self.connection():
            if self._currentsong is None or not (file := self._currentsong.get('file')):
                return (None, None)
            with suppress(mpd.ConnectionError):
                response: Optional[Dict[str, Any]] = await self._async_get_file_image_response(file)
            if response is None:
                return (None, None)
            image: bytes = bytes(response['binary'])
            mime: str = response.get('type', 'image/png')
            return (image, mime)

    async def _async_update_media_image_hash(self) -> None:
        """Update the hash value for the media image."""
        if self._currentsong is None:
            return
        file: Optional[str] = self._currentsong.get('file')
        if file == self._media_image_file:
            return
        if file is not None and (response := (await self._async_get_file_image_response(file))) is not None:
            self._media_image_hash = hashlib.sha256(bytes(response['binary'])).hexdigest()[:16]
        else:
            self._media_image_hash = None
        self._media_image_file = file

    async def _async_get_file_image_response(self, file: str) -> Optional[Dict[str, Any]]:
        commands: List[str] = []
        with suppress(mpd.ConnectionError):
            commands = list(await self._client.commands())
        can_albumart: bool = 'albumart' in commands
        can_readpicture: bool = 'readpicture' in commands
        response: Optional[Dict[str, Any]] = None
        if can_readpicture:
            try:
                with suppress(mpd.ConnectionError):
                    response = await self._client.readpicture(file)
            except mpd.CommandError as error:
                if error.errno is not mpd.FailureResponseCode.NO_EXIST:
                    LOGGER.warning('Retrieving artwork through `readpicture` command failed: %s', error)
        if can_albumart and (not response):
            try:
                with suppress(mpd.ConnectionError):
                    response = await self._client.albumart(file)
            except mpd.CommandError as error:
                if error.errno is not mpd.FailureResponseCode.NO_EXIST:
                    LOGGER.warning('Retrieving artwork through `albumart` command failed: %s', error)
        if not response:
            return None
        return response

    @property
    def volume_level(self) -> Optional[float]:
        """Return the volume level."""
        if 'volume' in self._status:
            return int(self._status['volume']) / 100
        return None

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        """Flag media player features that are supported."""
        if not self._status:
            return MediaPlayerEntityFeature(0)
        supported: MediaPlayerEntityFeature = SUPPORT_MPD
        if 'volume' in self._status:
            supported |= (
                MediaPlayerEntityFeature.VOLUME_SET
                | MediaPlayerEntityFeature.VOLUME_STEP
                | MediaPlayerEntityFeature.VOLUME_MUTE
            )
        if self._attr_source_list is not None:  # type: ignore[attr-defined]
            supported |= MediaPlayerEntityFeature.SELECT_SOURCE
        return supported

    @property
    def source(self) -> Optional[str]:
        """Name of the current input source."""
        return self._current_playlist

    async def async_select_source(self, source: str) -> None:
        """Choose a different available playlist and play it."""
        await self.async_play_media(MediaType.PLAYLIST, source)

    @Throttle(PLAYLIST_UPDATE_INTERVAL)
    async def _update_playlists(self, **kwargs: Any) -> None:
        """Update available MPD playlists."""
        try:
            self._attr_source_list = []  # type: ignore[attr-defined]
            with suppress(mpd.ConnectionError):
                for playlist_data in await self._client.listplaylists():
                    self._attr_source_list.append(playlist_data['playlist'])  # type: ignore[attr-defined]
        except mpd.CommandError as error:
            self._attr_source_list = None  # type: ignore[attr-defined]
            LOGGER.warning('Playlists could not be updated: %s:', error)

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume of media player."""
        async with self.connection():
            if 'volume' in self._status:
                await self._client.setvol(int(volume * 100))

    async def async_volume_up(self) -> None:
        """Service to send the MPD the command for volume up."""
        async with self.connection():
            if 'volume' in self._status:
                current_volume: int = int(self._status['volume'])
                if current_volume <= 100:
                    await self._client.setvol(current_volume + 5)

    async def async_volume_down(self) -> None:
        """Service to send the MPD the command for volume down."""
        async with self.connection():
            if 'volume' in self._status:
                current_volume: int = int(self._status['volume'])
                if current_volume >= 0:
                    await self._client.setvol(current_volume - 5)

    async def async_media_play(self) -> None:
        """Service to send the MPD the command for play/pause."""
        async with self.connection():
            if self._status.get('state') == 'pause':
                await self._client.pause(0)
            else:
                await self._client.play()

    async def async_media_pause(self) -> None:
        """Service to send the MPD the command for play/pause."""
        async with self.connection():
            await self._client.pause(1)

    async def async_media_stop(self) -> None:
        """Service to send the MPD the command for stop."""
        async with self.connection():
            await self._client.stop()

    async def async_media_next_track(self) -> None:
        """Service to send the MPD the command for next track."""
        async with self.connection():
            await self._client.next()

    async def async_media_previous_track(self) -> None:
        """Service to send the MPD the command for previous track."""
        async with self.connection():
            await self._client.previous()

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute. Emulated with set_volume_level."""
        if 'volume' in self._status:
            if mute:
                self._muted_volume = self.volume_level
                await self.async_set_volume_level(0)
            elif self._muted_volume is not None:
                await self.async_set_volume_level(self._muted_volume)
            self._attr_is_volume_muted = mute  # type: ignore[attr-defined]

    async def async_play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:
        """Send the media player the command for playing a playlist."""
        async with self.connection():
            if media_source.is_media_source_id(media_id):
                media_type = MediaType.MUSIC
                play_item = await media_source.async_resolve_media(self.hass, media_id, self.entity_id)  # type: ignore[attr-defined]
                media_id = async_process_play_media_url(self.hass, play_item.url)
            if media_type == MediaType.PLAYLIST:
                LOGGER.debug('Playing playlist: %s', media_id)
                if self._attr_source_list and media_id in self._attr_source_list:  # type: ignore[attr-defined]
                    self._current_playlist = media_id
                else:
                    self._current_playlist = None
                    LOGGER.warning('Unknown playlist name %s', media_id)
                await self._client.clear()
                await self._client.load(media_id)
                await self._client.play()
            else:
                await self._client.clear()
                self._current_playlist = None
                await self._client.add(media_id)
                await self._client.play()

    @property
    def repeat(self) -> RepeatMode:
        """Return current repeat mode."""
        if self._status.get('repeat') == '1':
            if self._status.get('single') == '1':
                return RepeatMode.ONE
            return RepeatMode.ALL
        return RepeatMode.OFF

    async def async_set_repeat(self, repeat: RepeatMode) -> None:
        """Set repeat mode."""
        async with self.connection():
            if repeat == RepeatMode.OFF:
                await self._client.repeat(0)
                await self._client.single(0)
            else:
                await self._client.repeat(1)
                if repeat == RepeatMode.ONE:
                    await self._client.single(1)
                else:
                    await self._client.single(0)

    @property
    def shuffle(self) -> bool:
        """Boolean if shuffle is enabled."""
        return bool(int(self._status.get('random', 0)))

    async def async_set_shuffle(self, shuffle: bool) -> None:
        """Enable/disable shuffle mode."""
        async with self.connection():
            await self._client.random(int(shuffle))

    async def async_turn_off(self) -> None:
        """Service to send the MPD the command to stop playing."""
        async with self.connection():
            await self._client.stop()

    async def async_turn_on(self) -> None:
        """Service to send the MPD the command to start playing."""
        async with self.connection():
            await self._client.play()
            await self._update_playlists(no_throttle=True)

    async def async_clear_playlist(self) -> None:
        """Clear players playlist."""
        async with self.connection():
            await self._client.clear()

    async def async_media_seek(self, position: int) -> None:
        """Send seek command."""
        async with self.connection():
            await self._client.seekcur(position)

    async def async_browse_media(
        self, media_content_type: Optional[str] = None, media_content_id: Optional[str] = None
    ) -> BrowseMedia:
        """Implement the websocket media browsing helper."""
        async with self.connection():
            return await media_source.async_browse_media(self.hass, media_content_id, content_filter=lambda item: item.media_content_type.startswith('audio/'))
