"""Support for interfacing to iTunes API."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests
import voluptuous as vol
from homeassistant.components.media_player import (
    PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
)
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

DEFAULT_NAME: str = 'iTunes'
DEFAULT_PORT: int = 8181
DEFAULT_SSL: bool = False
DEFAULT_TIMEOUT: int = 10

DOMAIN: str = 'itunes'
PLATFORM_SCHEMA = MEDIA_PLAYER_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_SSL, default=DEFAULT_SSL): cv.boolean,
})


class Itunes:
    """The iTunes API client."""

    def __init__(self, host: str, port: int, use_ssl: bool) -> None:
        """Initialize the iTunes device."""
        self.host: str = host
        self.port: int = port
        self.use_ssl: bool = use_ssl

    @property
    def _base_url(self) -> str:
        """Return the base URL for endpoints."""
        if self.use_ssl:
            uri_scheme = 'https://'
        else:
            uri_scheme = 'http://'
        if self.port:
            return f'{uri_scheme}{self.host}:{self.port}'
        return f'{uri_scheme}{self.host}'

    def _request(self, method: str, path: str, params: Optional[Any] = None) -> Any:
        """Make the actual request and return the parsed response."""
        url: str = f'{self._base_url}{path}'
        try:
            if method == 'GET':
                response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            elif method in ('POST', 'PUT'):
                response = requests.put(url, params, timeout=DEFAULT_TIMEOUT)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=DEFAULT_TIMEOUT)
            return response.json()
        except requests.exceptions.HTTPError:
            return {'player_state': 'error'}
        except requests.exceptions.RequestException:
            return {'player_state': 'offline'}

    def _command(self, named_command: str) -> Any:
        """Make a request for a controlling command."""
        return self._request('PUT', f'/{named_command}')

    def now_playing(self) -> Any:
        """Return the current state."""
        return self._request('GET', '/now_playing')

    def set_volume(self, level: int) -> Any:
        """Set the volume and returns the current state, level 0-100."""
        return self._request('PUT', '/volume', {'level': level})

    def set_muted(self, muted: bool) -> Any:
        """Mute and returns the current state, muted True or False."""
        return self._request('PUT', '/mute', {'muted': muted})

    def set_shuffle(self, shuffle: bool) -> Any:
        """Set the shuffle mode, shuffle True or False."""
        return self._request('PUT', '/shuffle', {'mode': 'songs' if shuffle else 'off'})

    def play(self) -> Any:
        """Set playback to play and returns the current state."""
        return self._command('play')

    def pause(self) -> Any:
        """Set playback to paused and returns the current state."""
        return self._command('pause')

    def next(self) -> Any:
        """Skip to the next track and returns the current state."""
        return self._command('next')

    def previous(self) -> Any:
        """Skip back and returns the current state."""
        return self._command('previous')

    def stop(self) -> Any:
        """Stop playback and return the current state."""
        return self._command('stop')

    def play_playlist(self, playlist_id_or_name: str) -> Any:
        """Set a playlist to be current and returns the current state."""
        response: Any = self._request('GET', '/playlists')
        playlists: List[Dict[str, Any]] = response.get('playlists', [])
        found_playlists: List[Dict[str, Any]] = [
            playlist for playlist in playlists if playlist_id_or_name in [playlist['name'], playlist['id']]
        ]
        if found_playlists:
            playlist: Dict[str, Any] = found_playlists[0]
            path: str = f"/playlists/{playlist['id']}/play"
            return self._request('PUT', path)
        raise ValueError(f'Playlist {playlist_id_or_name} not found')

    def artwork_url(self) -> str:
        """Return a URL of the current track's album art."""
        return f'{self._base_url}/artwork'

    def airplay_devices(self) -> Any:
        """Return a list of AirPlay devices."""
        return self._request('GET', '/airplay_devices')

    def airplay_device(self, device_id: str) -> Any:
        """Return an AirPlay device."""
        return self._request('GET', f'/airplay_devices/{device_id}')

    def toggle_airplay_device(self, device_id: str, toggle: bool) -> Any:
        """Toggle airplay device on or off, id, toggle True or False."""
        command: str = 'on' if toggle else 'off'
        path: str = f'/airplay_devices/{device_id}/{command}'
        return self._request('PUT', path)

    def set_volume_airplay_device(self, device_id: str, level: int) -> Any:
        """Set volume, returns current state of device, id, level 0-100."""
        path: str = f'/airplay_devices/{device_id}/volume'
        return self._request('PUT', path, {'level': level})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the iTunes platform."""
    add_entities([
        ItunesDevice(
            config.get(CONF_NAME),
            config.get(CONF_HOST),
            config.get(CONF_PORT),
            config[CONF_SSL],
            add_entities,
        )
    ])


class ItunesDevice(MediaPlayerEntity):
    """Representation of an iTunes API instance."""
    _attr_media_content_type: MediaType = MediaType.MUSIC
    _attr_supported_features: int = (
        MediaPlayerEntityFeature.PAUSE |
        MediaPlayerEntityFeature.VOLUME_SET |
        MediaPlayerEntityFeature.VOLUME_MUTE |
        MediaPlayerEntityFeature.PREVIOUS_TRACK |
        MediaPlayerEntityFeature.NEXT_TRACK |
        MediaPlayerEntityFeature.SEEK |
        MediaPlayerEntityFeature.PLAY_MEDIA |
        MediaPlayerEntityFeature.PLAY |
        MediaPlayerEntityFeature.TURN_OFF |
        MediaPlayerEntityFeature.SHUFFLE_SET
    )

    def __init__(self, name: str, host: str, port: int, use_ssl: bool, add_entities: AddEntitiesCallback) -> None:
        """Initialize the iTunes device."""
        self._name: str = name
        self._host: str = host
        self._port: int = port
        self._use_ssl: bool = use_ssl
        self._add_entities: AddEntitiesCallback = add_entities
        self.client: Itunes = Itunes(self._host, self._port, self._use_ssl)
        self.current_volume: Optional[int] = None
        self.muted: Optional[bool] = None
        self.shuffled: Optional[bool] = None
        self.current_title: Optional[str] = None
        self.current_album: Optional[str] = None
        self.current_artist: Optional[str] = None
        self.current_playlist: Optional[str] = None
        self.content_id: Optional[str] = None
        self.player_state: Optional[str] = None
        self.airplay_devices: Dict[str, AirPlayDevice] = {}
        self.update()

    def update_state(self, state_hash: Dict[str, Any]) -> None:
        """Update all the state properties with the passed in dictionary."""
        self.player_state = state_hash.get('player_state', None)
        self.current_volume = state_hash.get('volume', 0)
        self.muted = state_hash.get('muted', None)
        self.current_title = state_hash.get('name', None)
        self.current_album = state_hash.get('album', None)
        self.current_artist = state_hash.get('artist', None)
        self.current_playlist = state_hash.get('playlist', None)
        self.content_id = state_hash.get('id', None)
        _shuffle: Optional[str] = state_hash.get('shuffle', None)
        self.shuffled = _shuffle == 'songs'

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def state(self) -> str:
        """Return the state of the device."""
        if self.player_state == 'offline' or self.player_state is None:
            return 'offline'
        if self.player_state == 'error':
            return 'error'
        if self.player_state == 'stopped':
            return MediaPlayerState.IDLE
        if self.player_state == 'paused':
            return MediaPlayerState.PAUSED
        return MediaPlayerState.PLAYING

    def update(self) -> None:
        """Retrieve latest state."""
        now_playing: Any = self.client.now_playing()
        self.update_state(now_playing)
        found_devices: Any = self.client.airplay_devices()
        found_devices = found_devices.get('airplay_devices', [])
        new_devices: List[AirPlayDevice] = []
        for device_data in found_devices:
            device_id: str = device_data.get('id')
            if self.airplay_devices.get(device_id):
                airplay_device: AirPlayDevice = self.airplay_devices.get(device_id)
                airplay_device.update_state(device_data)
            else:
                airplay_device = AirPlayDevice(device_id, self.client)
                airplay_device.update_state(device_data)
                self.airplay_devices[device_id] = airplay_device
                new_devices.append(airplay_device)
        if new_devices:
            self._add_entities(new_devices)

    @property
    def is_volume_muted(self) -> Optional[bool]:
        """Boolean if volume is currently muted."""
        return self.muted

    @property
    def volume_level(self) -> Optional[float]:
        """Volume level of the media player (0..1)."""
        if self.current_volume is None:
            return None
        return self.current_volume / 100.0

    @property
    def media_content_id(self) -> Optional[str]:
        """Content ID of current playing media."""
        return self.content_id

    @property
    def media_image_url(self) -> str:
        """Image url of current playing media."""
        if self.player_state in {MediaPlayerState.PLAYING, MediaPlayerState.IDLE, MediaPlayerState.PAUSED} and self.current_title is not None:
            return f'{self.client.artwork_url()}?id={self.content_id}'
        return 'https://cloud.githubusercontent.com/assets/260/9829355/33fab972-58cf-11e5-8ea2-2ca74bdaae40.png'

    @property
    def media_title(self) -> Optional[str]:
        """Title of current playing media."""
        return self.current_title

    @property
    def media_artist(self) -> Optional[str]:
        """Artist of current playing media (Music track only)."""
        return self.current_artist

    @property
    def media_album_name(self) -> Optional[str]:
        """Album of current playing media (Music track only)."""
        return self.current_album

    @property
    def media_playlist(self) -> Optional[str]:
        """Title of the currently playing playlist."""
        return self.current_playlist

    @property
    def shuffle(self) -> Optional[bool]:
        """Boolean if shuffle is enabled."""
        return self.shuffled

    def set_volume_level(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        response: Any = self.client.set_volume(int(volume * 100))
        self.update_state(response)

    def mute_volume(self, mute: bool) -> None:
        """Mute (true) or unmute (false) media player."""
        response: Any = self.client.set_muted(mute)
        self.update_state(response)

    def set_shuffle(self, shuffle: bool) -> None:
        """Shuffle (true) or no shuffle (false) media player."""
        response: Any = self.client.set_shuffle(shuffle)
        self.update_state(response)

    def media_play(self) -> None:
        """Send media_play command to media player."""
        response: Any = self.client.play()
        self.update_state(response)

    def media_pause(self) -> None:
        """Send media_pause command to media player."""
        response: Any = self.client.pause()
        self.update_state(response)

    def media_next_track(self) -> None:
        """Send media_next command to media player."""
        response: Any = self.client.next()
        self.update_state(response)

    def media_previous_track(self) -> None:
        """Send media_previous command media player."""
        response: Any = self.client.previous()
        self.update_state(response)

    def play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:
        """Send the play_media command to the media player."""
        if media_type == MediaType.PLAYLIST:
            response: Any = self.client.play_playlist(media_id)
            self.update_state(response)

    def turn_off(self) -> None:
        """Turn the media player off."""
        response: Any = self.client.stop()
        self.update_state(response)


class AirPlayDevice(MediaPlayerEntity):
    """Representation an AirPlay device via an iTunes API instance."""
    _attr_media_content_type: MediaType = MediaType.MUSIC
    _attr_supported_features: int = (
        MediaPlayerEntityFeature.VOLUME_SET |
        MediaPlayerEntityFeature.TURN_ON |
        MediaPlayerEntityFeature.TURN_OFF
    )

    def __init__(self, device_id: str, client: Itunes) -> None:
        """Initialize the AirPlay device."""
        self._id: str = device_id
        self.client: Itunes = client
        self.device_name: str = 'AirPlay'
        self.kind: Optional[str] = None
        self.active: Optional[bool] = False
        self.selected: Optional[bool] = False
        self.volume: int = 0
        self.supports_audio: Optional[bool] = False
        self.supports_video: Optional[bool] = False
        self.player_state: Optional[str] = None

    def update_state(self, state_hash: Dict[str, Any]) -> None:
        """Update all the state properties with the passed in dictionary."""
        if 'player_state' in state_hash:
            self.player_state = state_hash.get('player_state', None)
        if 'name' in state_hash:
            name: str = state_hash.get('name', '')
            self.device_name = f'{name} AirTunes Speaker'.strip()
        if 'kind' in state_hash:
            self.kind = state_hash.get('kind', None)
        if 'active' in state_hash:
            self.active = state_hash.get('active', None)
        if 'selected' in state_hash:
            self.selected = state_hash.get('selected', None)
        if 'sound_volume' in state_hash:
            self.volume = state_hash.get('sound_volume', 0)
        if 'supports_audio' in state_hash:
            self.supports_audio = state_hash.get('supports_audio', None)
        if 'supports_video' in state_hash:
            self.supports_video = state_hash.get('supports_video', None)

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self.device_name

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        if self.selected is True:
            return 'mdi:volume-high'
        return 'mdi:volume-off'

    @property
    def state(self) -> str:
        """Return the state of the device."""
        if self.selected is True:
            return MediaPlayerState.ON
        return MediaPlayerState.OFF

    def update(self) -> None:
        """Retrieve latest state."""
        # This method could be implemented to fetch updated state if needed.
        pass

    @property
    def volume_level(self) -> float:
        """Return the volume."""
        return float(self.volume) / 100.0

    def set_volume_level(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        volume_int: int = int(volume * 100)
        response: Any = self.client.set_volume_airplay_device(self._id, volume_int)
        self.update_state(response)

    def turn_on(self) -> None:
        """Select AirPlay."""
        self.update_state({'selected': True})
        self.schedule_update_ha_state()
        response: Any = self.client.toggle_airplay_device(self._id, True)
        self.update_state(response)

    def turn_off(self) -> None:
        """Deselect AirPlay."""
        self.update_state({'selected': False})
        self.schedule_update_ha_state()
        response: Any = self.client.toggle_airplay_device(self._id, False)
        self.update_state(response)