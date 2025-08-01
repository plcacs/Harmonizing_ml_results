"""Support forked_daapd media player."""
from __future__ import annotations
import asyncio
from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Union
from pyforked_daapd import ForkedDaapdAPI
from pylibrespot_java import LibrespotJavaAPI
from homeassistant.components import media_source
from homeassistant.components.media_player import (
    ATTR_MEDIA_ANNOUNCE,
    ATTR_MEDIA_ENQUEUE,
    BrowseMedia,
    MediaPlayerEnqueue,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
    async_process_play_media_url,
)
from homeassistant.components.spotify import (
    async_browse_media as spotify_async_browse_media,
    is_spotify_media_type,
    resolve_spotify_media_type,
    spotify_uri_from_media_browser_url,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.dt import utcnow
from .browse_media import (
    convert_to_owntone_uri,
    get_owntone_content,
    is_owntone_media_content_id,
    library,
)
from .const import (
    CALLBACK_TIMEOUT,
    CAN_PLAY_TYPE,
    CONF_LIBRESPOT_JAVA_PORT,
    CONF_MAX_PLAYLISTS,
    CONF_TTS_PAUSE_TIME,
    CONF_TTS_VOLUME,
    DEFAULT_TTS_PAUSE_TIME,
    DEFAULT_TTS_VOLUME,
    DEFAULT_UNMUTE_VOLUME,
    DOMAIN,
    FD_NAME,
    HASS_DATA_UPDATER_KEY,
    KNOWN_PIPES,
    PIPE_FUNCTION_MAP,
    SIGNAL_ADD_ZONES,
    SIGNAL_CONFIG_OPTIONS_UPDATE,
    SIGNAL_UPDATE_DATABASE,
    SIGNAL_UPDATE_MASTER,
    SIGNAL_UPDATE_OUTPUTS,
    SIGNAL_UPDATE_PLAYER,
    SIGNAL_UPDATE_QUEUE,
    SOURCE_NAME_CLEAR,
    SOURCE_NAME_DEFAULT,
    STARTUP_DATA,
    SUPPORTED_FEATURES,
    SUPPORTED_FEATURES_ZONE,
    TTS_TIMEOUT,
)
from .coordinator import ForkedDaapdUpdater

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up forked-daapd from a config entry."""
    host: str = config_entry.data[CONF_HOST]
    port: int = config_entry.data[CONF_PORT]
    password: str = config_entry.data[CONF_PASSWORD]
    forked_daapd_api = ForkedDaapdAPI(
        async_get_clientsession(hass), host, port, password
    )
    forked_daapd_master = ForkedDaapdMaster(
        clientsession=async_get_clientsession(hass),
        api=forked_daapd_api,
        ip_address=host,
        config_entry=config_entry,
    )

    @callback
    def async_add_zones(api: ForkedDaapdAPI, outputs: List[Dict[str, Any]]) -> None:
        async_add_entities(
            (ForkedDaapdZone(api, output, config_entry.entry_id) for output in outputs)
        )

    config_entry.async_on_unload(
        async_dispatcher_connect(
            hass, SIGNAL_ADD_ZONES.format(config_entry.entry_id), async_add_zones
        )
    )
    config_entry.async_on_unload(config_entry.add_update_listener(update_listener))
    if not hass.data.get(DOMAIN):
        hass.data[DOMAIN] = {config_entry.entry_id: {}}
    async_add_entities([forked_daapd_master], False)
    forked_daapd_updater = ForkedDaapdUpdater(
        hass, forked_daapd_api, config_entry.entry_id
    )
    hass.data[DOMAIN][config_entry.entry_id][HASS_DATA_UPDATER_KEY] = forked_daapd_updater
    await forked_daapd_updater.async_init()

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    async_dispatcher_send(
        hass, SIGNAL_CONFIG_OPTIONS_UPDATE.format(entry.entry_id), entry.options
    )

class ForkedDaapdZone(MediaPlayerEntity):
    """Representation of a forked-daapd output."""
    _attr_should_poll = False

    def __init__(self, api: ForkedDaapdAPI, output: Dict[str, Any], entry_id: str) -> None:
        """Initialize the ForkedDaapd Zone."""
        self._api = api
        self._output = output
        self._output_id = output['id']
        self._last_volume = DEFAULT_UNMUTE_VOLUME
        self._available = True
        self._entry_id = entry_id

    async def async_added_to_hass(self) -> None:
        """Use lifecycle hooks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_OUTPUTS.format(self._entry_id),
                self._async_update_output_callback,
            )
        )

    @callback
    def _async_update_output_callback(
        self, outputs: List[Dict[str, Any]], _event: Optional[Any] = None
    ) -> None:
        new_output = next(
            (output for output in outputs if output['id'] == self._output_id), None
        )
        self._available = bool(new_output)
        if self._available:
            self._output = new_output
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return f'{self._entry_id}-{self._output_id}'

    async def async_toggle(self) -> None:
        """Toggle the power on the zone."""
        if self.state == MediaPlayerState.OFF:
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    @property
    def available(self) -> bool:
        """Return whether the zone is available."""
        return self._available

    async def async_turn_on(self) -> None:
        """Enable the output."""
        await self._api.change_output(self._output_id, selected=True)

    async def async_turn_off(self) -> None:
        """Disable the output."""
        await self._api.change_output(self._output_id, selected=False)

    @property
    def name(self) -> str:
        """Return the name of the zone."""
        return f'{FD_NAME} output ({self._output["name"]})'

    @property
    def state(self) -> MediaPlayerState:
        """State of the zone."""
        if self._output['selected']:
            return MediaPlayerState.ON
        return MediaPlayerState.OFF

    @property
    def volume_level(self) -> float:
        """Volume level of the media player (0..1)."""
        return self._output['volume'] / 100

    @property
    def is_volume_muted(self) -> bool:
        """Boolean if volume is currently muted."""
        return self._output['volume'] == 0

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute the volume."""
        if mute:
            if self.volume_level == 0:
                return
            self._last_volume = self.volume_level
            target_volume = 0
        else:
            target_volume = self._last_volume
        await self.async_set_volume_level(volume=target_volume)

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume - input range [0,1]."""
        await self._api.set_volume(volume=volume * 100, output_id=self._output_id)

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        """Flag media player features that are supported."""
        return SUPPORTED_FEATURES_ZONE

class ForkedDaapdMaster(MediaPlayerEntity):
    """Representation of the main forked-daapd device."""
    _attr_should_poll = False

    def __init__(
        self,
        clientsession: Any,
        api: ForkedDaapdAPI,
        ip_address: str,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the ForkedDaapd Master Device."""
        self.api = api
        self._player = STARTUP_DATA['player']
        self._outputs = STARTUP_DATA['outputs']
        self._queue = STARTUP_DATA['queue']
        self._track_info = defaultdict(str)
        self._last_outputs: List[Dict[str, Any]] = []
        self._last_volume = DEFAULT_UNMUTE_VOLUME
        self._player_last_updated: Optional[Any] = None
        self._pipe_control_api: Dict[str, LibrespotJavaAPI] = {}
        self._ip_address = ip_address
        self._tts_pause_time = DEFAULT_TTS_PAUSE_TIME
        self._tts_volume = DEFAULT_TTS_VOLUME
        self._tts_requested = False
        self._tts_queued = False
        self._tts_playing_event = asyncio.Event()
        self._on_remove: Optional[Any] = None
        self._available = False
        self._clientsession = clientsession
        self._entry_id = config_entry.entry_id
        self.update_options(config_entry.options)
        self._paused_event = asyncio.Event()
        self._pause_requested = False
        self._sources_uris: Dict[str, Optional[str]] = {}
        self._source = SOURCE_NAME_DEFAULT
        self._max_playlists: Optional[int] = None

    async def async_added_to_hass(self) -> None:
        """Use lifecycle hooks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_PLAYER.format(self._entry_id),
                self._update_player,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_QUEUE.format(self._entry_id),
                self._update_queue,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_OUTPUTS.format(self._entry_id),
                self._update_outputs,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_MASTER.format(self._entry_id),
                self._update_callback,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_CONFIG_OPTIONS_UPDATE.format(self._entry_id),
                self.update_options,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_UPDATE_DATABASE.format(self._entry_id),
                self._update_database,
            )
        )

    @callback
    def _update_callback(self, available: bool) -> None:
        """Call update method."""
        self._available = available
        self.async_write_ha_state()

    @callback
    def update_options(self, options: Dict[str, Any]) -> None:
        """Update forked-daapd server options."""
        if CONF_LIBRESPOT_JAVA_PORT in options:
            self._pipe_control_api['librespot-java'] = LibrespotJavaAPI(
                self._clientsession, self._ip_address, options[CONF_LIBRESPOT_JAVA_PORT]
            )
        if CONF_TTS_PAUSE_TIME in options:
            self._tts_pause_time = options[CONF_TTS_PAUSE_TIME]
        if CONF_TTS_VOLUME in options:
            self._tts_volume = options[CONF_TTS_VOLUME]
        if CONF_MAX_PLAYLISTS in options:
            self._max_playlists = options[CONF_MAX_PLAYLISTS]

    @callback
    def _update_player(self, player: Dict[str, Any], event: Any) -> None:
        self._player = player
        self._player_last_updated = utcnow()
        self._update_track_info()
        if self._tts_queued:
            self._tts_playing_event.set()
            self._tts_queued = False
        if self._pause_requested:
            self._paused_event.set()
            self._pause_requested = False
        event.set()

    @callback
    def _update_queue(self, queue: Dict[str, Any], event: Any) -> None:
        self._queue = queue
        if self._tts_requested:
            self._tts_requested = False
            self._tts_queued = True
        if (
            self._queue['count'] >= 1
            and self._queue['items'][0]['data_kind'] == 'pipe'
            and (self._queue['items'][0]['title'] in KNOWN_PIPES)
        ):
            self._source = f'{self._queue["items"][0]["title"]} (pipe)'
        self._update_track_info()
        event.set()

    @callback
    def _update_outputs(self, outputs: List[Dict[str, Any]], event: Optional[Any] = None) -> None:
        if event:
            self._outputs = outputs
            event.set()

    @callback
    def _update_database(
        self, pipes: List[Dict[str, Any]], playlists: List[Dict[str, Any]], event: Any
    ) -> None:
        self._sources_uris = {SOURCE_NAME_CLEAR: None, SOURCE_NAME_DEFAULT: None}
        if pipes:
            self._sources_uris.update(
                {
                    f'{pipe["title"]} (pipe)': pipe['uri']
                    for pipe in pipes
                    if pipe['title'] in KNOWN_PIPES
                }
            )
        if playlists:
            self._sources_uris.update(
                {
                    f'{playlist["name"]} (playlist)': playlist['uri']
                    for playlist in playlists[: self._max_playlists]
                }
            )
        event.set()

    def _update_track_info(self) -> None:
        try:
            self._track_info = next(
                (
                    track
                    for track in self._queue['items']
                    if track['id'] == self._player['item_id']
                )
            )
        except (StopIteration, TypeError, KeyError):
            _LOGGER.debug('Could not get track info')
            self._track_info = defaultdict(str)

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return self._entry_id

    @property
    def available(self) -> bool:
        """Return whether the master is available."""
        return self._available

    async def async_turn_on(self) -> None:
        """Restore the last on outputs state."""
        await self.api.set_volume(volume=self._last_volume * 100)
        if self._last_outputs:
            futures = [
                asyncio.create_task(
                    self.api.change_output(
                        output['id'], selected=output['selected'], volume=output['volume']
                    )
                )
                for output in self._last_outputs
            ]
            await asyncio.wait(futures)
        else:
            await self.api.set_enabled_outputs([output['id'] for output in self._outputs])

    async def async_turn_off(self) -> None:
        """Pause player and store outputs state."""
        await self.async_media_pause()
        self._last_outputs = self._outputs
        if any((output['selected'] for output in self._outputs)):
            await self.api.set_enabled_outputs([])

    async def async_toggle(self) -> None:
        """Toggle the power on the device.

        Default media player component method counts idle as off.
        We consider idle to be on but just not playing.
        """
        if self.state == MediaPlayerState.OFF:
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return f'{FD_NAME} server'

    @property
    def state(self) -> Optional[MediaPlayerState]:
        """State of the player."""
        if self._player['state'] == 'play':
            return MediaPlayerState.PLAYING
        if self._player['state'] == 'pause':
            return MediaPlayerState.PAUSED
        if not any((output['selected'] for output in self._outputs)):
            return MediaPlayerState.OFF
        if self._player['state'] == 'stop':
            return MediaPlayerState.IDLE
        return None

    @property
    def volume_level(self) -> float:
        """Volume level of the media player (0..1)."""
        return self._player['volume'] / 100

    @property
    def is_volume_muted(self) -> bool:
        """Boolean if volume is currently muted."""
        return self._player['volume'] == 0

    @property
    def media_content_id(self) -> str:
        """Content ID of current playing media."""
        return self._player['item_id']

    @property
    def media_content_type(self) -> str:
        """Content type of current playing media."""
        return self._track_info['media_kind']

    @property
    def media_duration(self) -> float:
        """Duration of current playing media in seconds."""
        return self._player['item_length_ms'] / 1000

    @property
    def media_position(self) -> float:
        """Position of current playing media in seconds."""
        return self._player['item_progress_ms'] / 1000

    @property
    def media_position_updated_at(self) -> Optional[Any]:
        """When was the position of the current playing media valid."""
        return self._player_last_updated

    @property
    def media_title(self) -> str:
        """Title of current playing media."""
        if self._track_info['data_kind'] == 'url':
            return self._track_info['album']
        return self._track_info['title']

    @property
    def media_artist(self) -> str:
        """Artist of current playing media, music track only."""
        return self._track_info['artist']

    @property
    def media_album_name(self) -> str:
        """Album name of current playing media, music track only."""
        if self._track_info['data_kind'] == 'url':
            return self._track_info['title']
        return self._track_info['album']

    @property
    def media_album_artist(self) -> str:
        """Album artist of current playing media, music track only."""
        return self._track_info['album_artist']

    @property
    def media_track(self) -> str:
        """Track number of current playing media, music track only."""
        return self._track_info['track_number']

    @property
    def shuffle(self) -> bool:
        """Boolean if shuffle is enabled."""
        return self._player['shuffle']

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        """Flag media player features that are supported."""
        return SUPPORTED_FEATURES

    @property
    def source(self) -> str:
        """Name of the current input source."""
        return self._source

    @property
    def source_list(self) -> List[str]:
        """List of available input sources."""
        return [*self._sources_uris]

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute the volume."""
        if mute:
            if self.volume_level == 0:
                return
            self._last_volume = self.volume_level
            target_volume = 0
        else:
            target_volume = self._last_volume
        await self.api.set_volume(volume=target_volume * 100)

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume - input range [0,1]."""
        await self.api.set_volume(volume=volume * 100)

    async def async_media_play(self) -> None:
        """Start playback."""
        if self._use_pipe_control():
            await self._pipe_call(self._use_pipe_control(), 'async_media_play')
        else:
            await self.api.start_playback()

    async def async_media_pause(self) -> None:
        """Pause playback."""
        if self._use_pipe_control():
            await self._pipe_call(self._use_pipe_control(), 'async_media_pause')
        else:
            await self.api.pause_playback()

    async def async_media_stop(self) -> None:
        """Stop playback."""
        if self._use_pipe_control():
            await self._pipe_call(self._use_pipe_control(), 'async_media_stop')
        else:
            await self.api.stop_playback()

    async def async_media_previous_track(self) -> None:
        """Skip to previous track."""
        if self._use_pipe_control():
            await self._pipe_call(self._use_pipe_control(), 'async_media_previous_track')
        else:
            await self.api.previous_track()

    async def async_media_next_track(self) -> None:
        """Skip to next track."""
        if self._use_pipe_control():
            await self._pipe_call(self._use_pipe_control(), 'async_media_next_track')
        else:
            await self.api.next_track()

    async def async_media_seek(self, position: float) -> None:
        """Seek to position."""
        await self.api.seek(position_ms=position * 1000)

    async def async_clear_playlist(self) -> None:
        """Clear playlist."""
        await self.api.clear_queue()

    async def async_set_shuffle(self, shuffle: bool) -> None:
        """Enable/disable shuffle mode."""
        await self.api.shuffle(shuffle)

    @property
    def media_image_url(self) -> Optional[str]:
        """Image url of current playing media."""
        if (url := self._track_info.get('artwork_url')):
            url = self.api.full_url(url)
        return url

    async def _save_and_set_tts_volumes(self) -> None:
        if self.volume_level:
            self._last_volume = self.volume_level
        self._last_outputs = self._outputs
        if self._outputs:
            await self.api.set_volume(volume=self._tts_volume * 100)
            futures = [
                asyncio.create_task(
                    self.api.change_output(
                        output['id'], selected=True, volume=self._tts_volume * 100
                    )
                )
                for output in self._outputs
            ]
            await asyncio.wait(futures)

    async def _pause_and_wait_for_callback(self) -> None:
        """Send pause and wait for the pause callback to be received."""
        self._pause_requested = True
        await self.async_media_pause()
        try:
            async with asyncio.timeout(CALLBACK_TIMEOUT):
                await self._paused_event.wait()
        except TimeoutError:
            self._pause_requested = False
        self._paused_event.clear()

    async def async_play_media(
        self, media_type: str, media_id: str, **kwargs: Any
    ) -> None:
        """Play a URI."""
        if media_source.is_media_source_id(media_id):
            media_type = MediaType.MUSIC
            play_item = await media_source.async_resolve_media(
                self.hass, media_id, self.entity_id
            )
            media_id = play_item.url
        elif is_owntone_media_content_id(media_id):
            media_id = convert_to_owntone_uri(media_id)
        elif is_spotify_media_type(media_type):
            media_type = resolve_spotify_media_type(media_type)
            media_id = spotify_uri_from_media_browser_url(media_id)
        if media_type not in CAN_PLAY_TYPE:
            _LOGGER.warning("Media type '%s' not supported", media_type)
            return
        if media_type == MediaType.MUSIC:
            media_id = async_process_play_media_url(self.hass, media_id)
        elif media_type not in CAN_PLAY_TYPE:
            _LOGGER.warning("Media type '%s' not supported", media_type)
            return
        if kwargs.get(ATTR_MEDIA_ANNOUNCE):
            await self._async_announce(media_id)
            return
        enqueue = kwargs.get(ATTR_MEDIA_ENQUEUE, MediaPlayerEnqueue.REPLACE)
        if enqueue in {True, MediaPlayerEnqueue.ADD, MediaPlayerEnqueue.REPLACE}:
            await self.api.add_to_queue(
                uris=media_id, playback='start', clear=enqueue == MediaPlayerEnqueue.REPLACE
            )
            return
        current_position = next(
            (
                item['position']
                for item in self._queue['items']
                if item['id'] == self._player['item_id']
            ),
            0,
        )
        if enqueue == MediaPlayerEnqueue.NEXT:
            await self.api.add_to_queue(
                uris=media_id, playback='start', position=current_position + 1
            )
            return
        await self.api.add_to_queue(
            uris=media_id,
            playback='start',
            position=current_position,
            playback_from_position=current_position,
        )

    async def _async_announce(self, media_id: str) -> None:
        """Play a URI."""
        saved_state = self.state
        saved_mute = self.is_volume_muted
        sleep_future = asyncio.create_task(asyncio.sleep(self._tts_pause_time))
        await self._pause_and_wait_for_callback()
        await self._save_and_set_tts_volumes()
        saved_song_position = self._player['item_progress_ms']
        saved_queue = self._queue if self._queue['count'] > 0 else None
        if saved_queue:
            saved_queue_position = next(
                (
                    i
                    for i, item in enumerate(saved_queue['items'])
                    if item['id'] == self._player['item_id']
                )
            )
        self._tts_requested = True
        await sleep_future
        await self.api.add_to_queue(uris=media_id, playback='start', clear=True)
        try:
            async with asyncio.timeout(TTS_TIMEOUT):
                await self._tts_playing_event.wait()
        except TimeoutError:
            self._tts_requested = False
            _LOGGER.warning('TTS request timed out')
        await asyncio.sleep(
            self._queue['items'][0]['length_ms'] / 1000 + self._tts_pause_time
        )
        self._tts_playing_event.clear()
        await self.async_turn_on()
        if saved_mute:
            await self.async_mute_volume(True)
        if self._use_pipe_control():
            await self.api.add_to_queue(uris=self._sources_uris[self._source], clear=True)
            if saved_state == MediaPlayerState.PLAYING:
                await self.async_media_play()
            return
        if not saved_queue:
            return
        await self.api.add_to_queue(
            uris=','.join((item['uri'] for item in saved_queue['items'])),
            playback='start',
            playback_from_position=saved_queue_position,
            clear=True,
        )
        await self.api.seek(position_ms=saved_song_position)
        if saved_state == MediaPlayerState.PAUSED:
            await self.async_media_pause()
            return
        if saved_state != MediaPlayerState.PLAYING:
            await self.async_media_stop()

    async def async_select_source(self, source: str) -> None:
        """Change source.

        Source name reflects whether in default mode or pipe mode.
        Selecting playlists/clear sets the playlists/clears but ends up in default mode.
        """
        if source == self._source:
            return
        if self._use_pipe_control():
            await self._pause_and_wait_for_callback()
        self._source = source
        if not self._use_pipe_control():
            self._source = SOURCE_NAME_DEFAULT
        if self._sources_uris.get(source):
            await self.api.add_to_queue(uris=self._sources_uris[source], clear=True)
        elif source == SOURCE_NAME_CLEAR:
            await self.api.clear_queue()
        self.async_write_ha_state()

    def _use_pipe_control(self) -> str:
        """Return which pipe control from KNOWN_PIPES to use."""
        if self._source[-7:] == ' (pipe)':
            return self._source[:-7]
        return ''

    async def _pipe_call(self, pipe_name: str, base_function_name: str) -> None:
        if (pipe := self._pipe_control_api.get(pipe_name)):
            await getattr(pipe, PIPE_FUNCTION_MAP[pipe_name][base_function_name])()
            return
        _LOGGER.warning('No pipe control available for %s', pipe_name)

    async def async_browse_media(
        self, media_content_type: Optional[str] = None, media_content_id: Optional[str] = None
    ) -> BrowseMedia:
        """Implement the websocket media browsing helper."""
        other_sources: List[BrowseMedia] = []
        if media_content_id is None or media_source.is_media_source_id(media_content_id):
            ms_result = await media_source.async_browse_media(
                self.hass,
                media_content_id,
                content_filter=lambda bm: bm.media_content_type in CAN_PLAY_TYPE,
            )
            if media_content_type is not None:
                return ms_result
            other_sources = list(ms_result.children) if ms_result.children else []
        if 'spotify' in self.hass.config.components and (
            media_content_type is None or is_spotify_media_type(media_content_type)
        ):
            spotify_result = await spotify_async_browse_media(
                self.hass, media_content_type, media_content_id
            )
            if media_content_type is not None:
                return spotify_result
            if spotify_result.children:
                other_sources += spotify_result.children
        if media_content_id is None or media_content_type is None:
            return library(other_sources)
        return await get_owntone_content(self, media_content_id)

    async def async_get_browse_image(
        self,
        media_content_type: str,
        media_content_id: str,
        media_image_id: Optional[str] = None,
    ) -> Union[tuple[Optional[bytes], Optional[str]], tuple[None, None]]:
        """Fetch image for media browser."""
        if media_content_type not in {MediaType.TRACK, MediaType.ALBUM, MediaType.ARTIST}:
            return (None, None)
        owntone_uri = convert_to_owntone_uri(media_content_id)
        item_id_str = owntone_uri.rsplit(':', maxsplit=1)[-1]
        result: Optional[Dict[str, Any]] = None
        if media_content_type == MediaType.TRACK:
            result = await self.api.get_track(int(item_id_str))
        elif media_content_type == MediaType.ALBUM:
            if (result := (await self.api.get_albums())):
                result = next((item for item in result if item['id'] == item_id_str), None)
        elif (result := (await self.api.get_artists())):
            result = next((item for item in result if item['id'] == item_id_str), None)
        if result and (url := result.get('artwork_url')):
            return await self._async_fetch_image(self.api.full_url(url))
        return (None, None)
