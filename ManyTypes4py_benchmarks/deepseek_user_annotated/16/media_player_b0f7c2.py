"""Support forked_daapd media player."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
from typing import Any, cast

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
    def async_add_zones(api: ForkedDaapdAPI, outputs: list[dict[str, Any]]) -> None:
        async_add_entities(
            ForkedDaapdZone(api, output, config_entry.entry_id) for output in outputs
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
    hass.data[DOMAIN][config_entry.entry_id][HASS_DATA_UPDATER_KEY] = (
        forked_daapd_updater
    )
    await forked_daapd_updater.async_init()


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    async_dispatcher_send(
        hass, SIGNAL_CONFIG_OPTIONS_UPDATE.format(entry.entry_id), entry.options
    )


class ForkedDaapdZone(MediaPlayerEntity):
    """Representation of a forked-daapd output."""

    _attr_should_poll = False

    def __init__(self, api: ForkedDaapdAPI, output: dict[str, Any], entry_id: str) -> None:
        """Initialize the ForkedDaapd Zone."""
        self._api = api
        self._output = output
        self._output_id = output["id"]
        self._last_volume = DEFAULT_UNMUTE_VOLUME  # used for mute/unmute
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
    def _async_update_output_callback(self, outputs: list[dict[str, Any]], _event: asyncio.Event | None = None) -> None:
        new_output = next(
            (output for output in outputs if output["id"] == self._output_id), None
        )
        self._available = bool(new_output)
        if self._available:
            self._output = new_output
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return f"{self._entry_id}-{self._output_id}"

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
        return f"{FD_NAME} output ({self._output['name']})"

    @property
    def state(self) -> MediaPlayerState:
        """State of the zone."""
        if self._output["selected"]:
            return MediaPlayerState.ON
        return MediaPlayerState.OFF

    @property
    def volume_level(self) -> float | None:
        """Volume level of the media player (0..1)."""
        return self._output["volume"] / 100

    @property
    def is_volume_muted(self) -> bool:
        """Boolean if volume is currently muted."""
        return self._output["volume"] == 0

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute the volume."""
        if mute:
            if self.volume_level == 0:
                return
            self._last_volume = cast(float, self.volume_level)  # store volume level to restore later
            target_volume = 0
        else:
            target_volume = self._last_volume  # restore volume level
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
        # Leave the api public so the browse media helpers can use it
        self.api = api
        self._player = STARTUP_DATA[
            "player"
        ]  # _player, _outputs, and _queue are loaded straight from api
        self._outputs = STARTUP_DATA["outputs"]
        self._queue = STARTUP_DATA["queue"]
        self._track_info = defaultdict(
            str
        )  # _track info is found by matching _player data with _queue data
        self._last_outputs: list[dict[str, Any]] = []  # used for device on/off
        self._last_volume = DEFAULT_UNMUTE_VOLUME
        self._player_last_updated = None
        self._pipe_control_api: dict[str, Any] = {}
        self._ip_address = (
            ip_address  # need to save this because pipe control is on same ip
        )
        self._tts_pause_time = DEFAULT_TTS_PAUSE_TIME
        self._tts_volume = DEFAULT_TTS_VOLUME
        self._tts_requested = False
        self._tts_queued = False
        self._tts_playing_event = asyncio.Event()
        self._on_remove = None
        self._available = False
        self._clientsession = clientsession
        self._entry_id = config_entry.entry_id
        self.update_options(config_entry.options)
        self._paused_event = asyncio.Event()
        self._pause_requested = False
        self._sources_uris: dict[str, str | None] = {}
        self._source = SOURCE_NAME_DEFAULT
        self._max_playlists: int | None = None

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
    def update_options(self, options: dict[str, Any]) -> None:
        """Update forked-daapd server options."""
        if CONF_LIBRESPOT_JAVA_PORT in options:
            self._pipe_control_api["librespot-java"] = LibrespotJavaAPI(
                self._clientsession, self._ip_address, options[CONF_LIBRESPOT_JAVA_PORT]
            )
        if CONF_TTS_PAUSE_TIME in options:
            self._tts_pause_time = options[CONF_TTS_PAUSE_TIME]
        if CONF_TTS_VOLUME in options:
            self._tts_volume = options[CONF_TTS_VOLUME]
        if CONF_MAX_PLAYLISTS in options:
            # sources not updated until next _update_database call
            self._max_playlists = options[CONF_MAX_PLAYLISTS]

    @callback
    def _update_player(self, player: dict[str, Any], event: asyncio.Event) -> None:
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
    def _update_queue(self, queue: dict[str, Any], event: asyncio.Event) -> None:
        self._queue = queue
        if self._tts_requested:
            # Assume the change was due to the request
            self._tts_requested = False
            self._tts_queued = True

        if (
            self._queue["count"] >= 1
            and self._queue["items"][0]["data_kind"] == "pipe"
            and self._queue["items"][0]["title"] in KNOWN_PIPES
        ):  # if we're playing a pipe, set the source automatically so we can forward controls
            self._source = f"{self._queue['items'][0]['title']} (pipe)"
        self._update_track_info()
        event.set()

    @callback
    def _update_outputs(self, outputs: list[dict[str, Any]], event: asyncio.Event | None = None) -> None:
        if event:  # Calling without event is meant for zone, so ignore
            self._outputs = outputs
            event.set()

    @callback
    def _update_database(self, pipes: list[dict[str, Any]], playlists: list[dict[str, Any]], event: asyncio.Event) -> None:
        self._sources_uris = {SOURCE_NAME_CLEAR: None, SOURCE_NAME_DEFAULT: None}
        if pipes:
            self._sources_uris.update(
                {
                    f"{pipe['title']} (pipe)": pipe["uri"]
                    for pipe in pipes
                    if pipe["title"] in KNOWN_PIPES
                }
            )
        if playlists:
            self._sources_uris.update(
                {
                    f"{playlist['name']} (playlist)": playlist["uri"]
                    for playlist in playlists[: self._max_playlists]
                }
            )
        event.set()

    def _update_track_info(self) -> None:  # run during every player or queue update
        try:
            self._track_info = next(
                track
                for track in self._queue["items"]
                if track["id"] == self._player["item_id"]
            )
        except (StopIteration, TypeError, KeyError):
            _LOGGER.debug("Could not get track info")
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
        # restore state
        await self.api.set_volume(volume=self._last_volume * 100)
        if self._last_outputs:
            futures: list[asyncio.Task[int]] = [
                asyncio.create_task(
                    self.api.change_output(
                        output["id"],
                        selected=output["selected"],
                        volume=output["volume"],
                    )
                )
                for output in self._last_outputs
            ]
            await asyncio.wait(futures)
        else:  # enable all outputs
            await self.api.set_enabled_outputs(
                [output["id"] for output in self._outputs]
            )

    async def async_turn_off(self) -> None:
        """Pause player and store outputs state."""
        await self.async_media_pause()
        self._last_outputs = self._outputs
        if any(output["selected"] for output in self._outputs):
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
        return f"{FD_NAME} server"

    @property
    def state(self) -> MediaPlayerState | None:
        """State of the player."""
        if self._player["state"] == "play":
            return MediaPlayerState.PLAYING
        if self._player["state"] == "pause":
            return MediaPlayerState.PAUSED
        if not any(output["selected"] for output in self._outputs):
            return MediaPlayerState.OFF
        if self._player["state"] == "stop":  # this should catch all remaining cases
            return MediaPlayerState.ID