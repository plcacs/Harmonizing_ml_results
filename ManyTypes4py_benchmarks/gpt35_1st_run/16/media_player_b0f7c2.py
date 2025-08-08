from __future__ import annotations
import asyncio
from collections import defaultdict
import logging
from typing import Any
from pyforked_daapd import ForkedDaapdAPI
from pylibrespot_java import LibrespotJavaAPI
from homeassistant.components import media_source
from homeassistant.components.media_player import ATTR_MEDIA_ANNOUNCE, ATTR_MEDIA_ENQUEUE, BrowseMedia, MediaPlayerEnqueue, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType, async_process_play_media_url
from homeassistant.components.spotify import async_browse_media as spotify_async_browse_media, is_spotify_media_type, resolve_spotify_media_type, spotify_uri_from_media_browser_url
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.dt import utcnow
from .browse_media import convert_to_owntone_uri, get_owntone_content, is_owntone_media_content_id, library
from .const import CALLBACK_TIMEOUT, CAN_PLAY_TYPE, CONF_LIBRESPOT_JAVA_PORT, CONF_MAX_PLAYLISTS, CONF_TTS_PAUSE_TIME, CONF_TTS_VOLUME, DEFAULT_TTS_PAUSE_TIME, DEFAULT_TTS_VOLUME, DEFAULT_UNMUTE_VOLUME, DOMAIN, FD_NAME, HASS_DATA_UPDATER_KEY, KNOWN_PIPES, PIPE_FUNCTION_MAP, SIGNAL_ADD_ZONES, SIGNAL_CONFIG_OPTIONS_UPDATE, SIGNAL_UPDATE_DATABASE, SIGNAL_UPDATE_MASTER, SIGNAL_UPDATE_OUTPUTS, SIGNAL_UPDATE_PLAYER, SIGNAL_UPDATE_QUEUE, SOURCE_NAME_CLEAR, SOURCE_NAME_DEFAULT, STARTUP_DATA, SUPPORTED_FEATURES, SUPPORTED_FEATURES_ZONE, TTS_TIMEOUT
from .coordinator import ForkedDaapdUpdater
_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    host: str = config_entry.data[CONF_HOST]
    port: int = config_entry.data[CONF_PORT]
    password: str = config_entry.data[CONF_PASSWORD]
    forked_daapd_api: ForkedDaapdAPI = ForkedDaapdAPI(async_get_clientsession(hass), host, port, password)
    forked_daapd_master: ForkedDaapdMaster = ForkedDaapdMaster(clientsession=async_get_clientsession(hass), api=forked_daapd_api, ip_address=host, config_entry=config_entry)

    @callback
    def async_add_zones(api: Any, outputs: Any) -> None:
        async_add_entities((ForkedDaapdZone(api, output, config_entry.entry_id) for output in outputs))
    config_entry.async_on_unload(async_dispatcher_connect(hass, SIGNAL_ADD_ZONES.format(config_entry.entry_id), async_add_zones))
    config_entry.async_on_unload(config_entry.add_update_listener(update_listener))
    if not hass.data.get(DOMAIN):
        hass.data[DOMAIN] = {config_entry.entry_id: {}}
    async_add_entities([forked_daapd_master], False)
    forked_daapd_updater: ForkedDaapdUpdater = ForkedDaapdUpdater(hass, forked_daapd_api, config_entry.entry_id)
    hass.data[DOMAIN][config_entry.entry_id][HASS_DATA_UPDATER_KEY] = forked_daapd_updater
    await forked_daapd_updater.async_init()

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    async_dispatcher_send(hass, SIGNAL_CONFIG_OPTIONS_UPDATE.format(entry.entry_id), entry.options)

class ForkedDaapdZone(MediaPlayerEntity):
    _attr_should_poll: bool = False

    def __init__(self, api: Any, output: Any, entry_id: str) -> None:
        self._api: Any = api
        self._output: Any = output
        self._output_id: str = output['id']
        self._last_volume: float = DEFAULT_UNMUTE_VOLUME
        self._available: bool = True
        self._entry_id: str = entry_id

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_UPDATE_OUTPUTS.format(self._entry_id), self._async_update_output_callback))

    @callback
    def _async_update_output_callback(self, outputs: Any, _event: Any = None) -> None:
        new_output = next((output for output in outputs if output['id'] == self._output_id), None)
        self._available = bool(new_output)
        if self._available:
            self._output = new_output
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        return f'{self._entry_id}-{self._output_id}'

    async def async_toggle(self) -> None:
        if self.state == MediaPlayerState.OFF:
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    @property
    def available(self) -> bool:
        return self._available

    async def async_turn_on(self) -> None:
        await self._api.change_output(self._output_id, selected=True)

    async def async_turn_off(self) -> None:
        await self._api.change_output(self._output_id, selected=False)

    @property
    def name(self) -> str:
        return f'{FD_NAME} output ({self._output['name']})'

    @property
    def state(self) -> MediaPlayerState:
        if self._output['selected']:
            return MediaPlayerState.ON
        return MediaPlayerState.OFF

    @property
    def volume_level(self) -> float:
        return self._output['volume'] / 100

    @property
    def is_volume_muted(self) -> bool:
        return self._output['volume'] == 0

    async def async_mute_volume(self, mute: bool) -> None:
        if mute:
            if self.volume_level == 0:
                return
            self._last_volume = self.volume_level
            target_volume = 0
        else:
            target_volume = self._last_volume
        await self.async_set_volume_level(volume=target_volume)

    async def async_set_volume_level(self, volume: float) -> None:
        await self._api.set_volume(volume=volume * 100, output_id=self._output_id)

    @property
    def supported_features(self) -> int:
        return SUPPORTED_FEATURES_ZONE
