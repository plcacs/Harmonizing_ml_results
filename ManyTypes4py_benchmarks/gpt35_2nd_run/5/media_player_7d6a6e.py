from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from pylgnetcast import LG_COMMAND, LgNetCastClient, LgNetCastError
from requests import RequestException
from homeassistant.components.media_player import MediaPlayerDeviceClass, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_HOST, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.trigger import PluggableAction
from .const import ATTR_MANUFACTURER, DOMAIN
from .triggers.turn_on import async_get_turn_on_trigger

DEFAULT_NAME: str = 'LG TV Remote'
CONF_ON_ACTION: str = 'turn_on_action'
SUPPORT_LGTV: int = MediaPlayerEntityFeature.PAUSE | MediaPlayerEntityFeature.VOLUME_STEP | MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.VOLUME_MUTE | MediaPlayerEntityFeature.PREVIOUS_TRACK | MediaPlayerEntityFeature.NEXT_TRACK | MediaPlayerEntityFeature.TURN_OFF | MediaPlayerEntityFeature.SELECT_SOURCE | MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PLAY_MEDIA | MediaPlayerEntityFeature.STOP

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    host: str = config_entry.data[CONF_HOST]
    access_token: str = config_entry.data[CONF_ACCESS_TOKEN]
    unique_id: str = config_entry.unique_id
    name: str = config_entry.data.get(CONF_NAME, DEFAULT_NAME)
    model: str = config_entry.data[CONF_MODEL]
    client: LgNetCastClient = LgNetCastClient(host, access_token)
    hass.data[DOMAIN][config_entry.entry_id] = client
    async_add_entities([LgTVDevice(client, name, model, unique_id=unique_id)])

class LgTVDevice(MediaPlayerEntity):
    _attr_assumed_state: bool = True
    _attr_device_class: str = MediaPlayerDeviceClass.TV
    _attr_media_content_type: str = MediaType.CHANNEL
    _attr_has_entity_name: bool = True
    _attr_name: str = None

    def __init__(self, client: LgNetCastClient, name: str, model: str, unique_id: str) -> None:
        self._client: LgNetCastClient = client
        self._muted: bool = False
        self._turn_on: PluggableAction = PluggableAction(self.async_write_ha_state)
        self._volume: int = 0
        self._channel_id: int = None
        self._channel_name: str = ''
        self._program_name: str = ''
        self._sources: Dict[str, Any] = {}
        self._source_names: List[str] = []
        self._attr_unique_id: str = unique_id
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, unique_id)}, manufacturer=ATTR_MANUFACTURER, name=name, model=model)

    async def async_added_to_hass(self) -> None:
        entry: ConfigEntry = self.registry_entry
        if TYPE_CHECKING:
            assert entry is not None and entry.device_id is not None
        self.async_on_remove(self._turn_on.async_register(self.hass, async_get_turn_on_trigger(entry.device_id)))

    def send_command(self, command: str) -> None:
        try:
            with self._client as client:
                client.send_command(command)
        except (LgNetCastError, RequestException):
            self._attr_state = MediaPlayerState.OFF

    def update(self) -> None:
        try:
            with self._client as client:
                self._attr_state = MediaPlayerState.ON
                self.__update_volume()
                channel_info = client.query_data('cur_channel')
                if channel_info:
                    channel_info = channel_info[0]
                    channel_id = channel_info.find('major')
                    self._channel_name = channel_info.find('chname').text
                    self._program_name = channel_info.find('progName').text
                    if channel_id is not None:
                        self._channel_id = int(channel_id.text)
                    if self._channel_name is None:
                        self._channel_name = channel_info.find('inputSourceName').text
                    if self._program_name is None:
                        self._program_name = channel_info.find('labelName').text
                channel_list = client.query_data('channel_list')
                if channel_list:
                    channel_names = []
                    for channel in channel_list:
                        channel_name = channel.find('chname')
                        if channel_name is not None:
                            channel_names.append(str(channel_name.text))
                    self._sources = dict(zip(channel_names, channel_list, strict=False))
                    source_tuples = [(k, source.find('major').text) for k, source in self._sources.items()]
                    sorted_sources = sorted(source_tuples, key=lambda channel: int(channel[1]))
                    self._source_names = [n for n, k in sorted_sources]
        except (LgNetCastError, RequestException):
            self._attr_state = MediaPlayerState.OFF

    def __update_volume(self) -> None:
        volume_info: Tuple[int, bool] = self._client.get_volume()
        if volume_info:
            volume, muted = volume_info
            self._volume = volume
            self._muted = muted

    @property
    def is_volume_muted(self) -> bool:
        return self._muted

    @property
    def volume_level(self) -> float:
        return self._volume / 100.0

    @property
    def source(self) -> str:
        return self._channel_name

    @property
    def source_list(self) -> List[str]:
        return self._source_names

    @property
    def media_content_id(self) -> int:
        return self._channel_id

    @property
    def media_channel(self) -> str:
        return self._channel_name

    @property
    def media_title(self) -> str:
        return self._program_name

    @property
    def supported_features(self) -> int:
        if self._turn_on:
            return SUPPORT_LGTV | MediaPlayerEntityFeature.TURN_ON
        return SUPPORT_LGTV

    @property
    def media_image_url(self) -> str:
        return f'{self._client.url}data?target=screen_image&_={datetime.now().timestamp()}'

    def turn_off(self) -> None:
        self.send_command(LG_COMMAND.POWER)

    async def async_turn_on(self) -> None:
        await self._turn_on.async_run(self.hass, self._context)

    def volume_up(self) -> None:
        self.send_command(LG_COMMAND.VOLUME_UP)

    def volume_down(self) -> None:
        self.send_command(LG_COMMAND.VOLUME_DOWN)

    def set_volume_level(self, volume: float) -> None:
        self._client.set_volume(float(volume * 100))

    def mute_volume(self, mute: bool) -> None:
        self.send_command(LG_COMMAND.MUTE_TOGGLE)

    def select_source(self, source: str) -> None:
        self._client.change_channel(self._sources[source])

    def media_play(self) -> None:
        self.send_command(LG_COMMAND.PLAY)

    def media_pause(self) -> None:
        self.send_command(LG_COMMAND.PAUSE)

    def media_stop(self) -> None:
        self.send_command(LG_COMMAND.STOP)

    def media_next_track(self) -> None:
        self.send_command(LG_COMMAND.FAST_FORWARD)

    def media_previous_track(self) -> None:
        self.send_command(LG_COMMAND.REWIND)

    def play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:
        if media_type != MediaType.CHANNEL:
            raise ValueError(f'Invalid media type: {media_type}')
        for name, channel in self._sources.items():
            channel_id = channel.find('major')
            if channel_id is not None and int(channel_id.text) == int(media_id):
                self.select_source(name)
                return
        raise ValueError(f'Invalid media id: {media_id}')
