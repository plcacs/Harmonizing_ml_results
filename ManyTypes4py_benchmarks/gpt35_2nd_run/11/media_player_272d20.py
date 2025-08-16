from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import enum
import logging
from typing import Any, Dict, List, Union
from pyControl4.error_handling import C4Exception
from pyControl4.room import C4Room
from homeassistant.components.media_player import MediaPlayerDeviceClass, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState, MediaType
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from . import Control4ConfigEntry, Control4RuntimeData
from .director_utils import update_variables_for_config_entry
from .entity import Control4Entity

_LOGGER: logging.Logger = logging.getLogger(__name__)
CONTROL4_POWER_STATE: str = 'POWER_STATE'
CONTROL4_VOLUME_STATE: str = 'CURRENT_VOLUME'
CONTROL4_MUTED_STATE: str = 'IS_MUTED'
CONTROL4_CURRENT_VIDEO_DEVICE: str = 'CURRENT_VIDEO_DEVICE'
CONTROL4_PLAYING: str = 'PLAYING'
CONTROL4_PAUSED: str = 'PAUSED'
CONTROL4_STOPPED: str = 'STOPPED'
CONTROL4_MEDIA_INFO: str = 'CURRENT MEDIA INFO'
CONTROL4_PARENT_ID: str = 'parentId'
VARIABLES_OF_INTEREST: set[str] = {CONTROL4_POWER_STATE, CONTROL4_VOLUME_STATE, CONTROL4_MUTED_STATE, CONTROL4_CURRENT_VIDEO_DEVICE, CONTROL4_MEDIA_INFO, CONTROL4_PLAYING, CONTROL4_PAUSED, CONTROL4_STOPPED}

class _SourceType(enum.Enum):
    AUDIO: int = 1
    VIDEO: int = 2

@dataclass
class _RoomSource:
    source_type: set[_SourceType]
    idx: int
    name: str

async def get_rooms(hass: HomeAssistant, entry: Control4ConfigEntry) -> List[Dict[str, Any]]:
    return [item for item in entry.runtime_data.director_all_items if 'typeName' in item and item['typeName'] == 'room']

async def async_setup_entry(hass: HomeAssistant, entry: Control4ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    runtime_data: Control4RuntimeData = entry.runtime_data
    ui_config: Dict[str, Any] = runtime_data.ui_configuration
    if not ui_config:
        _LOGGER.debug('No UI Configuration found for Control4')
        return
    all_rooms: List[Dict[str, Any]] = await get_rooms(hass, entry)
    if not all_rooms:
        return
    scan_interval: int = runtime_data.scan_interval
    _LOGGER.debug('Scan interval = %s', scan_interval)

    async def async_update_data() -> Dict[int, Dict[str, Any]]:
        try:
            return await update_variables_for_config_entry(hass, entry, VARIABLES_OF_INTEREST)
        except C4Exception as err:
            raise UpdateFailed(f'Error communicating with API: {err}') from err

    coordinator: DataUpdateCoordinator[Dict[int, Dict[str, Any]]] = DataUpdateCoordinator(hass, _LOGGER, name='room', update_method=async_update_data, update_interval=timedelta(seconds=scan_interval))
    await coordinator.async_refresh()
    items_by_id: Dict[int, Dict[str, Any]] = {item['id']: item for item in runtime_data.director_all_items}
    item_to_parent_map: Dict[int, int] = {k: item['parentId'] for k, item in items_by_id.items() if 'parentId' in item and k > 1}
    entity_list: List[Control4Room] = []
    for room in all_rooms:
        room_id: int = room['id']
        sources: Dict[int, _RoomSource] = {}
        for exp in ui_config['experiences']:
            if room_id == exp['room_id']:
                exp_type: str = exp['type']
                if exp_type not in ('listen', 'watch'):
                    continue
                dev_type: _SourceType = _SourceType.AUDIO if exp_type == 'listen' else _SourceType.VIDEO
                for source in exp['sources']['source']:
                    dev_id: int = source['id']
                    name: str = items_by_id.get(dev_id, {}).get('name', f'Unknown Device - {dev_id}')
                    if dev_id in sources:
                        sources[dev_id].source_type.add(dev_type)
                    else:
                        sources[dev_id] = _RoomSource(source_type={dev_type}, idx=dev_id, name=name)
        try:
            hidden: bool = room['roomHidden']
            entity_list.append(Control4Room(runtime_data, coordinator, room['name'], room_id, item_to_parent_map, sources, hidden))
        except KeyError:
            _LOGGER.exception('Unknown device properties received from Control4: %s', room)
            continue
    async_add_entities(entity_list, True)

class Control4Room(Control4Entity, MediaPlayerEntity):
    _attr_has_entity_name: bool = True

    def __init__(self, runtime_data: Control4RuntimeData, coordinator: DataUpdateCoordinator[Dict[int, Dict[str, Any]]], name: str, room_id: int, id_to_parent: Dict[int, int], sources: Dict[int, _RoomSource], room_hidden: bool) -> None:
        super().__init__(runtime_data, coordinator, None, room_id, device_name=name, device_manufacturer=None, device_model=None, device_id=room_id)
        self._attr_entity_registry_enabled_default: bool = not room_hidden
        self._id_to_parent: Dict[int, int] = id_to_parent
        self._sources: Dict[int, _RoomSource] = sources
        self._attr_supported_features: int = MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PAUSE | MediaPlayerEntityFeature.STOP | MediaPlayerEntityFeature.VOLUME_MUTE | MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.VOLUME_STEP | MediaPlayerEntityFeature.TURN_OFF | MediaPlayerEntityFeature.SELECT_SOURCE

    def _create_api_object(self) -> C4Room:
        return C4Room(self.runtime_data.director, self._idx)

    def _get_device_from_variable(self, var: str) -> Union[int, None]:
        current_device: int = self.coordinator.data[self._idx][var]
        if current_device == 0:
            return None
        return current_device

    def _get_current_video_device_id(self) -> Union[int, None]:
        return self._get_device_from_variable(CONTROL4_CURRENT_VIDEO_DEVICE)

    def _get_current_playing_device_id(self) -> int:
        media_info: Dict[str, Any] = self._get_media_info()
        if media_info:
            if 'medSrcDev' in media_info:
                return media_info['medSrcDev']
            if 'deviceid' in media_info:
                return media_info['deviceid']
        return 0

    def _get_media_info(self) -> Union[Dict[str, Any], None]:
        media_info: Dict[str, Any] = self.coordinator.data[self._idx][CONTROL4_MEDIA_INFO]
        if 'mediainfo' in media_info:
            return media_info['mediainfo']
        return None

    def _get_current_source_state(self) -> Union[MediaPlayerState, None]:
        current_source: int = self._get_current_playing_device_id()
        while current_source:
            current_data: Dict[str, Any] = self.coordinator.data.get(current_source, None)
            if current_data:
                if current_data.get(CONTROL4_PLAYING, None):
                    return MediaPlayerState.PLAYING
                if current_data.get(CONTROL4_PAUSED, None):
                    return MediaPlayerState.PAUSED
                if current_data.get(CONTROL4_STOPPED, None):
                    return MediaPlayerState.ON
            current_source = self._id_to_parent.get(current_source, None)
        return None

    @property
    def device_class(self) -> MediaPlayerDeviceClass:
        for avail_source in self._sources.values():
            if _SourceType.VIDEO in avail_source.source_type:
                return MediaPlayerDeviceClass.TV
        return MediaPlayerDeviceClass.SPEAKER

    @property
    def state(self) -> MediaPlayerState:
        if (source_state := self._get_current_source_state()):
            return source_state
        if self.coordinator.data[self._idx][CONTROL4_POWER_STATE]:
            return MediaPlayerState.ON
        return MediaPlayerState.IDLE

    @property
    def source(self) -> Union[str, None]:
        current_source: int = self._get_current_playing_device_id()
        if not current_source or current_source not in self._sources:
            return None
        return self._sources[current_source].name

    @property
    def media_title(self) -> Union[str, None]:
        media_info: Dict[str, Any] = self._get_media_info()
        if not media_info:
            return None
        if 'title' in media_info:
            return media_info['title']
        current_source: int = self._get_current_playing_device_id()
        if not current_source or current_source not in self._sources:
            return None
        return self._sources[current_source].name

    @property
    def media_content_type(self) -> Union[MediaType, None]:
        current_source: int = self._get_current_playing_device_id()
        if not current_source:
            return None
        if current_source == self._get_current_video_device_id():
            return MediaType.VIDEO
        return MediaType.MUSIC

    async def async_media_play_pause(self) -> None:
        if self._get_current_source_state():
            await super().async_media_play_pause()

    @property
    def source_list(self) -> List[str]:
        return [x.name for x in self._sources.values()]

    @property
    def volume_level(self) -> float:
        return self.coordinator.data[self._idx][CONTROL4_VOLUME_STATE] / 100

    @property
    def is_volume_muted(self) -> bool:
        return bool(self.coordinator.data[self._idx][CONTROL4_MUTED_STATE])

    async def async_select_source(self, source: str) -> None:
        for avail_source in self._sources.values():
            if avail_source.name == source:
                audio_only: bool = _SourceType.VIDEO not in avail_source.source_type
                if audio_only:
                    await self._create_api_object().setAudioSource(avail_source.idx)
                else:
                    await self._create_api_object().setVideoAndAudioSource(avail_source.idx)
                break
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self) -> None:
        await self._create_api_object().setRoomOff()
        await self.coordinator.async_request_refresh()

    async def async_mute_volume(self, mute: bool) -> None:
        if mute:
            await self._create_api_object().setMuteOn()
        else:
            await self._create_api_object().setMuteOff()
        await self.coordinator.async_request_refresh()

    async def async_set_volume_level(self, volume: float) -> None:
        await self._create_api_object().setVolume(int(volume * 100))
        await self.coordinator.async_request_refresh()

    async def async_volume_up(self) -> None:
        await self._create_api_object().setIncrementVolume()
        await self.coordinator.async_request_refresh()

    async def async_volume_down(self) -> None:
        await self._create_api_object().setDecrementVolume()
        await self.coordinator.async_request_refresh()

    async def async_media_pause(self) -> None:
        await self._create_api_object().setPause()
        await self.coordinator.async_request_refresh()

    async def async_media_play(self) -> None:
        await self._create_api_object().setPlay()
        await self.coordinator.async_request_refresh()

    async def async_media_stop(self) -> None:
        await self._create_api_object().setStop()
        await self.coordinator.async_request_refresh()
