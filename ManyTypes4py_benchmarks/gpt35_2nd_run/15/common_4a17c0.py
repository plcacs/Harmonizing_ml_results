from homeassistant.components.media_player import ATTR_INPUT_SOURCE, ATTR_MEDIA_CONTENT_ID, ATTR_MEDIA_CONTENT_TYPE, ATTR_MEDIA_ENQUEUE, ATTR_MEDIA_SEEK_POSITION, ATTR_MEDIA_VOLUME_LEVEL, ATTR_MEDIA_VOLUME_MUTED, DOMAIN, SERVICE_CLEAR_PLAYLIST, SERVICE_PLAY_MEDIA, SERVICE_SELECT_SOURCE, MediaPlayerEnqueue
from homeassistant.const import ATTR_ENTITY_ID, ENTITY_MATCH_ALL, SERVICE_MEDIA_NEXT_TRACK, SERVICE_MEDIA_PAUSE, SERVICE_MEDIA_PLAY, SERVICE_MEDIA_PLAY_PAUSE, SERVICE_MEDIA_PREVIOUS_TRACK, SERVICE_MEDIA_SEEK, SERVICE_MEDIA_STOP, SERVICE_TOGGLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, SERVICE_VOLUME_DOWN, SERVICE_VOLUME_MUTE, SERVICE_VOLUME_SET, SERVICE_VOLUME_UP
from homeassistant.core import HomeAssistant
from homeassistant.loader import bind_hass

async def async_turn_on(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def turn_on(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_turn_off(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def turn_off(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_toggle(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def toggle(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_volume_up(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def volume_up(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_volume_down(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def volume_down(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_mute_volume(hass: HomeAssistant, mute: bool, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def mute_volume(hass: HomeAssistant, mute: bool, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_set_volume_level(hass: HomeAssistant, volume: float, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def set_volume_level(hass: HomeAssistant, volume: float, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_play_pause(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_play_pause(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_play(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_play(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_pause(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_pause(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_stop(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_stop(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_next_track(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_next_track(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_previous_track(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_previous_track(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_media_seek(hass: HomeAssistant, position: int, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def media_seek(hass: HomeAssistant, position: int, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_play_media(hass: HomeAssistant, media_type: str, media_id: str, entity_id: str = ENTITY_MATCH_ALL, enqueue: bool = None) -> None:
    ...

@bind_hass
def play_media(hass: HomeAssistant, media_type: str, media_id: str, entity_id: str = ENTITY_MATCH_ALL, enqueue: bool = None) -> None:
    ...

async def async_select_source(hass: HomeAssistant, source: str, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def select_source(hass: HomeAssistant, source: str, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

async def async_clear_playlist(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...

@bind_hass
def clear_playlist(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
    ...
