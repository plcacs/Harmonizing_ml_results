from __future__ import annotations
from typing import Any

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class MediaroomDevice(MediaPlayerEntity):
    def set_state(self, mediaroom_state: State) -> None:

    def __init__(self, host: str, device_id: str, optimistic: bool = False, timeout: int = DEFAULT_TIMEOUT) -> None:

    async def async_added_to_hass(self) -> None:

    async def async_play_media(self, media_type: MediaType, media_id: str, **kwargs: Any) -> None:

    @property
    def unique_id(self) -> str:

    @property
    def name(self) -> str:

    @property
    def media_channel(self) -> Any:

    async def async_turn_on(self) -> None:

    async def async_turn_off(self) -> None:

    async def async_media_play(self) -> None:

    async def async_media_pause(self) -> None:

    async def async_media_stop(self) -> None:

    async def async_media_previous_track(self) -> None:

    async def async_media_next_track(self) -> None:

    async def async_volume_up(self) -> None:

    async def async_volume_down(self) -> None:

    async def async_mute_volume(self, mute: bool) -> None:
