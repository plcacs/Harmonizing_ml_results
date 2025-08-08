from __future__ import annotations
from typing import ClassVar, List, Optional, Union

class ChromecastInfo:
    cast_info: CastInfo
    is_dynamic_group: Optional[Union[bool, None]]

    def fill_out_missing_chromecast_info(self, hass: HomeAssistant) -> ChromecastInfo:

class ChromeCastZeroconf:
    __zconf: zeroconf.Zeroconf

    @classmethod
    def set_zeroconf(cls, zconf: zeroconf.Zeroconf) -> None:

    @classmethod
    def get_zeroconf(cls) -> zeroconf.Zeroconf:

class CastStatusListener:
    _cast_device: CastDevice
    _uuid: str
    _valid: bool
    _mz_mgr: MultiZoneManager

    def __init__(self, cast_device: CastDevice, chromecast: pychromecast.Chromecast, mz_mgr: MultiZoneManager, mz_only: bool) -> None:

    def new_cast_status(self, status: pychromecast.controllers.media.MediaStatus) -> None:

    def new_media_status(self, status: pychromecast.controllers.media.MediaStatus) -> None:

    def load_media_failed(self, queue_item_id: int, error_code: int) -> None:

    def new_connection_status(self, status: pychromecast.socket_client.ConnectionStatus) -> None:

    def added_to_multizone(self, group_uuid: str) -> None:

    def removed_from_multizone(self, group_uuid: str) -> None:

    def multizone_new_cast_status(self, group_uuid: str, cast_status: pychromecast.controllers.media.MediaStatus) -> None:

    def multizone_new_media_status(self, group_uuid: str, media_status: pychromecast.controllers.media.MediaStatus) -> None:

    def invalidate(self) -> None:

class PlaylistItem:
    length: Optional[str]
    title: Optional[str]
    url: str

async def _fetch_playlist(hass: HomeAssistant, url: str, supported_content_types: Tuple[str]) -> str:

async def parse_m3u(hass: HomeAssistant, url: str) -> List[PlaylistItem]:

async def parse_pls(hass: HomeAssistant, url: str) -> List[PlaylistItem]:

async def parse_playlist(hass: HomeAssistant, url: str) -> List[PlaylistItem]:
