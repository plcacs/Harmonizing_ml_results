from __future__ import annotations
from typing import ClassVar

class ChromecastInfo:
    cast_info: CastInfo
    is_dynamic_group: bool | None

    @property
    def friendly_name(self) -> str:
        ...

    @property
    def is_audio_group(self) -> bool:
        ...

    @property
    def uuid(self) -> str:
        ...

    def fill_out_missing_chromecast_info(self, hass: HomeAssistant) -> ChromecastInfo:
        ...

class ChromeCastZeroconf:
    __zconf: zeroconf.Zeroconf

    @classmethod
    def set_zeroconf(cls, zconf: zeroconf.Zeroconf) -> None:
        ...

    @classmethod
    def get_zeroconf(cls) -> zeroconf.Zeroconf:
        ...

class CastStatusListener:
    def __init__(self, cast_device, chromecast, mz_mgr, mz_only=False) -> None:
        ...

    def new_cast_status(self, status) -> None:
        ...

    def new_media_status(self, status) -> None:
        ...

    def load_media_failed(self, queue_item_id, error_code) -> None:
        ...

    def new_connection_status(self, status) -> None:
        ...

    def added_to_multizone(self, group_uuid) -> None:
        ...

    def removed_from_multizone(self, group_uuid) -> None:
        ...

    def multizone_new_cast_status(self, group_uuid, cast_status) -> None:
        ...

    def multizone_new_media_status(self, group_uuid, media_status) -> None:
        ...

    def invalidate(self) -> None:
        ...

class PlaylistItem:
    length: str
    title: str
    url: str

async def _fetch_playlist(hass: HomeAssistant, url: str, supported_content_types: tuple[str]) -> str:
    ...

async def parse_m3u(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    ...

async def parse_pls(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    ...

async def parse_playlist(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    ...
