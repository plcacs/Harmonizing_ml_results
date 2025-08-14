from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp
import attr
import pychromecast
from pychromecast import dial
from pychromecast.const import CAST_TYPE_GROUP
from pychromecast.models import CastInfo

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.components import zeroconf

_LOGGER: logging.Logger = logging.getLogger(__name__)

_PLS_SECTION_PLAYLIST: str = "playlist"


@attr.s(slots=True, frozen=True)
class ChromecastInfo:
    cast_info: CastInfo = attr.ib()
    is_dynamic_group: Optional[bool] = attr.ib(default=None)

    @property
    def friendly_name(self) -> str:
        return self.cast_info.friendly_name

    @property
    def is_audio_group(self) -> bool:
        return self.cast_info.cast_type == CAST_TYPE_GROUP

    @property
    def uuid(self) -> bool:
        return self.cast_info.uuid

    def fill_out_missing_chromecast_info(self, hass: HomeAssistant) -> ChromecastInfo:
        cast_info: CastInfo = self.cast_info
        if self.cast_info.cast_type is None or self.cast_info.manufacturer is None:
            unknown_models: dict[str, Tuple[Any, Any]] = hass.data[DOMAIN]["unknown_models"]
            if self.cast_info.model_name not in unknown_models:
                cast_info = dial.get_cast_type(
                    cast_info,
                    zconf=ChromeCastZeroconf.get_zeroconf(),
                )
                unknown_models[self.cast_info.model_name] = (
                    cast_info.cast_type,
                    cast_info.manufacturer,
                )
                report_issue: str = (
                    "create a bug report at "
                    "https://github.com/home-assistant/core/issues?q=is%3Aopen+is%3Aissue"
                    "+label%3A%22integration%3A+cast%22"
                )
                _LOGGER.debug(
                    (
                        "Fetched cast details for unknown model '%s' manufacturer:"
                        " '%s', type: '%s'. Please %s"
                    ),
                    cast_info.model_name,
                    cast_info.manufacturer,
                    cast_info.cast_type,
                    report_issue,
                )
            else:
                cast_type, manufacturer = unknown_models[self.cast_info.model_name]
                cast_info = CastInfo(
                    cast_info.services,
                    cast_info.uuid,
                    cast_info.model_name,
                    cast_info.friendly_name,
                    cast_info.host,
                    cast_info.port,
                    cast_type,
                    manufacturer,
                )

        if not self.is_audio_group or self.is_dynamic_group is not None:
            return ChromecastInfo(cast_info=cast_info)

        is_dynamic_group: bool = False
        http_group_status: Any = dial.get_multizone_status(
            None,
            services=self.cast_info.services,
            zconf=ChromeCastZeroconf.get_zeroconf(),
        )
        if http_group_status is not None:
            is_dynamic_group = any(
                g.uuid == self.cast_info.uuid for g in http_group_status.dynamic_groups
            )

        return ChromecastInfo(
            cast_info=cast_info,
            is_dynamic_group=is_dynamic_group,
        )


class ChromeCastZeroconf:
    __zconf: ClassVar[Optional[zeroconf.HaZeroconf]] = None

    @classmethod
    def set_zeroconf(cls, zconf: zeroconf.HaZeroconf) -> None:
        cls.__zconf = zconf

    @classmethod
    def get_zeroconf(cls) -> Optional[zeroconf.HaZeroconf]:
        return cls.__zconf


class CastStatusListener(
    pychromecast.controllers.media.MediaStatusListener,
    pychromecast.controllers.multizone.MultiZoneManagerListener,
    pychromecast.controllers.receiver.CastStatusListener,
    pychromecast.socket_client.ConnectionStatusListener,
):
    def __init__(
        self,
        cast_device: Any,
        chromecast: pychromecast.Chromecast,
        mz_mgr: Any,
        mz_only: bool = False,
    ) -> None:
        self._cast_device: Any = cast_device
        self._uuid: Any = chromecast.uuid
        self._valid: bool = True
        self._mz_mgr: Any = mz_mgr

        if cast_device._cast_info.is_audio_group:  # noqa: SLF001
            self._mz_mgr.add_multizone(chromecast)
        if mz_only:
            return

        chromecast.register_status_listener(self)
        chromecast.socket_client.media_controller.register_status_listener(self)
        chromecast.register_connection_listener(self)
        if not cast_device._cast_info.is_audio_group:  # noqa: SLF001
            self._mz_mgr.register_listener(chromecast.uuid, self)

    def new_cast_status(self, status: Any) -> None:
        if self._valid:
            self._cast_device.new_cast_status(status)

    def new_media_status(self, status: Any) -> None:
        if self._valid:
            self._cast_device.new_media_status(status)

    def load_media_failed(self, queue_item_id: Any, error_code: Any) -> None:
        if self._valid:
            self._cast_device.load_media_failed(queue_item_id, error_code)

    def new_connection_status(self, status: Any) -> None:
        if self._valid:
            self._cast_device.new_connection_status(status)

    def added_to_multizone(self, group_uuid: Any) -> None:
        pass

    def removed_from_multizone(self, group_uuid: Any) -> None:
        if self._valid:
            self._cast_device.multizone_new_media_status(group_uuid, None)

    def multizone_new_cast_status(self, group_uuid: Any, cast_status: Any) -> None:
        pass

    def multizone_new_media_status(self, group_uuid: Any, media_status: Any) -> None:
        if self._valid:
            self._cast_device.multizone_new_media_status(group_uuid, media_status)

    def invalidate(self) -> None:
        if self._cast_device._cast_info.is_audio_group:  # noqa: SLF001
            self._mz_mgr.remove_multizone(self._uuid)
        else:
            self._mz_mgr.deregister_listener(self._uuid, self)
        self._valid = False


class PlaylistError(Exception):
    pass


class PlaylistSupported(PlaylistError):
    pass


@dataclass
class PlaylistItem:
    length: Optional[str]
    title: Optional[str]
    url: str


def _is_url(url: str) -> bool:
    result = urlparse(url)
    return all([result.scheme, result.netloc])


async def _fetch_playlist(
    hass: HomeAssistant, url: str, supported_content_types: Tuple[str, ...]
) -> str:
    try:
        session: aiohttp.ClientSession = aiohttp_client.async_get_clientsession(hass, verify_ssl=False)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            charset: str = resp.charset or "utf-8"
            if resp.content_type in supported_content_types:
                raise PlaylistSupported
            try:
                content = await resp.content.read(64 * 1024)
                playlist_data: str = content.decode(charset)
            except ValueError as err:
                raise PlaylistError(f"Could not decode playlist {url}") from err
    except TimeoutError as err:
        raise PlaylistError(f"Timeout while fetching playlist {url}") from err
    except aiohttp.client_exceptions.ClientError as err:
        raise PlaylistError(f"Error while fetching playlist {url}") from err

    return playlist_data


async def parse_m3u(hass: HomeAssistant, url: str) -> List[PlaylistItem]:
    hls_content_types: Tuple[str, ...] = (
        "application/vnd.apple.mpegurl",
    )
    m3u_data: str = await _fetch_playlist(hass, url, hls_content_types)
    m3u_lines: List[str] = m3u_data.splitlines()

    playlist: List[PlaylistItem] = []
    length: Optional[str] = None
    title: Optional[str] = None

    for line in m3u_lines:
        line = line.strip()
        if line.startswith("#EXTINF:"):
            info = line.split("#EXTINF:")[1].split(",", 1)
            if len(info) != 2:
                _LOGGER.warning("Ignoring invalid extinf %s in playlist %s", line, url)
                continue
            length = info[0].split(" ", 1)[0]
            title = info[1].strip()
        elif line.startswith(("#EXT-X-VERSION:", "#EXT-X-STREAM-INF:")):
            raise PlaylistSupported("HLS")
        elif line.startswith("#"):
            continue
        elif len(line) != 0:
            if not _is_url(line):
                raise PlaylistError(f"Invalid item {line} in playlist {url}")
            playlist.append(PlaylistItem(length=length, title=title, url=line))
            length = None
            title = None

    return playlist


async def parse_pls(hass: HomeAssistant, url: str) -> List[PlaylistItem]:
    pls_data: str = await _fetch_playlist(hass, url, ())
    pls_parser: configparser.ConfigParser = configparser.ConfigParser()
    try:
        pls_parser.read_string(pls_data, url)
    except configparser.Error as err:
        raise PlaylistError(f"Can't parse playlist {url}") from err

    if (
        _PLS_SECTION_PLAYLIST not in pls_parser
        or pls_parser[_PLS_SECTION_PLAYLIST].getint("Version") != 2
    ):
        raise PlaylistError(f"Invalid playlist {url}")

    try:
        num_entries: int = pls_parser.getint(_PLS_SECTION_PLAYLIST, "NumberOfEntries")
    except (configparser.NoOptionError, ValueError) as err:
        raise PlaylistError(f"Invalid NumberOfEntries in playlist {url}") from err

    playlist_section: configparser.SectionProxy = pls_parser[_PLS_SECTION_PLAYLIST]
    playlist: List[PlaylistItem] = []
    for entry in range(1, num_entries + 1):
        file_option: str = f"File{entry}"
        if file_option not in playlist_section:
            _LOGGER.warning("Missing %s in pls from %s", file_option, url)
            continue
        item_url: str = playlist_section[file_option]
        if not _is_url(item_url):
            raise PlaylistError(f"Invalid item {item_url} in playlist {url}")
        playlist.append(
            PlaylistItem(
                length=playlist_section.get(f"Length{entry}"),
                title=playlist_section.get(f"Title{entry}"),
                url=item_url,
            )
        )
    return playlist


async def parse_playlist(hass: HomeAssistant, url: str) -> List[PlaylistItem]:
    if url.endswith((".m3u", ".m3u8")):
        playlist = await parse_m3u(hass, url)
    else:
        playlist = await parse_pls(hass, url)

    if not playlist:
        raise PlaylistError(f"Empty playlist {url}")

    return playlist