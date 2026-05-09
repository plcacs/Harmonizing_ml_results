from __future__ import annotations
import configparser
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, ClassVar
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
_LOGGER: ClassVar[logging.Logger] = logging.getLogger(__name__)
_PLS_SECTION_PLAYLIST: ClassVar[str] = 'playlist'

@attr.s(slots=True, frozen=True)
class ChromecastInfo:
    """Class to hold all data about a chromecast for creating connections.

    This also has the same attributes as the mDNS fields by zeroconf.
    """
    cast_info: CastInfo
    is_dynamic_group: bool | None = None

    @property
    def friendly_name(self) -> str:
        """Return the Friendly Name."""
        return self.cast_info.friendly_name

    @property
    def is_audio_group(self) -> bool:
        """Return if the cast is an audio group."""
        return self.cast_info.cast_type == CAST_TYPE_GROUP

    @property
    def uuid(self) -> str:
        """Return the UUID."""
        return self.cast_info.uuid

    def fill_out_missing_chromecast_info(self, hass: HomeAssistant) -> ChromecastInfo:
        """Return a new ChromecastInfo object with missing attributes filled in.

        Uses blocking HTTP / HTTPS.
        """
        # ... (rest of the method remains the same)

class ChromeCastZeroconf:
    """Class to hold a zeroconf instance."""
    __zconf: ClassVar[zeroconf.Zeroconf] | None = None

    @classmethod
    def set_zeroconf(cls, zconf: zeroconf.Zeroconf) -> None:
        """Set zeroconf."""
        cls.__zconf = zconf

    @classmethod
    def get_zeroconf(cls) -> zeroconf.Zeroconf | None:
        """Get zeroconf."""
        return cls.__zconf

class CastStatusListener(pychromecast.controllers.media.MediaStatusListener, pychromecast.controllers.multizone.MultiZoneManagerListener, pychromecast.controllers.receiver.CastStatusListener, pychromecast.socket_client.ConnectionStatusListener):
    """Helper class to handle pychromecast status callbacks.

    Necessary because a CastDevice entity or dynamic group can create a new
    socket client and therefore callbacks from multiple chromecast connections can
    potentially arrive. This class allows invalidating past chromecast objects.
    """
    def __init__(self, cast_device: pychromecast.CastDevice, chromecast: pychromecast.Chromecast, mz_mgr: pychromecast.controllers.multizone.MultiZoneManager, mz_only: bool = False) -> None:
        """Initialize the status listener."""
        self._cast_device: pychromecast.CastDevice
        self._uuid: str
        self._valid: bool
        self._mz_mgr: pychromecast.controllers.multizone.MultiZoneManager
        if cast_device._cast_info.is_audio_group:
            self._mz_mgr.add_multizone(chromecast)
        if mz_only:
            return
        chromecast.register_status_listener(self)
        chromecast.socket_client.media_controller.register_status_listener(self)
        chromecast.register_connection_listener(self)
        if not cast_device._cast_info.is_audio_group:
            self._mz_mgr.register_listener(chromecast.uuid, self)

    # ... (rest of the class remains the same)

class PlaylistError(Exception):
    """Exception wrapper for pls and m3u helpers."""

class PlaylistSupported(PlaylistError):
    """The playlist is supported by cast devices and should not be parsed."""

@dataclass
class PlaylistItem:
    """Playlist item."""

def _is_url(url: str) -> bool:
    """Validate the URL can be parsed and at least has scheme + netloc."""
    result = urlparse(url)
    return all([result.scheme, result.netloc])

async def _fetch_playlist(hass: HomeAssistant, url: str, supported_content_types: tuple[str, ...]) -> str:
    """Fetch a playlist from the given url."""
    try:
        session = aiohttp_client.async_get_clientsession(hass, verify_ssl=False)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            charset = resp.charset or 'utf-8'
            if resp.content_type in supported_content_types:
                raise PlaylistSupported
            try:
                playlist_data = (await resp.content.read(64 * 1024)).decode(charset)
            except ValueError as err:
                raise PlaylistError(f'Could not decode playlist {url}') from err
    except TimeoutError as err:
        raise PlaylistError(f'Timeout while fetching playlist {url}') from err
    except aiohttp.client_exceptions.ClientError as err:
        raise PlaylistError(f'Error while fetching playlist {url}') from err
    return playlist_data

async def parse_m3u(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    """Very simple m3u parser.

    Based on https://github.com/dvndrsn/M3uParser/blob/master/m3uparser.py
    """
    hls_content_types: tuple[str, ...] = ('application/vnd.apple.mpegurl',)
    m3u_data = await _fetch_playlist(hass, url, hls_content_types)
    m3u_lines = m3u_data.splitlines()
    playlist: list[PlaylistItem] = []
    length: tuple[str, str] | None = None
    title: str | None = None
    for line in m3u_lines:
        line = line.strip()
        if line.startswith('#EXTINF:'):
            info = line.split('#EXTINF:')[1].split(',', 1)
            if len(info) != 2:
                _LOGGER.warning('Ignoring invalid extinf %s in playlist %s', line, url)
                continue
            length = info[0].split(' ', 1)
            title = info[1].strip()
        elif line.startswith(('#EXT-X-VERSION:', '#EXT-X-STREAM-INF:')):
            raise PlaylistSupported('HLS')
        elif line.startswith('#'):
            continue
        elif len(line) != 0:
            if not _is_url(line):
                raise PlaylistError(f'Invalid item {line} in playlist {url}')
            playlist.append(PlaylistItem(length=length, title=title, url=line))
            length = None
            title = None
    return playlist

async def parse_pls(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    """Very simple pls parser.

    Based on https://github.com/mariob/plsparser/blob/master/src/plsparser.py
    """
    pls_data = await _fetch_playlist(hass, url, ())
    pls_parser = configparser.ConfigParser()
    try:
        pls_parser.read_string(pls_data, url)
    except configparser.Error as err:
        raise PlaylistError(f"Can't parse playlist {url}") from err
    if _PLS_SECTION_PLAYLIST not in pls_parser or pls_parser[_PLS_SECTION_PLAYLIST].getint('Version') != 2:
        raise PlaylistError(f'Invalid playlist {url}')
    try:
        num_entries = pls_parser.getint(_PLS_SECTION_PLAYLIST, 'NumberOfEntries')
    except (configparser.NoOptionError, ValueError) as err:
        raise PlaylistError(f'Invalid NumberOfEntries in playlist {url}') from err
    playlist_section = pls_parser[_PLS_SECTION_PLAYLIST]
    playlist: list[PlaylistItem] = []
    for entry in range(1, num_entries + 1):
        file_option = f'File{entry}'
        if file_option not in playlist_section:
            _LOGGER.warning('Missing %s in pls from %s', file_option, url)
            continue
        item_url = playlist_section[file_option]
        if not _is_url(item_url):
            raise PlaylistError(f'Invalid item {item_url} in playlist {url}')
        playlist.append(PlaylistItem(length=playlist_section.get(f'Length{entry}'), title=playlist_section.get(f'Title{entry}'), url=item_url))
    return playlist

async def parse_playlist(hass: HomeAssistant, url: str) -> list[PlaylistItem]:
    """Parse an m3u or pls playlist."""
    if url.endswith(('.m3u', '.m3u8')):
        playlist = await parse_m3u(hass, url)
    else:
        playlist = await parse_pls(hass, url)
    if not playlist:
        raise PlaylistError(f'Empty playlist {url}')
    return playlist
