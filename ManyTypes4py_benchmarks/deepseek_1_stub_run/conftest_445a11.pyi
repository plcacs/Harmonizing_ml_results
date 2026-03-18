```python
import asyncio
from collections.abc import Callable, Coroutine, Generator
from ipaddress import IPv4Address, IPv6Address
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest
from soco import SoCo
from soco.alarms import Alarms
from soco.data_structures import DidlFavorite, DidlMusicTrack, DidlPlaylistContainer, SearchResult
from soco.events_base import Event as SonosEvent
from homeassistant.components.service_info.ssdp import SsdpServiceInfo
from homeassistant.components.service_info.zeroconf import ZeroconfServiceInfo
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

class SonosMockEventListener:
    address: list[str]
    def __init__(self, ip_address: str) -> None: ...
    def __init__(self, ip_address: str) -> None: ...

class SonosMockSubscribe:
    event_listener: SonosMockEventListener
    service: Mock
    callback_future: asyncio.Future[Any] | None
    _callback: Any | None
    @property
    def callback(self) -> Any: ...
    @callback.setter
    def callback(self, callback: Any) -> None: ...
    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None: ...
    def _get_callback_future(self) -> asyncio.Future[Any]: ...
    async def wait_for_callback_to_be_set(self) -> Any: ...
    async def unsubscribe(self) -> None: ...

class SonosMockService:
    service_type: str
    subscribe: AsyncMock
    def __init__(self, service_type: str, ip_address: str = ...) -> None: ...

class SonosMockEvent:
    sid: str
    seq: str
    timestamp: float
    service: Any
    variables: dict[str, Any]
    def __init__(self, soco: Any, service: Any, variables: dict[str, Any]) -> None: ...
    def increment_variable(self, var_name: str) -> str: ...

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo: ...

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Any) -> None: ...

@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Any) -> Callable[[], Coroutine[Any, Any, None]]: ...

@pytest.fixture(name="config_entry")
def config_entry_fixture() -> MockConfigEntry: ...

class MockSoCo(MagicMock):
    uid: str
    play_mode: str
    mute: bool
    night_mode: bool
    dialog_level: bool
    loudness: bool
    volume: int
    audio_delay: int
    balance: tuple[int, int]
    bass: int
    treble: int
    mic_enabled: bool
    sub_crossover: Any | None
    sub_enabled: bool
    sub_gain: int
    surround_enabled: bool
    surround_mode: bool
    surround_level: int
    music_surround_level: int
    soundbar_audio_input_format: str
    @property
    def visible_zones(self) -> set[Any]: ...

class SoCoMockFactory:
    mock_list: dict[str, MockSoCo]
    music_library: MagicMock
    speaker_info: dict[str, Any]
    current_track_info: dict[str, str]
    battery_info: dict[str, Any]
    alarm_clock: SonosMockService
    sonos_playlists: SearchResult
    sonos_queue: list[DidlMusicTrack]
    def __init__(self, music_library: MagicMock, speaker_info: dict[str, Any], current_track_info_empty: dict[str, str], battery_info: dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_queue: list[DidlMusicTrack]) -> None: ...
    def cache_mock(self, mock_soco: MockSoCo, ip_address: str, name: str = ...) -> MockSoCo: ...
    def get_mock(self, *args: Any) -> MockSoCo: ...

def patch_gethostbyname(host: str) -> str: ...

@pytest.fixture(name="soco_sharelink")
def soco_sharelink() -> Generator[MagicMock, None, None]: ...

@pytest.fixture(name="sonos_websocket")
def sonos_websocket() -> Generator[AsyncMock, None, None]: ...

@pytest.fixture(name="soco_factory")
def soco_factory(music_library: MagicMock, speaker_info: dict[str, Any], current_track_info_empty: dict[str, str], battery_info: dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_websocket: AsyncMock, sonos_queue: list[DidlMusicTrack]) -> Generator[SoCoMockFactory, None, None]: ...

@pytest.fixture(name="soco")
def soco_fixture(soco_factory: SoCoMockFactory) -> MockSoCo: ...

@pytest.fixture(autouse=True)
def silent_ssdp_scanner() -> Generator[None, None, None]: ...

@pytest.fixture(name="discover", autouse=True)
def discover_fixture(soco: MockSoCo) -> Generator[Mock, None, None]: ...

@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]: ...

@pytest.fixture(name="sonos_favorites")
def sonos_favorites_fixture() -> SearchResult: ...

@pytest.fixture(name="sonos_playlists")
def sonos_playlists_fixture() -> SearchResult: ...

@pytest.fixture(name="sonos_queue")
def sonos_queue() -> list[DidlMusicTrack]: ...

class MockMusicServiceItem:
    title: str
    item_id: str
    parent_id: str
    item_class: str
    album_art_uri: str | None
    def __init__(self, title: str, item_id: str, parent_id: str, item_class: str, album_art_uri: str | None = ...) -> None: ...

def list_from_json_fixture(file_name: str) -> list[MockMusicServiceItem]: ...

def mock_browse_by_idstring(search_type: str, idstring: str, start: int = ..., max_items: int = ..., full_album_art_uri: bool = ...) -> list[MockMusicServiceItem]: ...

def mock_get_music_library_information(search_type: str, search_term: str, full_album_art_uri: bool = ...) -> list[MockMusicServiceItem]: ...

@pytest.fixture(name="music_library_browse_categories")
def music_library_browse_categories() -> list[MockMusicServiceItem]: ...

@pytest.fixture(name="music_library")
def music_library_fixture(sonos_favorites: SearchResult, music_library_browse_categories: list[MockMusicServiceItem]) -> MagicMock: ...

@pytest.fixture(name="alarm_clock")
def alarm_clock_fixture() -> SonosMockService: ...

@pytest.fixture(name="alarm_clock_extended")
def alarm_clock_fixture_extended() -> SonosMockService: ...

@pytest.fixture(name="speaker_info")
def speaker_info_fixture() -> dict[str, Any]: ...

@pytest.fixture(name="current_track_info_empty")
def current_track_info_empty_fixture() -> dict[str, str]: ...

@pytest.fixture(name="battery_info")
def battery_info_fixture() -> dict[str, Any]: ...

@pytest.fixture(name="device_properties_event")
def device_properties_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture(name="alarm_event")
def alarm_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture(name="no_media_event")
def no_media_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture(name="tv_event")
def tv_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture(name="zgs_discovery", scope="package")
def zgs_discovery_fixture() -> str: ...

@pytest.fixture(name="fire_zgs_event")
def zgs_event_fixture(hass: HomeAssistant, soco: MockSoCo, zgs_discovery: str) -> Callable[[], Coroutine[Any, Any, None]]: ...

@pytest.fixture(name="sonos_setup_two_speakers")
async def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> list[MockSoCo]: ...
```