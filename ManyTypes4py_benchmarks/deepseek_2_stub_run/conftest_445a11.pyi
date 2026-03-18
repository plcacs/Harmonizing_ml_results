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
    @property
    def address(self) -> list[str]: ...
    @address.setter
    def address(self, value: list[str]) -> None: ...

class SonosMockSubscribe:
    event_listener: SonosMockEventListener
    service: Mock
    callback_future: asyncio.Future[Any] | None
    _callback: Callable[..., Any] | None
    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None: ...
    @property
    def callback(self) -> Callable[..., Any] | None: ...
    @callback.setter
    def callback(self, callback: Callable[..., Any] | None) -> None: ...
    def _get_callback_future(self) -> asyncio.Future[Any]: ...
    async def wait_for_callback_to_be_set(self) -> Callable[..., Any]: ...
    async def unsubscribe(self) -> None: ...

class SonosMockService:
    service_type: str
    subscribe: AsyncMock
    def __init__(self, service_type: str, ip_address: str = ...) -> None: ...

class SonosMockEvent:
    sid: str
    seq: str
    timestamp: float
    service: SonosMockService
    variables: dict[str, Any]
    def __init__(self, soco: SoCo, service: SonosMockService, variables: dict[str, Any]) -> None: ...
    def increment_variable(self, var_name: str) -> str: ...

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo: ...

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Callable[[], Coroutine[Any, Any, None]]) -> None: ...

@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Callable[[], Coroutine[Any, Any, None]]) -> Callable[[], Coroutine[Any, Any, None]]: ...

@pytest.fixture
def config_entry() -> MockConfigEntry: ...

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
    sub_crossover: Any
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

@pytest.fixture
def soco_sharelink() -> Generator[MagicMock, None, None]: ...

@pytest.fixture
def sonos_websocket() -> Generator[AsyncMock, None, None]: ...

@pytest.fixture
def soco_factory(music_library: MagicMock, speaker_info: dict[str, Any], current_track_info_empty: dict[str, str], battery_info: dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_websocket: AsyncMock, sonos_queue: list[DidlMusicTrack]) -> Generator[SoCoMockFactory, None, None]: ...

@pytest.fixture
def soco(soco_factory: SoCoMockFactory) -> MockSoCo: ...

@pytest.fixture
def silent_ssdp_scanner() -> Generator[None, None, None]: ...

@pytest.fixture
def discover(soco: MockSoCo) -> Generator[Mock, None, None]: ...

@pytest.fixture
def config() -> dict[str, Any]: ...

@pytest.fixture
def sonos_favorites() -> SearchResult: ...

@pytest.fixture
def sonos_playlists() -> SearchResult: ...

@pytest.fixture
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

@pytest.fixture
def music_library_browse_categories() -> list[MockMusicServiceItem]: ...

@pytest.fixture
def music_library(sonos_favorites: SearchResult, music_library_browse_categories: list[MockMusicServiceItem]) -> MagicMock: ...

@pytest.fixture
def alarm_clock() -> SonosMockService: ...

@pytest.fixture
def alarm_clock_extended() -> SonosMockService: ...

@pytest.fixture
def speaker_info() -> dict[str, str]: ...

@pytest.fixture
def current_track_info_empty() -> dict[str, str]: ...

@pytest.fixture
def battery_info() -> dict[str, Any]: ...

@pytest.fixture
def device_properties_event(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture
def alarm_event(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture
def no_media_event(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture
def tv_event(soco: MockSoCo) -> SonosMockEvent: ...

@pytest.fixture
def zgs_discovery() -> str: ...

@pytest.fixture
def fire_zgs_event(hass: HomeAssistant, soco: MockSoCo, zgs_discovery: str) -> Callable[[], Coroutine[Any, Any, None]]: ...

@pytest.fixture
async def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> list[MockSoCo]: ...
```