"""Configuration for Sonos tests."""

import asyncio
from collections.abc import Callable, Coroutine, Generator
from copy import copy
from ipaddress import ip_address
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest
from soco import SoCo
from soco.alarms import Alarms
from soco.data_structures import DidlFavorite, DidlMusicTrack, DidlPlaylistContainer, SearchResult
from soco.events_base import Event as SonosEvent
from homeassistant.components import ssdp
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.components.sonos import DOMAIN
from homeassistant.const import CONF_HOSTS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service_info.ssdp import ATTR_UPNP_UDN, SsdpServiceInfo
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo

class SonosMockEventListener:
    """Mock the event listener."""
    def __init__(self, ip_address: str) -> None: ...
    address: List[str]

class SonosMockSubscribe:
    """Mock the subscription."""
    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None: ...
    event_listener: SonosMockEventListener
    service: Mock
    callback_future: Optional[asyncio.Future]
    _callback: Optional[Callable]

    @property
    def callback(self) -> Optional[Callable]: ...
    @callback.setter
    def callback(self, callback: Callable) -> None: ...

    def _get_callback_future(self) -> asyncio.Future: ...
    async def wait_for_callback_to_be_set(self) -> None: ...
    async def unsubscribe(self) -> None: ...

class SonosMockService:
    """Mock a Sonos Service used in callbacks."""
    def __init__(self, service_type: str, ip_address: str = '192.168.42.2') -> None: ...
    subscribe: AsyncMock

class SonosMockEvent:
    """Mock a sonos Event used in callbacks."""
    def __init__(self, soco: SoCo, service: SonosMockService, variables: Dict[str, Any]) -> None: ...
    sid: str
    seq: str
    timestamp: float
    service: SonosMockService
    variables: Dict[str, Any]

    def increment_variable(self, var_name: str) -> str: ...

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo: ...

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Callable) -> None: ...

@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Callable) -> Callable: ...

@pytest.fixture
def config_entry() -> MockConfigEntry: ...

class MockSoCo(MagicMock):
    """Mock the Soco Object."""
    uid: str
    play_mode: str
    mute: bool
    night_mode: bool
    dialog_level: bool
    loudness: bool
    volume: int
    audio_delay: int
    balance: Tuple[int, int]
    bass: int
    treble: int
    mic_enabled: bool
    sub_crossover: Optional[str]
    sub_enabled: bool
    sub_gain: int
    surround_enabled: bool
    surround_mode: bool
    surround_level: int
    music_surround_level: int
    soundbar_audio_input_format: str

    @property
    def visible_zones(self) -> Set[SoCo]: ...

class SoCoMockFactory:
    """Factory for creating SoCo Mocks."""
    def __init__(self, music_library: MagicMock, speaker_info: Dict[str, Any], current_track_info_empty: Dict[str, Any], battery_info: Dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_queue: List[DidlMusicTrack]) -> None: ...
    mock_list: Dict[str, MockSoCo]
    music_library: MagicMock
    speaker_info: Dict[str, Any]
    current_track_info: Dict[str, Any]
    battery_info: Dict[str, Any]
    alarm_clock: SonosMockService
    sonos_playlists: SearchResult
    sonos_queue: List[DidlMusicTrack]

    def cache_mock(self, mock_soco: MockSoCo, ip_address: str, name: str = 'Zone A') -> MockSoCo: ...
    def get_mock(self, *args: Any) -> MockSoCo: ...

def patch_gethostbyname(host: str) -> str: ...

@pytest.fixture
def soco_sharelink() -> MagicMock: ...

@pytest.fixture
def sonos_websocket() -> AsyncMock: ...

@pytest.fixture
def soco_factory(music_library: MagicMock, speaker_info: Dict[str, Any], current_track_info_empty: Dict[str, Any], battery_info: Dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_websocket: AsyncMock, sonos_queue: List[DidlMusicTrack]) -> SoCoMockFactory: ...

@pytest.fixture
def soco(soco_factory: SoCoMockFactory) -> MockSoCo: ...

@pytest.fixture
def discover() -> MagicMock: ...

@pytest.fixture
def config() -> Dict[str, Dict[str, Dict[str, List[str]]]]: ...

@pytest.fixture
def sonos_favorites() -> SearchResult: ...

@pytest.fixture
def sonos_playlists() -> SearchResult: ...

@pytest.fixture
def sonos_queue() -> List[DidlMusicTrack]: ...

class MockMusicServiceItem:
    """Mocks a Soco MusicServiceItem."""
    def __init__(self, title: str, item_id: str, parent_id: str, item_class: str, album_art_uri: Optional[str] = None) -> None: ...
    title: str
    item_id: str
    item_class: str
    parent_id: str
    album_art_uri: Optional[str]

def list_from_json_fixture(file_name: str) -> List[MockMusicServiceItem]: ...

def mock_browse_by_idstring(search_type: str, idstring: str, start: int = 0, max_items: int = 100, full_album_art_uri: bool = False) -> List[Union[MockMusicServiceItem, Any]]: ...

def mock_get_music_library_information(search_type: str, search_term: str, full_album_art_uri: bool = True) -> List[MockMusicServiceItem]: ...

@pytest.fixture
def music_library_browse_categories() -> List[MockMusicServiceItem]: ...

@pytest.fixture
def music_library(sonos_favorites: SearchResult, music_library_browse_categories: List[MockMusicServiceItem]) -> MagicMock: ...

@pytest.fixture
def alarm_clock() -> SonosMockService: ...

@pytest.fixture
def alarm_clock_extended() -> SonosMockService: ...

@pytest.fixture
def speaker_info() -> Dict[str, Any]: ...

@pytest.fixture
def current_track_info_empty() -> Dict[str, Any]: ...

@pytest.fixture
def battery_info() -> Dict[str, Any]: ...

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
def fire_zgs_event(hass: HomeAssistant, soco: MockSoCo, zgs_discovery: str) -> Callable: ...

@pytest.fixture
async def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> List[MockSoCo]: ...