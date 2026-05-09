"""Configuration for Sonos tests."""

import asyncio
from collections.abc import Callable, Coroutine, Generator
from copy import copy
from ipaddress import ip_address
from typing import Any, Dict, List, Optional, Union
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    PropertyMock,
    call,
    patch,
)
import pytest
from soco import SoCo
from soco.alarms import Alarms
from soco.data_structures import (
    DidlFavorite,
    DidlMusicTrack,
    DidlPlaylistContainer,
    SearchResult,
)
from soco.events_base import Event as SonosEvent
from homeassistant.components import ssdp
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.components.sonos import DOMAIN
from homeassistant.const import CONF_HOSTS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service_info.ssdp import (
    ATTR_UPNP_UDN,
    SsdpServiceInfo,
)
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry

class SonosMockEventListener:
    """Mock the event listener."""
    def __init__(self, ip_address: str) -> None:
        ...
    address: List[str]

class SonosMockSubscribe:
    """Mock the subscription."""
    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None:
        ...
    event_listener: SonosMockEventListener
    service: MagicMock
    callback_future: Optional[asyncio.Future]
    _callback: Optional[Callable]

    @property
    def callback(self) -> Optional[Callable]:
        ...
    @callback.setter
    def callback(self, callback: Callable) -> None:
        ...

    async def wait_for_callback_to_be_set(self) -> None:
        ...
    async def unsubscribe(self) -> None:
        ...

class SonosMockService:
    """Mock a Sonos Service used in callbacks."""
    def __init__(self, service_type: str, ip_address: str = '192.168.42.2') -> None:
        ...
    subscribe: AsyncMock

class SonosMockEvent:
    """Mock a sonos Event used in callbacks."""
    def __init__(self, soco: SoCo, service: SonosMockService, variables: Dict[str, Any]) -> None:
        ...
    sid: str
    seq: str
    timestamp: float
    service: SonosMockService
    variables: Dict[str, Any]

    def increment_variable(self, var_name: str) -> str:
        ...

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo:
    ...

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Callable) -> None:
    ...

@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Callable) -> Callable:
    ...

@pytest.fixture(name='config_entry')
def config_entry_fixture() -> MockConfigEntry:
    ...

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
    balance: tuple[int, int]
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
    def visible_zones(self) -> set[SoCo]:
        ...

class SoCoMockFactory:
    """Factory for creating SoCo Mocks."""
    def __init__(self, music_library: MagicMock, speaker_info: Dict[str, Any], current_track_info_empty: Dict[str, Any], battery_info: Dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_queue: List[DidlMusicTrack]) -> None:
        ...
    mock_list: Dict[str, MockSoCo]
    music_library: MagicMock
    speaker_info: Dict[str, Any]
    current_track_info: Dict[str, Any]
    battery_info: Dict[str, Any]
    alarm_clock: SonosMockService
    sonos_playlists: SearchResult
    sonos_queue: List[DidlMusicTrack]

    def cache_mock(self, mock_soco: MockSoCo, ip_address: str, name: str = 'Zone A') -> MockSoCo:
        ...
    def get_mock(self, *args: Any) -> MockSoCo:
        ...

def patch_gethostbyname(host: str) -> str:
    ...

@pytest.fixture(name='soco_sharelink')
def soco_sharelink() -> MagicMock:
    ...

@pytest.fixture(name='sonos_websocket')
def sonos_websocket() -> AsyncMock:
    ...

@pytest.fixture(name='soco_factory')
def soco_factory(music_library: MagicMock, speaker_info: Dict[str, Any], current_track_info_empty: Dict[str, Any], battery_info: Dict[str, Any], alarm_clock: SonosMockService, sonos_playlists: SearchResult, sonos_queue: List[DidlMusicTrack]) -> SoCoMockFactory:
    ...

@pytest.fixture(name='soco')
def soco_fixture(soco_factory: SoCoMockFactory) -> MockSoCo:
    ...

@pytest.fixture(autouse=True)
def silent_ssdp_scanner() -> None:
    ...

@pytest.fixture(name='discover', autouse=True)
def discover_fixture(soco: MockSoCo) -> patch:
    ...

@pytest.fixture(name='config')
def config_fixture() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    ...

@pytest.fixture(name='sonos_favorites')
def sonos_favorites_fixture() -> SearchResult:
    ...

@pytest.fixture(name='sonos_playlists')
def sonos_playlists_fixture() -> SearchResult:
    ...

@pytest.fixture(name='sonos_queue')
def sonos_queue() -> List[DidlMusicTrack]:
    ...

class MockMusicServiceItem:
    """Mocks a Soco MusicServiceItem."""
    def __init__(self, title: str, item_id: str, parent_id: str, item_class: str, album_art_uri: Optional[str] = None) -> None:
        ...
    title: str
    item_id: str
    item_class: str
    parent_id: str
    album_art_uri: Optional[str]

def list_from_json_fixture(file_name: str) -> List[MockMusicServiceItem]:
    ...

def mock_browse_by_idstring(search_type: str, idstring: str, start: int = 0, max_items: int = 100, full_album_art_uri: bool = False) -> Union[List[MockMusicServiceItem], List]:
    ...

def mock_get_music_library_information(search_type: str, search_term: str, full_album_art_uri: bool = True) -> List[MockMusicServiceItem]:
    ...

@pytest.fixture(name='music_library_browse_categories')
def music_library_browse_categories() -> List[MockMusicServiceItem]:
    ...

@pytest.fixture(name='music_library')
def music_library_fixture(sonos_favorites: SearchResult, music_library_browse_categories: List[MockMusicServiceItem]) -> MagicMock:
    ...

@pytest.fixture(name='alarm_clock')
def alarm_clock_fixture() -> SonosMockService:
    ...

@pytest.fixture(name='alarm_clock_extended')
def alarm_clock_fixture_extended() -> SonosMockService:
    ...

@pytest.fixture(name='speaker_info')
def speaker_info_fixture() -> Dict[str, Any]:
    ...

@pytest.fixture(name='current_track_info_empty')
def current_track_info_empty_fixture() -> Dict[str, Any]:
    ...

@pytest.fixture(name='battery_info')
def battery_info_fixture() -> Dict[str, Any]:
    ...

@pytest.fixture(name='device_properties_event')
def device_properties_event_fixture(soco: MockSoCo) -> SonosMockEvent:
    ...

@pytest.fixture(name='alarm_event')
def alarm_event_fixture(soco: MockSoCo) -> SonosMockEvent:
    ...

@pytest.fixture(name='no_media_event')
def no_media_event_fixture(soco: MockSoCo) -> SonosMockEvent:
    ...

@pytest.fixture(name='tv_event')
def tv_event_fixture(soco: MockSoCo) -> SonosMockEvent:
    ...

@pytest.fixture(name='zgs_discovery', scope='package')
def zgs_discovery_fixture() -> str:
    ...

@pytest.fixture(name='fire_zgs_event')
def zgs_event_fixture(hass: HomeAssistant, soco: MockSoCo, zgs_discovery: str) -> Callable:
    ...

@pytest.fixture(name='sonos_setup_two_speakers')
async def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> List[MockSoCo]:
    ...