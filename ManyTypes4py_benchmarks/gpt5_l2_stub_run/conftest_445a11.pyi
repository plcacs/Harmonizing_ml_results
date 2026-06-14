from collections.abc import Callable, Coroutine
import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock
from soco import SoCo
from soco.alarms import Alarms
from soco.data_structures import DidlMusicTrack, SearchResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo
from tests.common import MockConfigEntry


class SonosMockEventListener:
    def __init__(self, ip_address: str) -> None: ...
    address: list[str]


class SonosMockSubscribe:
    event_listener: SonosMockEventListener
    service: Mock
    callback_future: Optional[asyncio.Future[Callable[[Any], Any]]]
    _callback: Optional[Callable[[Any], Any]]

    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None: ...
    @property
    def callback(self) -> Optional[Callable[[Any], Any]]: ...
    @callback.setter
    def callback(self, callback: Callable[[Any], Any]) -> None: ...
    def _get_callback_future(self) -> asyncio.Future[Callable[[Any], Any]]: ...
    def wait_for_callback_to_be_set(self) -> Coroutine[Any, Any, Callable[[Any], Any]]: ...
    def unsubscribe(self) -> Coroutine[Any, Any, None]: ...


class SonosMockService:
    service_type: str
    subscribe: AsyncMock

    def __init__(self, service_type: str, ip_address: str = ...) -> None: ...


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
    sub_crossover: Optional[int]
    sub_enabled: bool
    sub_gain: int
    surround_enabled: bool
    surround_mode: bool
    surround_level: int
    music_surround_level: int
    soundbar_audio_input_format: str
    ip_address: str
    music_library: MagicMock
    avTransport: SonosMockService
    renderingControl: SonosMockService
    zoneGroupTopology: SonosMockService
    contentDirectory: SonosMockService
    deviceProperties: SonosMockService
    alarmClock: SonosMockService
    all_zones: set["MockSoCo"]
    group: Any

    @property
    def visible_zones(self) -> set["MockSoCo"]: ...


class SonosMockEvent:
    sid: str
    seq: str
    timestamp: float
    service: SonosMockService
    variables: dict[str, Any]

    def __init__(self, soco: MockSoCo, service: SonosMockService, variables: dict[str, Any]) -> None: ...
    def increment_variable(self, var_name: str) -> str: ...


class SoCoMockFactory:
    mock_list: dict[str, MockSoCo]
    music_library: MagicMock
    speaker_info: dict[str, str]
    current_track_info: dict[str, str]
    battery_info: dict[str, int | str]
    alarm_clock: SonosMockService
    sonos_playlists: SearchResult
    sonos_queue: list[DidlMusicTrack]

    def __init__(
        self,
        music_library: MagicMock,
        speaker_info: dict[str, str],
        current_track_info_empty: dict[str, str],
        battery_info: dict[str, int | str],
        alarm_clock: SonosMockService,
        sonos_playlists: SearchResult,
        sonos_queue: list[DidlMusicTrack],
    ) -> None: ...
    def cache_mock(self, mock_soco: MockSoCo, ip_address: str, name: str = ...) -> MockSoCo: ...
    def get_mock(self, *args: Any) -> MockSoCo: ...


def patch_gethostbyname(host: str) -> str: ...


def zeroconf_payload() -> ZeroconfServiceInfo: ...
def async_autosetup_sonos(async_setup_sonos: Callable[[], Coroutine[Any, Any, None]]) -> Coroutine[Any, Any, None]: ...
def async_setup_sonos(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    fire_zgs_event: Callable[[], Coroutine[Any, Any, None]],
) -> Callable[[], Coroutine[Any, Any, None]]: ...
def config_entry_fixture() -> MockConfigEntry: ...
def soco_sharelink() -> MagicMock: ...
def sonos_websocket() -> AsyncMock: ...
def soco_factory(
    music_library: MagicMock,
    speaker_info: dict[str, str],
    current_track_info_empty: dict[str, str],
    battery_info: dict[str, int | str],
    alarm_clock: SonosMockService,
    sonos_playlists: SearchResult,
    sonos_websocket: AsyncMock,
    sonos_queue: list[DidlMusicTrack],
) -> SoCoMockFactory: ...
def soco_fixture(soco_factory: SoCoMockFactory) -> MockSoCo: ...
def silent_ssdp_scanner() -> None: ...
def discover_fixture(soco: MockSoCo) -> MagicMock: ...
def config_fixture() -> dict[str, Any]: ...
def sonos_favorites_fixture() -> SearchResult: ...
def sonos_playlists_fixture() -> SearchResult: ...
def sonos_queue() -> list[DidlMusicTrack]: ...


class MockMusicServiceItem:
    title: str
    item_id: str
    parent_id: str
    item_class: str
    album_art_uri: Optional[str]

    def __init__(
        self,
        title: str,
        item_id: str,
        parent_id: str,
        item_class: str,
        album_art_uri: Optional[str] = ...,
    ) -> None: ...


def list_from_json_fixture(file_name: str) -> list[MockMusicServiceItem]: ...
def mock_browse_by_idstring(
    search_type: str,
    idstring: str,
    start: int = ...,
    max_items: int = ...,
    full_album_art_uri: bool = ...,
) -> list[MockMusicServiceItem]: ...
def mock_get_music_library_information(
    search_type: str, search_term: str, full_album_art_uri: bool = ...
) -> list[MockMusicServiceItem]: ...
def music_library_browse_categories() -> list[MockMusicServiceItem]: ...
def music_library_fixture(
    sonos_favorites: SearchResult, music_library_browse_categories: list[MockMusicServiceItem]
) -> MagicMock: ...
def alarm_clock_fixture() -> SonosMockService: ...
def alarm_clock_fixture_extended() -> SonosMockService: ...
def speaker_info_fixture() -> dict[str, str]: ...
def current_track_info_empty_fixture() -> dict[str, str]: ...
def battery_info_fixture() -> dict[str, int | str]: ...
def device_properties_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...
def alarm_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...
def no_media_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...
def tv_event_fixture(soco: MockSoCo) -> SonosMockEvent: ...
def zgs_discovery_fixture() -> str: ...
def zgs_event_fixture(
    hass: HomeAssistant, soco: MockSoCo, zgs_discovery: str
) -> Callable[[], Coroutine[Any, Any, None]]: ...
def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> Coroutine[Any, Any, list[MockSoCo]]: ...