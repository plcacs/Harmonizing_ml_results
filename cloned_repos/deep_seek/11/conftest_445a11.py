"""Configuration for Sonos tests."""
import asyncio
from collections.abc import Callable, Coroutine, Generator, Iterable
from copy import copy
from ipaddress import IPv4Address
from typing import Any, Optional, Union, Dict, List, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch
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
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, load_fixture, load_json_value_fixture

class SonosMockEventListener:
    """Mock the event listener."""

    def __init__(self, ip_address: str) -> None:
        """Initialize the mock event listener."""
        self.address: List[Union[str, int]] = [ip_address, '8080']

class SonosMockSubscribe:
    """Mock the subscription."""

    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the mock subscriber."""
        self.event_listener: SonosMockEventListener = SonosMockEventListener(ip_address)
        self.service: Mock = Mock()
        self.callback_future: Optional[asyncio.Future] = None
        self._callback: Optional[Callable] = None

    @property
    def callback(self) -> Optional[Callable]:
        """Return the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable) -> None:
        """Set the callback."""
        self._callback = callback
        future = self._get_callback_future()
        if not future.done():
            future.set_result(callback)

    def _get_callback_future(self) -> asyncio.Future:
        """Get the callback future."""
        if not self.callback_future:
            self.callback_future = asyncio.get_running_loop().create_future()
        return self.callback_future

    async def wait_for_callback_to_be_set(self) -> Callable:
        """Wait for the callback to be set."""
        return await self._get_callback_future()

    async def unsubscribe(self) -> None:
        """Unsubscribe mock."""

class SonosMockService:
    """Mock a Sonos Service used in callbacks."""

    def __init__(self, service_type: str, ip_address: str = '192.168.42.2') -> None:
        """Initialize the instance."""
        self.service_type: str = service_type
        self.subscribe: AsyncMock = AsyncMock(return_value=SonosMockSubscribe(ip_address))

class SonosMockEvent:
    """Mock a sonos Event used in callbacks."""

    def __init__(self, soco: 'MockSoCo', service: SonosMockService, variables: Dict[str, Any]) -> None:
        """Initialize the instance."""
        self.sid: str = f'{soco.uid}_sub0000000001'
        self.seq: str = '0'
        self.timestamp: float = 1621000000.0
        self.service: SonosMockService = service
        self.variables: Dict[str, Any] = variables

    def increment_variable(self, var_name: str) -> str:
        """Increment the value of the var_name key in variables dict attribute.

        Assumes value has a format of <str>:<int>.
        """
        self.variables = copy(self.variables)
        base, count = self.variables[var_name].split(':')
        newcount = int(count) + 1
        self.variables[var_name] = ':'.join([base, str(newcount)])
        return self.variables[var_name]

@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo:
    """Return a default zeroconf payload."""
    return ZeroconfServiceInfo(
        ip_address=ip_address('192.168.4.2'),
        ip_addresses=[ip_address('192.168.4.2')],
        hostname='Sonos-aaa',
        name='Sonos-aaa@Living Room._sonos._tcp.local.',
        port=None,
        properties={'bootseq': '1234'},
        type='mock_type'
    )

@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Set up a Sonos integration instance on test run."""
    await async_setup_sonos()

@pytest.fixture
def async_setup_sonos(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    fire_zgs_event: Callable[[], Coroutine[Any, Any, None]]
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Return a coroutine to set up a Sonos integration instance on demand."""

    async def _wrapper() -> None:
        config_entry.add_to_hass(hass)
        sonos_alarms: Alarms = Alarms()
        sonos_alarms.last_alarm_list_version = 'RINCON_test:0'
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
        await fire_zgs_event()
        await hass.async_block_till_done(wait_background_tasks=True)
    return _wrapper

@pytest.fixture(name='config_entry')
def config_entry_fixture() -> MockConfigEntry:
    """Create a mock Sonos config entry."""
    return MockConfigEntry(domain=DOMAIN, title='Sonos')

class MockSoCo(MagicMock):
    """Mock the Soco Object."""
    uid: str = 'RINCON_test'
    play_mode: str = 'NORMAL'
    mute: bool = False
    night_mode: bool = True
    dialog_level: bool = True
    loudness: bool = True
    volume: int = 19
    audio_delay: int = 2
    balance: Tuple[int, int] = (61, 100)
    bass: int = 1
    treble: int = -1
    mic_enabled: bool = False
    sub_crossover: Optional[int] = None
    sub_enabled: bool = False
    sub_gain: int = 5
    surround_enabled: bool = True
    surround_mode: bool = True
    surround_level: int = 3
    music_surround_level: int = 4
    soundbar_audio_input_format: str = 'Dolby 5.1'

    @property
    def visible_zones(self) -> set['MockSoCo']:
        """Return visible zones and allow property to be overridden by device classes."""
        return {self}

class SoCoMockFactory:
    """Factory for creating SoCo Mocks."""

    def __init__(
        self,
        music_library: MagicMock,
        speaker_info: Dict[str, Any],
        current_track_info_empty: Dict[str, Any],
        battery_info: Dict[str, Any],
        alarm_clock: SonosMockService,
        sonos_playlists: SearchResult,
        sonos_queue: List[DidlMusicTrack]
    ) -> None:
        """Initialize the mock factory."""
        self.mock_list: Dict[str, MockSoCo] = {}
        self.music_library: MagicMock = music_library
        self.speaker_info: Dict[str, Any] = speaker_info
        self.current_track_info: Dict[str, Any] = current_track_info_empty
        self.battery_info: Dict[str, Any] = battery_info
        self.alarm_clock: SonosMockService = alarm_clock
        self.sonos_playlists: SearchResult = sonos_playlists
        self.sonos_queue: List[DidlMusicTrack] = sonos_queue

    def cache_mock(self, mock_soco: MockSoCo, ip_address: str, name: str = 'Zone A') -> MockSoCo:
        """Put a user created mock into the cache."""
        mock_soco.mock_add_spec(SoCo)
        mock_soco.ip_address: str = ip_address
        if ip_address != '192.168.42.2':
            mock_soco.uid += f'_{ip_address}'
        mock_soco.music_library: MagicMock = self.music_library
        mock_soco.get_current_track_info.return_value = self.current_track_info
        mock_soco.music_source_from_uri = SoCo.music_source_from_uri
        mock_soco.get_sonos_playlists.return_value = self.sonos_playlists
        mock_soco.get_queue.return_value = self.sonos_queue
        my_speaker_info: Dict[str, Any] = self.speaker_info.copy()
        my_speaker_info['zone_name'] = name
        my_speaker_info['uid'] = mock_soco.uid
        mock_soco.get_speaker_info = Mock(return_value=my_speaker_info)
        mock_soco.add_to_queue = Mock(return_value=10)
        mock_soco.add_uri_to_queue = Mock(return_value=10)
        mock_soco.avTransport: SonosMockService = SonosMockService('AVTransport', ip_address)
        mock_soco.renderingControl: SonosMockService = SonosMockService('RenderingControl', ip_address)
        mock_soco.zoneGroupTopology: SonosMockService = SonosMockService('ZoneGroupTopology', ip_address)
        mock_soco.contentDirectory: SonosMockService = SonosMockService('ContentDirectory', ip_address)
        mock_soco.deviceProperties: SonosMockService = SonosMockService('DeviceProperties', ip_address)
        mock_soco.alarmClock: SonosMockService = self.alarm_clock
        mock_soco.get_battery_info.return_value = self.battery_info
        mock_soco.all_zones: set[MockSoCo] = {mock_soco}
        mock_soco.group.coordinator: MockSoCo = mock_soco
        self.mock_list[ip_address] = mock_soco
        return mock_soco

    def get_mock(self, *args: str) -> MockSoCo:
        """Return a mock."""
        ip_address: str = args[0] if len(args) > 0 else '192.168.42.2'
        if ip_address in self.mock_list:
            return self.mock_list[ip_address]
        mock_soco: MockSoCo = MockSoCo(name=f'Soco Mock {ip_address}')
        self.cache_mock(mock_soco, ip_address)
        return mock_soco

def patch_gethostbyname(host: str) -> str:
    """Mock to return host name as ip address for testing."""
    return host

@pytest.fixture(name='soco_sharelink')
def soco_sharelink() -> Generator[MagicMock, None, None]:
    """Fixture to mock soco.plugins.sharelink.ShareLinkPlugin."""
    with patch('homeassistant.components.sonos.speaker.ShareLinkPlugin') as mock_share:
        mock_instance: MagicMock = MagicMock()
        mock_instance.is_share_link.return_value = True
        mock_instance.add_share_link_to_queue.return_value = 10
        mock_share.return_value = mock_instance
        yield mock_instance

@pytest.fixture(name='sonos_websocket')
def sonos_websocket() -> Generator[AsyncMock, None, None]:
    """Fixture to mock SonosWebSocket."""
    with patch('homeassistant.components.sonos.speaker.SonosWebsocket') as mock_sonos_ws:
        mock_instance: AsyncMock = AsyncMock()
        mock_instance.play_clip = AsyncMock()
        mock_instance.play_clip.return_value = [{'success': 1}, {}]
        mock_sonos_ws.return_value = mock_instance
        yield mock_instance

@pytest.fixture(name='soco_factory')
def soco_factory(
    music_library: MagicMock,
    speaker_info: Dict[str, Any],
    current_track_info_empty: Dict[str, Any],
    battery_info: Dict[str, Any],
    alarm_clock: SonosMockService,
    sonos_playlists: SearchResult,
    sonos_websocket: AsyncMock,
    sonos_queue: List[DidlMusicTrack]
) -> Generator[SoCoMockFactory, None, None]:
    """Create factory for instantiating SoCo mocks."""
    factory: SoCoMockFactory = SoCoMockFactory(
        music_library,
        speaker_info,
        current_track_info_empty,
        battery_info,
        alarm_clock,
        sonos_playlists,
        sonos_queue=sonos_queue
    )
    with patch('homeassistant.components.sonos.SoCo', new=factory.get_mock), \
         patch('socket.gethostbyname', side_effect=patch_gethostbyname), \
         patch('homeassistant.components.sonos.ZGS_SUBSCRIPTION_TIMEOUT', 0):
        yield factory

@pytest.fixture(name='soco')
def soco_fixture(soco_factory: SoCoMockFactory) -> MockSoCo:
    """Create a default mock soco SoCo fixture."""
    return soco_factory.get_mock()

@pytest.fixture(autouse=True)
def silent_ssdp_scanner() -> Generator[None, None, None]:
    """Start SSDP component and get Scanner, prevent actual SSDP traffic."""
    with patch('homeassistant.components.ssdp.Scanner._async_start_ssdp_listeners'), \
         patch('homeassistant.components.ssdp.Scanner._async_stop_ssdp_listeners'), \
         patch('homeassistant.components.ssdp.Scanner.async_scan'), \
         patch('homeassistant.components.ssdp.Server._async_start_upnp_servers'), \
         patch('homeassistant.components.ssdp.Server._async_stop_upnp_servers'):
        yield

@pytest.fixture(name='discover', autouse=True)
def discover_fixture(soco: MockSoCo) -> Generator[Mock, None, None]:
    """Create a mock soco discover fixture."""

    def do_callback(
        hass: HomeAssistant,
        callback: Callable,
        match_dict: Optional[Dict[str, Any]] = None
    ) -> MagicMock:
        callback(
            SsdpServiceInfo(
                ssdp_location=f'http://{soco.ip_address}/',
                ssdp_st='urn:schemas-upnp-org:device:ZonePlayer:1',
                ssdp_usn=f'uuid:{soco.uid}_MR::urn:schemas-upnp-org:service:GroupRenderingControl:1',
                upnp={ATTR_UPNP_UDN: f'uuid:{soco.uid}'}
            ),
            ssdp.SsdpChange.ALIVE
        )
        return MagicMock()
    with patch('homeassistant.components.ssdp.async_register_callback', side_effect=do_callback) as mock:
        yield mock

@pytest.fixture(name='config')
def config_fixture() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Create hass config fixture."""
    return {DOMAIN: {MP_DOMAIN: {CONF_HOSTS: ['192.168.42.2']}}}

@pytest.fixture(name='sonos_favorites')
def sonos_favorites_fixture() -> SearchResult:
    """Create sonos favorites fixture."""
    favorites: List[Dict[str, Any]] = load_json_value_fixture('sonos_favorites.json', 'sonos')
    favorite_list: List[DidlFavorite] = [DidlFavorite.from_dict(fav) for fav in favorites]
    return SearchResult(favorite_list, 'favorites', 3, 3, 1)

@pytest.fixture(name='sonos_playlists')
def sonos_playlists_fixture() -> SearchResult:
    """Create sonos playlist fixture."""
    playlists: List[Dict[str, Any]] = load_json_value_fixture('sonos_playlists.json', 'sonos')
    playlists_list: List[DidlPlaylistContainer] = [DidlPlaylistContainer.from_dict(pl) for pl in playlists]
    return SearchResult(playlists_list, 'sonos_playlists', 1, 1, 0)

@pytest.fixture(name='sonos_queue')
def sonos_queue() -> List[DidlMusicTrack]:
    """Create sonos queue fixture."""
    queue: List[Dict[str, Any]] = load_json_value_fixture('sonos_queue.json', 'sonos')
    return [DidlMusicTrack.from_dict(track) for track in queue]

class MockMusicServiceItem:
    """Mocks a Soco MusicServiceItem."""

    def __init__(
        self,
        title: str,
        item_id: str,
        parent_id: str,
        item_class: str,
        album_art_uri: Optional[str] = None
    ) -> None:
        """Initialize the mock item."""
        self.title: str = title
        self.item_id: str = item_id
        self.item_class: str = item_class
        self.parent_id: str = parent_id
        self.album_art_uri: Optional[str] = album_art_uri

def list_from_json_fixture(file_name: str) -> List[MockMusicServiceItem]:
    """Create a list of music service items from a json fixture file."""
    item_list: List[Dict[str, Any]] = load_json_value_fixture(file_name, 'sonos')
    return [
        MockMusicServiceItem(
            item.get('title'),
            item.get('item_id'),
            item.get('parent_id'),
            item.get('item_class'),
            item.get('album_art_uri')
        ) for item in item_list
    ]

def mock_browse_by_idstring(
    search_type: str,
   